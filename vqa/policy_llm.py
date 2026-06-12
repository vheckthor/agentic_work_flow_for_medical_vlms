"""
Implemented by me

Policy for choosing the next experiment (JSON output).
**Hugging Face** — load with `transformers` + weights from the Hub:

  export EXPERIMENT_AGENT_HF_MODEL=gemma-4-4B-it
"""

import json
import os
import random
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agentic_machine_learning_experiment.vqa.experiment_config import (
    MAX_LEARNING_RATE,
    MAX_WEIGHT_DECAY,
    MIN_LEARNING_RATE,
    MIN_TRAINING_EPOCHS,
    MIN_WEIGHT_DECAY,
    ExperimentChoice,
)

DEFAULT_HF_POLICY_MODEL = "google/gemma-4-E4B-it"

POLICY_HISTORY_MAX_LEN = 12


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _choice_from_parsed(parsed: Any, pending_ids: List[str]) -> ExperimentChoice:
    if not isinstance(parsed, dict):
        raise ValueError("policy JSON must be an object")
    eid = parsed.get("experiment_id")
    if eid not in pending_ids:
        raise ValueError(f"invalid experiment_id: {eid!r}")
    ep = parsed.get("epochs")
    epochs: Optional[int] = None
    if ep is not None:
        try:
            epochs = int(ep)
        except (TypeError, ValueError):
            epochs = None
    lr = parsed.get("learning_rate")
    learning_rate: Optional[float] = None
    if lr is not None:
        try:
            learning_rate = float(lr)
        except (TypeError, ValueError):
            learning_rate = None
    wd = parsed.get("weight_decay")
    weight_decay: Optional[float] = None
    if wd is not None:
        try:
            weight_decay = float(wd)
        except (TypeError, ValueError):
            weight_decay = None
    return ExperimentChoice(eid, epochs, learning_rate, weight_decay)


class ExperimentPolicy(ABC):
    @abstractmethod
    def choose_experiment(
        self,
        *,
        pending_ids: List[str],
        experiments_catalog: Dict[str, Dict[str, Any]],
        history: List[Dict[str, Any]],
        state_summary: Dict[str, Any],
    ) -> ExperimentChoice:
        """Return chosen experiment and optional hyperparameter overrides."""


class HeuristicPolicy(ExperimentPolicy):
    """Random choice (reproducible with AGENT_SEED)."""

    def __init__(self, seed: Optional[int] = None):
        if seed is None:
            seed = int(os.environ.get("AGENT_SEED", "42"))
        self._rng = random.Random(seed)

    def choose_experiment(
        self,
        *,
        pending_ids: List[str],
        experiments_catalog: Dict[str, Dict[str, Any]],
        history: List[Dict[str, Any]],
        state_summary: Dict[str, Any],
    ) -> ExperimentChoice:
        return ExperimentChoice(self._rng.choice(pending_ids))


class HFTransformersPolicy(ExperimentPolicy):
    """
    Local causal LM from Hugging Face (AutoModelForCausalLM + AutoTokenizer).
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        max_new_tokens: int = 256,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id = model_id or os.environ.get("EXPERIMENT_AGENT_HF_MODEL", DEFAULT_HF_POLICY_MODEL)
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        dtype = torch.float32
        if self._device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype if self._device.type != "cpu" else torch.float16,
            trust_remote_code=True
        )
        self.model.to(self._device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def choose_experiment(
        self,
        *,
        pending_ids: List[str],
        experiments_catalog: Dict[str, Dict[str, Any]],
        history: List[Dict[str, Any]],
        state_summary: Dict[str, Any],
    ) -> ExperimentChoice:
        import torch

        pending_subset = {k: experiments_catalog[k] for k in pending_ids if k in experiments_catalog}
        system = (
            "You are an expert in machine learning and a planner for machine learning experiments. Choose exactly one experiment_id from the pending set "
            "that is most informative next (explore diverse settings). "
            f"You may optionally set \"epochs\" (integer, at least {MIN_TRAINING_EPOCHS}), "
            f"\"learning_rate\" (float in [{MIN_LEARNING_RATE}, {MAX_LEARNING_RATE}]), and "
            f"\"weight_decay\" (float in [{MIN_WEIGHT_DECAY}, {MAX_WEIGHT_DECAY}]) for that run. "
            "Provide hyperparameters likely to perform well."
            "Respond ONLY with JSON: "
            "{\"experiment_id\": \"...\", \"epochs\": <optional int>, "
            "\"learning_rate\": <optional float>, \"weight_decay\": <optional float>}"
        )
        user = json.dumps(
            {
                "state": state_summary,
                "history": history[-POLICY_HISTORY_MAX_LEN:],
                "pending_experiments": pending_subset,
            },
            indent=2,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        text = self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()
        parsed = _extract_json_object(text)
        if not parsed:
            raise ValueError(f"HF policy returned unparseable JSON: {text!r}")
        return _choice_from_parsed(parsed, pending_ids)

#select the llm policy to use either huggingface or heuristic
def make_policy(
    kind: str = "auto",
    *,
    hf_model: Optional[str] = None,
):
    """
    kind:
      'hf' — Hugging Face causal LM (default small gemma-4-31B-it).
      'heuristic' — random among pending.
    """
    kind = (kind or "auto").lower()
    if kind == "heuristic":
        return HeuristicPolicy()
    if kind == "hf":
        mid = hf_model or os.environ.get("EXPERIMENT_AGENT_HF_MODEL", DEFAULT_HF_POLICY_MODEL)
        return HFTransformersPolicy(model_id=mid)
    if kind == "auto":
        try:
            return HFTransformersPolicy(
                model_id=hf_model or os.environ.get("EXPERIMENT_AGENT_HF_MODEL", DEFAULT_HF_POLICY_MODEL)
            )
        except Exception:
            return HeuristicPolicy()
    raise ValueError(f"Unknown policy kind: {kind}")
