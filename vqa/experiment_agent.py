#!/usr/bin/env python3
"""
LangGraph agent that schedules SLAKE experiments (classifier train.py + evaluate.py).
Implemented by me
Augmented with Cursor IDE 

**Policy default: Hugging Face** — loads a small instruct model with `transformers` (no separate server):

  .venv/bin/python experiment_agent.py --max-steps 5 --policy hf

  --policy hf         — local causal LM from the Hub (default: Qwen2.5-0.5B-Instruct).
  --hf-model Qwen/Qwen2.5-1.5B-Instruct  — override; or set EXPERIMENT_AGENT_HF_MODEL.
"""

from __future__ import annotations

import argparse
import json
import os
import operator
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import agentic_machine_learning_experiment.vqa as vqa 

from langgraph.graph import END, START, StateGraph

from agentic_machine_learning_experiment.vqa.experiment_config import (
    MIN_TRAINING_EPOCHS,
    clamp_epochs,
    clamp_learning_rate,
    clamp_weight_decay,
)
from agentic_machine_learning_experiment.vqa.policy_llm import DEFAULT_HF_POLICY_MODEL, HeuristicPolicy, make_policy


class AgentStateV2(TypedDict, total=False):
    step: int
    action: str
    experiment_id: str
    epoch_override: Optional[int]
    learning_rate_override: Optional[float]
    weight_decay_override: Optional[float]
    performance: float
    running_time: float
    error: str
    run_report: Optional[Dict[str, Any]]
    training_state: Dict[str, Any]
    resolved_config: Dict[str, Any]
    history: Annotated[List[Dict[str, Any]], operator.add]


@dataclass
class ExperimentEnv:
    """Runs real training + eval subprocesses under repo root."""

    repo_root: Path
    data_root: str = "SLAKE"
    dry_run: bool = False
    early_stopping: bool = True
    early_stopping_patience: int = 3

    def __post_init__(self):
        self.repo_root = self.repo_root.resolve()
        self.py = sys.executable

    def run(self, exp_id: str, cfg: Dict[str, Any]) -> tuple[float, float, Optional[Dict[str, Any]]]:
        t0 = time.perf_counter()
        if self.dry_run:
            return 0.0, 0.0, None
        family = cfg.get("family", "classifier")
        if family != "classifier":
            raise ValueError(f"Unknown family {family}")
        score = self._run_classifier(exp_id, cfg)
        report_path = self.repo_root / "checkpoints" / "agent" / exp_id / cfg["variant"] / "run_report_test.json"
        elapsed = time.perf_counter() - t0
        report: Optional[Dict[str, Any]] = None
        if report_path.is_file():
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
        return float(score), float(elapsed), report

    def _run_classifier(self, exp_id: str, cfg: Dict[str, Any]) -> float:
        out_base = self.repo_root / "checkpoints" / "agent" / exp_id
        out_base.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.py,
            str(self.repo_root / "train.py"),
            "--data_root",
            self.data_root,
            "--variant",
            cfg["variant"],
            "--epochs",
            str(cfg.get("epochs", 1)),
            "--batch_size",
            str(cfg.get("batch_size", 4)),
            "--lr",
            str(cfg.get("learning_rate", 1e-1)),
            "--weight_decay",
            str(cfg.get("weight_decay", 0.1)),
            "--last_n_trainable_layers",
            str(cfg.get("last_n_trainable_layers", 2)),
            "--out_dir",
            str(out_base),
            "--num_workers",
            "0",
        ]
        cmd += self._early_stopping_cli()
        print(' '.join(cmd))
        subprocess.run(cmd, cwd=str(self.repo_root), check=True)
        ckpt = out_base / cfg["variant"] / "model.pt"
        ev = [
            self.py,
            str(self.repo_root / "evaluate.py"),
            "--data_root",
            self.data_root,
            "--checkpoint",
            str(ckpt),
            "--split",
            "test",
            "--batch_size",
            str(cfg.get("eval_batch_size", 8)),
            "--num_workers",
            "0",
        ]
        subprocess.run(ev, cwd=str(self.repo_root), check=True)
        metrics_path = out_base / cfg["variant"] / "metrics_test.json"
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        return float(m.get("accuracy", m.get("balanced_accuracy", 0.0)))

    def _early_stopping_cli(self) -> List[str]:
        if self.early_stopping:
            return [
                "--early_stopping_patience",
                str(self.early_stopping_patience),
            ]
        return ["--no-early-stopping"]


def build_experiment_catalog(epochs: int) -> Dict[str, Dict[str, Any]]:
    """Ordered grid: classifier baselines."""
    ne = clamp_epochs(epochs)
    cat: Dict[str, Dict[str, Any]] = {
        # "cls_vit_bert": {
        #     "family": "classifier",
        #     "variant": "vit_bert",
        #     "epochs": ne,
        #     "batch_size": 2,
        #     "learning_rate": 2e-5,
        #     "weight_decay": 1e-2,
        #     "last_n_trainable_layers": 2,
        # },
        "cls_resnet_bert": {
            "family": "classifier",
            "variant": "resnet_bert",
            "epochs": ne,
            "batch_size": 2,
            "learning_rate": 2e-5,
            "weight_decay": 1e-2,
            "last_n_trainable_layers": 2,
        },
        "cls_clip": {
            "family": "classifier",
            "variant": "clip",
            "epochs": ne,
            "batch_size": 2,
            "learning_rate": 2e-5,
            "weight_decay": 1e-2,
            "last_n_trainable_layers": 2,
        },
        "cls_cross_attn": {
            "family": "classifier",
            "variant": "cross_attn",
            "epochs": ne,
            "batch_size": 2,
            "learning_rate": 2e-5,
            "weight_decay": 1e-2,
            "last_n_trainable_layers": 2,
        },
    }
    return cat


class ExperimentAgentGraph:
    def __init__(
        self,
        env: ExperimentEnv,
        experiments: Dict[str, Dict[str, Any]],
        policy: Any
    ):
        self.env = env
        self.experiments = experiments
        self.policy = policy

        graph = StateGraph(AgentStateV2)
        graph.add_node("perceive", self._perceive)
        graph.add_node("select_action", self._select_action)
        graph.add_node("run_experiment", self._run_experiment)
        graph.add_node("update_history", self._update_history)

        graph.add_edge(START, "perceive")
        graph.add_edge("perceive", "select_action")
        graph.add_edge("select_action", "run_experiment")
        graph.add_edge("run_experiment", "update_history")
        graph.add_edge("update_history", END)

        self.graph = graph.compile()

    def _perceive(self, state: AgentStateV2) -> Dict[str, Any]:
        return {}

    @staticmethod
    def _training_state_from_run(cfg: Dict[str, Any], report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        training = (report or {}).get("training", {}) if isinstance(report, dict) else {}
        optimizer = training.get("optimizer", {})
        fine_tuning = training.get("fine_tuning", {})
        return {
            "optimizer": {
                "name": optimizer.get("name", "AdamW"),
                "learning_rate": optimizer.get("learning_rate", cfg.get("learning_rate")),
                "weight_decay": optimizer.get("weight_decay", cfg.get("weight_decay")),
            },
            "fine_tuning": {
                "strategy": fine_tuning.get("strategy", "last_n_layers"),
                "last_n_trainable_layers": fine_tuning.get("last_n_trainable_layers", cfg.get("last_n_trainable_layers", 2)),
                "trainable_layers_count": fine_tuning.get("trainable_layers_count"),
                "trainable_parameter_tensors_count": fine_tuning.get("trainable_parameter_tensors_count"),
                "trainable_parameters_count": fine_tuning.get("trainable_parameters_count"),
                "total_parameters_count": fine_tuning.get("total_parameters_count"),
                "trainable_parameters_ratio": fine_tuning.get("trainable_parameters_ratio"),
            },
            "epochs_requested": training.get("epochs_requested", cfg.get("epochs")),
            "epochs_ran": training.get("epochs_ran"),
            "stopped_early": training.get("stopped_early"),
        }

    def _done_ids(self, state: AgentStateV2) -> set:
        # Only successful runs are considered complete so failures can be retried on resume.
        return {
            h["experiment_id"]
            for h in state.get("history", [])
            if h.get("experiment_id") and not h.get("error")
        }

    def _select_action(self, state: AgentStateV2) -> Dict[str, Any]:
        done = self._done_ids(state)
        pending = [eid for eid in self.experiments if eid not in done]
        if not pending:
            return {"action": "none", "experiment_id": ""}

        chosen: Optional[str] = None
        epoch_override: Optional[int] = None
        learning_rate_override: Optional[float] = None
        weight_decay_override: Optional[float] = None

        if chosen is None:
            try:
                choice = self.policy.choose_experiment(
                    pending_ids=pending,
                    experiments_catalog=self.experiments,
                    history=state.get("history", []),
                    state_summary={"step": state.get("step", 0)},
                )
            except Exception as e:
                print(f"[select_action] policy failed ({e}), using heuristic", file=sys.stderr)
                choice = HeuristicPolicy().choose_experiment(
                    pending_ids=pending,
                    experiments_catalog=self.experiments,
                    history=state.get("history", []),
                    state_summary={"step": state.get("step", 0)},
                )
            chosen = choice.experiment_id
            if choice.epochs is not None:
                epoch_override = clamp_epochs(choice.epochs)
            if choice.learning_rate is not None:
                learning_rate_override = clamp_learning_rate(choice.learning_rate)
            if choice.weight_decay is not None:
                weight_decay_override = clamp_weight_decay(choice.weight_decay)

        print(
            "[select_action] "
            f"experiment_id={chosen} "
            f"epoch_override={epoch_override} "
            f"learning_rate_override={learning_rate_override} "
            f"weight_decay_override={weight_decay_override}"
        )
        return {
            "action": "run_experiment",
            "experiment_id": chosen,
            "epoch_override": epoch_override,
            "learning_rate_override": learning_rate_override,
            "weight_decay_override": weight_decay_override,
        }

    def _run_experiment(self, state: AgentStateV2) -> Dict[str, Any]:
        if state.get("action") != "run_experiment":
            return {}
        eid = state.get("experiment_id", "")
        if not eid:
            return {}
        cfg = {**self.experiments[eid]}
        eo = state.get("epoch_override")
        lro = state.get("learning_rate_override")
        wdo = state.get("weight_decay_override")
        if eo is not None:
            cfg["epochs"] = clamp_epochs(int(eo))
        else:
            cfg["epochs"] = clamp_epochs(int(cfg.get("epochs", MIN_TRAINING_EPOCHS)))
        if lro is not None:
            cfg["learning_rate"] = clamp_learning_rate(float(lro))
        else:
            cfg["learning_rate"] = clamp_learning_rate(float(cfg.get("learning_rate", 2e-5)))
        if wdo is not None:
            cfg["weight_decay"] = clamp_weight_decay(float(wdo))
        else:
            cfg["weight_decay"] = clamp_weight_decay(float(cfg.get("weight_decay", 0.1)))
        try:
            perf, rt, report = self.env.run(eid, cfg)
            training_state = self._training_state_from_run(cfg, report)
            return {
                "performance": perf,
                "running_time": rt,
                "error": "",
                "run_report": report,
                "training_state": training_state,
                "resolved_config": cfg,
            }
        except subprocess.CalledProcessError as e:
            err = f"subprocess failed: {e}"
            print(err, file=sys.stderr)
            return {
                "performance": 0.0,
                "running_time": 0.0,
                "error": err,
                "run_report": None,
                "training_state": self._training_state_from_run(cfg, None),
                "resolved_config": cfg,
            }
        except Exception as e:
            err = str(e)
            print(err, file=sys.stderr)
            return {
                "performance": 0.0,
                "running_time": 0.0,
                "error": err,
                "run_report": None,
                "training_state": self._training_state_from_run(cfg, None),
                "resolved_config": cfg,
            }

    def _update_history(self, state: AgentStateV2) -> Dict[str, Any]:
        if state.get("action") != "run_experiment":
            return {}
        eid = state.get("experiment_id", "")
        if not eid:
            return {}
        rec = {
            "experiment_id": eid,
            "config": state.get("resolved_config", self.experiments.get(eid, {})),
            "performance": state.get("performance", 0.0),
            "running_time": state.get("running_time", 0.0),
            "error": state.get("error", ""),
            "run_report": state.get("run_report"),
            "training_state": state.get("training_state", {}),
        }
        return {"history": [rec]}


def parse_args():
    p = argparse.ArgumentParser(description="LangGraph SLAKE experiment agent")
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--data-root", type=str, default="SLAKE")
    p.add_argument("--dry-run", action="store_true", help="No training; policy still selects ids")
    p.add_argument(
        "--policy",
        type=str,
        default="hf",
        choices=["hf", "auto", "heuristic", "openai"],
        help="Default hf: Hugging Face causal LM (see --hf-model)",
    )
    p.add_argument(
        "--hf-model",
        type=str,
        default="",
        help="HF model id for --policy hf (default: Qwen2.5-0.5B-Instruct or EXPERIMENT_AGENT_HF_MODEL)",
    )
    p.add_argument(
        "--log-json",
        type=str,
        default="experiment_runs.json",
        help="Append final history JSON to this file",
    )
    p.add_argument(
        "--restart",
        action="store_true",
        help="Start from scratch and ignore any previous history in --log-json.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=MIN_TRAINING_EPOCHS,
        help=f"Default training epochs for catalog runs (minimum {MIN_TRAINING_EPOCHS}).",
    )
    p.add_argument(
        "--no-early-stopping",
        dest="no_early_stopping",
        action="store_true",
        default=False,
        help="Disable validation monitoring in train.py.",
    )
    p.add_argument("--early-stopping-patience", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parent
    default_epochs = clamp_epochs(args.epochs)
    experiments = build_experiment_catalog(epochs=default_epochs)
    env = ExperimentEnv(
        repo_root=repo,
        data_root=args.data_root,
        dry_run=args.dry_run,
        early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
    )

    hf_kw = {}
    if args.hf_model:
        hf_kw["hf_model"] = args.hf_model
    policy = make_policy(args.policy, **hf_kw)
    if args.policy == "hf":
        mid = args.hf_model or os.environ.get("EXPERIMENT_AGENT_HF_MODEL", DEFAULT_HF_POLICY_MODEL)
        print(f"[policy] Hugging Face  model={mid}")
    elif args.policy == "openai":
        print("[policy] OpenAI-compatible API (OPENAI_API_KEY)")
    elif args.policy == "auto":
        print("[policy] auto (OPENAI_API_KEY or HF or heuristic)")
    elif args.policy == "heuristic":
        print("[policy] heuristic")

    agent = ExperimentAgentGraph(env=env, experiments=experiments, policy=policy)

    history: List[Dict[str, Any]] = []
    if args.restart:
        print("[resume] --restart set; starting from empty history")
    elif args.log_json:
        log_path = Path(args.log_json)
        if not log_path.is_absolute():
            log_path = repo / log_path
        if log_path.is_file():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                loaded = prev.get("history", [])
                if isinstance(loaded, list):
                    history = loaded
                    ok_count = sum(1 for h in history if not h.get("error"))
                    fail_count = len(history) - ok_count
                    print(
                        f"[resume] loaded {len(history)} previous runs "
                        f"(ok={ok_count}, failed={fail_count}) from {log_path}"
                    )
            except Exception as e:
                print(f"[resume] could not load {log_path}: {e}", file=sys.stderr)

    state: AgentStateV2 = {"history": history, "step": 0}
    for i in range(args.max_steps):
        state["step"] = i + 1
        print(f"\n=== Outer step {state['step']} ===")
        state = agent.graph.invoke(state)
        if state.get("action") == "none":
            print("No pending experiments left.")
            break
        print(
            json.dumps(
                {
                    "experiment_id": state.get("experiment_id"),
                    "performance": state.get("performance"),
                    "running_time": state.get("running_time"),
                },
                indent=2,
            )
        )

    hist = state.get("history", [])
    overall_training_summary = [
        {
            "experiment_id": h.get("experiment_id"),
            "optimizer": (h.get("training_state") or {}).get("optimizer", {}),
            "fine_tuning": (h.get("training_state") or {}).get("fine_tuning", {}),
            "epochs_requested": (h.get("training_state") or {}).get("epochs_requested"),
            "epochs_ran": (h.get("training_state") or {}).get("epochs_ran"),
            "stopped_early": (h.get("training_state") or {}).get("stopped_early"),
        }
        for h in hist
    ]
    if hist:
        best = max(
            hist,
            key=lambda r: (r.get("performance", 0.0), -float(r.get("running_time", 0.0))),
        )
        print("\nBest run:", json.dumps(best, indent=2))
    if args.log_json:
        with open(args.log_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "history": hist,
                    "catalog": experiments,
                    "overall_training_summary": overall_training_summary,
                },
                f,
                indent=2,
            )
        print("Wrote", args.log_json)


if __name__ == "__main__":
    main()
