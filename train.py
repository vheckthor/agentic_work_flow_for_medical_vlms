#!/usr/bin/env python3
"""
implemented by me
Train SLAKE closed-set VQA classifiers (fine-tuned; not zero-shot).

Expects under ``--data_root``: ``train.json``, ``validation.json`` (unless ``--no-early-stopping``),
``test.json`` (for evaluate.py), and ``imgs/``. On CUDA, mixed precision and gradient checkpointing are automatic.

Variants:
  vit_bert       — ViT + BERT fusion MLP
  resnet_bert    — ResNet-50 + BERT fusion MLP
  clip           — CLIP dual encoders + MLP head (fine-tuned end-to-end)
  cross_attn     — ViT patch sequence + BERT query cross-attention
  vit_bert_focal — same backbone as vit_bert + training-time augmentation + focal loss

Early stopping (default on): validation balanced accuracy on ``validation.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import agentic_machine_learning_experiment.vqa as vqa

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, CLIPProcessor

from agentic_machine_learning_experiment.vqa.dataset import SlakeVQADataset, build_answer_vocab, default_eval_transform, default_train_transform
from agentic_machine_learning_experiment.vqa.early_stopping import EarlyStopping
from agentic_machine_learning_experiment.vqa.metrics import gather_predictions, metrics_from_logits
from agentic_machine_learning_experiment.vqa.models import (
    CLIPFinetunedClassifier,
    ResNetBERTClassifier,
    ViTBERTClassifier,
    ViTBERTCrossAttentionClassifier,
    focal_loss,
)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(variant: str, num_classes: int, clip_name: str):
    if variant == "vit_bert" or variant == "vit_bert_focal":
        return ViTBERTClassifier(num_classes)
    if variant == "resnet_bert":
        return ResNetBERTClassifier(num_classes)
    if variant == "clip":
        return CLIPFinetunedClassifier(num_classes, clip_name=clip_name)
    if variant == "cross_attn":
        return ViTBERTCrossAttentionClassifier(num_classes)
    raise ValueError(f"Unknown variant {variant}")

#Gradient checkpointing is a technique that reduces memory usage by trading compute time
#This is useful for training large models on limited hardware
def enable_gradient_checkpointing(model: torch.nn.Module, variant: str) -> None:
    """Reduce activation memory on Hugging Face backbones (more compute)."""
    if variant in ("vit_bert", "vit_bert_focal", "cross_attn"):
        if hasattr(model.vit, "gradient_checkpointing_enable"):
            model.vit.gradient_checkpointing_enable()
    if variant in ("vit_bert", "vit_bert_focal", "resnet_bert", "cross_attn"):
        if hasattr(model, "bert") and hasattr(model.bert, "gradient_checkpointing_enable"):
            model.bert.gradient_checkpointing_enable()
    if variant == "clip" and hasattr(model.clip, "gradient_checkpointing_enable"):
        model.clip.gradient_checkpointing_enable()


def _set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for param in module.parameters():
        param.requires_grad = value


def _unfreeze_last_n_layers(layers, n: int) -> None:
    if n <= 0:
        return
    for layer in layers[-n:]:
        _set_requires_grad(layer, True)


#Partial finetuning is a technique that freezes the backbone of a model and only trains selected layers

def configure_partial_finetuning(model: torch.nn.Module, variant: str, last_n_layers: int) -> None:
    """Freeze backbones and keep only task head + last N backbone layers trainable."""
    _set_requires_grad(model, False)

    if hasattr(model, "head"):
        _set_requires_grad(model.head, True)
    if variant == "cross_attn":
        _set_requires_grad(model.proj_t, True)
        _set_requires_grad(model.attn, True)
        _set_requires_grad(model.norm, True)
        _set_requires_grad(model.pool, True)

    if variant in ("vit_bert", "vit_bert_focal", "cross_attn"):
        vit_layers = list(model.vit.encoder.layer)
        _unfreeze_last_n_layers(vit_layers, last_n_layers)
    if variant in ("vit_bert", "vit_bert_focal", "resnet_bert", "cross_attn"):
        bert_layers = list(model.bert.encoder.layer)
        _unfreeze_last_n_layers(bert_layers, last_n_layers)
    if variant == "resnet_bert":
        resnet_stages = [stage for name, stage in model.cnn.named_children() if name.startswith("layer")]
        if resnet_stages:
            _unfreeze_last_n_layers(resnet_stages, last_n_layers)
    if variant == "clip":
        vision_layers = list(model.clip.vision_model.encoder.layers)
        text_layers = list(model.clip.text_model.encoder.layers)
        _unfreeze_last_n_layers(vision_layers, last_n_layers)
        _unfreeze_last_n_layers(text_layers, last_n_layers)
        if hasattr(model.clip, "visual_projection"):
            _set_requires_grad(model.clip.visual_projection, True)
        if hasattr(model.clip, "text_projection"):
            _set_requires_grad(model.clip.text_projection, True)


def collect_trainable_layer_info(model: torch.nn.Module) -> tuple[int, list[str]]:
    """Return count + names of modules that own at least one trainable parameter tensor."""
    layer_names: list[str] = []
    for name, module in model.named_modules():
        if any(p.requires_grad for p in module.parameters(recurse=False)):
            layer_names.append(name or "<root>")
    return len(layer_names), layer_names


#Created so the file can run as a script with proper arg parsing
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="SLAKE")
    p.add_argument(
        "--variant",
        type=str,
        default="vit_bert",
        choices=["vit_bert", "resnet_bert", "clip", "cross_attn", "vit_bert_focal"],
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--last_n_trainable_layers",
        type=int,
        default=2,
        help="Train only classifier/fusion modules plus the last N backbone layers.",
    )
    p.add_argument(
        "--no-early-stopping",
        dest="no_early_stopping",
        action="store_true",
        default=False,
        help="Train on train.json only (no validation monitoring).",
    )
    p.add_argument("--early_stopping_patience", type=int, default=3)
    args = p.parse_args()

    root = Path(args.data_root)
    train_path = root / "train.json"
    val_path = root / "validation.json"
    imgs_root = str(root / "imgs")

    train_rows = load_json(str(train_path))
    answer_to_idx, idx_to_answer = build_answer_vocab(train_rows)
    num_classes = len(answer_to_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    augment = args.variant == "vit_bert_focal"
    use_focal = args.variant == "vit_bert_focal"

    clip_processor = None
    tokenizer = None
    if args.variant == "clip":
        clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
        train_tf = default_eval_transform()
        val_tf = default_eval_transform()
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        train_tf = default_train_transform(augment=augment)
        val_tf = default_eval_transform()

    train_ds = SlakeVQADataset(
        str(train_path),
        imgs_root,
        answer_to_idx,
        tokenizer,
        train_tf,
        clip_processor=clip_processor,
        max_question_length=77 if args.variant == "clip" else 64,
    )

    val_loader = None
    use_es = False
    if not args.no_early_stopping:
        if not val_path.is_file():
            raise FileNotFoundError(
                f"Early stopping requires {val_path} (SLAKE layout: train.json, validation.json, test.json)."
            )
        val_ds = SlakeVQADataset(
            str(val_path),
            imgs_root,
            answer_to_idx,
            tokenizer,
            val_tf,
            clip_processor=clip_processor,
            max_question_length=77 if args.variant == "clip" else 64,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        use_es = True

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(args.variant, num_classes, args.clip_model).to(device)
    configure_partial_finetuning(model, args.variant, args.last_n_trainable_layers)
    enable_gradient_checkpointing(model, args.variant)

    use_amp = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp and amp_dtype == torch.float16,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters selected. Increase --last_n_trainable_layers.")
    optim = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    trainable_tensors_count = len(trainable_params)
    trainable_layer_count, trainable_layer_names = collect_trainable_layer_info(model)
    print(
        f"Partial fine-tuning enabled: {trainable_count}/{total_count} parameters trainable "
        f"(last_n_trainable_layers={args.last_n_trainable_layers})."
    )

    out_dir = Path(args.out_dir) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "variant": args.variant,
        "num_classes": num_classes,
        "answer_to_idx": answer_to_idx,
        "idx_to_answer": idx_to_answer,
        "clip_model": args.clip_model if args.variant == "clip" else None,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    early_stop = (
        EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=1e-6,
            mode="max",
        )
        if use_es
        else None
    )
    best_state: dict | None = None
    stopped_early = False
    epochs_ran = 0

    t_train0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        bar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in bar:
            labels = batch["labels"].to(device)
            valid = labels >= 0
            if not valid.any():
                continue
            optim.zero_grad(set_to_none=True)
            pixel = batch["pixel_values"].to(device)[valid]
            ids = batch["input_ids"].to(device)[valid]
            am = batch["attention_mask"].to(device)[valid]
            y = labels[valid]
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(pixel, ids, am)
                    if use_focal:
                        loss = focal_loss(logits, y, gamma=2.0)
                    else:
                        loss = torch.nn.functional.cross_entropy(logits, y)
            else:
                logits = model(pixel, ids, am)
                if use_focal:
                    loss = focal_loss(logits, y, gamma=2.0)
                else:
                    loss = torch.nn.functional.cross_entropy(logits, y)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
            total += float(loss.detach().item()) * y.size(0)
            n += y.size(0)
            bar.set_postfix(loss=f"{total/max(n,1):.4f}")

        epochs_ran = epoch + 1

        if val_loader is not None:
            logits, labels, mask = gather_predictions(model, val_loader, device)
            m = metrics_from_logits(logits, labels, mask)
            print(f"val {m}")
            metric = float(m.get("balanced_accuracy", m.get("accuracy", 0.0)))

            if early_stop is not None:
                stop = early_stop.step(
                    epoch + 1,
                    metric,
                    metric_name="val_balanced_accuracy",
                    verbose=True,
                )
                if early_stop.last_improved:
                    best_state = copy.deepcopy(model.state_dict())
                if stop:
                    stopped_early = True
                    break

    if use_es and best_state is not None:
        model.load_state_dict(best_state)

    t_train1 = time.perf_counter()
    train_seconds = t_train1 - t_train0

    ckpt = {
        "model_state_dict": model.state_dict(),
        "variant": args.variant,
        "num_classes": num_classes,
        "train_seconds": train_seconds,
    }
    torch.save(ckpt, out_dir / "model.pt")
    timing_path = out_dir / "train_timing.json"
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_seconds": train_seconds,
                "epochs": args.epochs,
                "epochs_ran": epochs_ran,
                "stopped_early": stopped_early,
                "early_stopping": use_es,
                "cuda_amp": use_amp,
                "optimizer": {
                    "name": type(optim).__name__,
                    "learning_rate": args.lr,
                    "weight_decay": args.weight_decay,
                },
                "fine_tuning": {
                    "strategy": "last_n_layers",
                    "last_n_trainable_layers": args.last_n_trainable_layers,
                    "trainable_layers_count": trainable_layer_count,
                    "trainable_layer_names": trainable_layer_names,
                    "trainable_parameter_tensors_count": trainable_tensors_count,
                    "trainable_parameters_count": trainable_count,
                    "total_parameters_count": total_count,
                    "trainable_parameters_ratio": (trainable_count / total_count) if total_count else 0.0,
                },
                "training_args": {
                    "variant": args.variant,
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                },
            },
            f,
            indent=2,
        )
    print(f"Saved {out_dir / 'model.pt'}; train time {train_seconds:.1f}s (epochs_ran={epochs_ran})")


if __name__ == "__main__":
    main()
