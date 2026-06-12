#!/usr/bin/env python3
"""Evaluate a fine-tuned checkpoint on SLAKE test.json only (plus optional validation).
Implemented by me
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import agentic_machine_learning_experiment.vqa as vqa

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, CLIPProcessor

from agentic_machine_learning_experiment.vqa.dataset import SlakeVQADataset, default_eval_transform
from agentic_machine_learning_experiment.vqa.metrics import gather_predictions, metrics_from_logits
from agentic_machine_learning_experiment.vqa.run_report import write_classifier_run_report
from agentic_machine_learning_experiment.vqa.models import (
    CLIPFinetunedClassifier,
    ResNetBERTClassifier,
    ViTBERTClassifier,
    ViTBERTCrossAttentionClassifier,
)


def load_model(variant: str, num_classes: int, path: Path, clip_name: str, device: torch.device):
    if variant == "vit_bert" or variant == "vit_bert_focal":
        m = ViTBERTClassifier(num_classes)
    elif variant == "resnet_bert":
        m = ResNetBERTClassifier(num_classes)
    elif variant == "clip":
        m = CLIPFinetunedClassifier(num_classes, clip_name=clip_name)
    elif variant == "cross_attn":
        m = ViTBERTCrossAttentionClassifier(num_classes)
    else:
        raise ValueError(variant)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    m.load_state_dict(ckpt["model_state_dict"])
    return m.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="SLAKE")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model.pt")
    p.add_argument("--split", type=str, default="test", choices=["test", "validation"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    args = p.parse_args()

    root = Path(args.data_root)
    split_path = root / f"{args.split}.json"
    imgs_root = str(root / "imgs")
    ckpt_path = Path(args.checkpoint)
    meta_path = ckpt_path.parent / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    variant = meta["variant"]
    num_classes = meta["num_classes"]
    answer_to_idx = meta["answer_to_idx"]
    clip_name = meta.get("clip_model") or args.clip_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    model = load_model(variant, num_classes, ckpt_path, clip_name, device)

    if variant == "clip":
        clip_processor = CLIPProcessor.from_pretrained(clip_name)
        ds = SlakeVQADataset(
            str(split_path),
            imgs_root,
            answer_to_idx,
            None,
            None,
            clip_processor=clip_processor,
            skip_unknown_answers=False,
            max_question_length=77,
        )
    else:
        tok = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        ds = SlakeVQADataset(
            str(split_path),
            imgs_root,
            answer_to_idx,
            tok,
            default_eval_transform(),
            skip_unknown_answers=False,
        )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    t0 = time.perf_counter()
    logits, labels, mask = gather_predictions(model, loader, device)
    t1 = time.perf_counter()
    metrics = metrics_from_logits(logits, labels, mask)
    metrics["in_vocab_examples"] = int(mask.sum())
    metrics["total_examples"] = int(len(labels))
    metrics["test_inference_seconds"] = t1 - t0
    out = ckpt_path.parent / f"metrics_{args.split}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    report_path = write_classifier_run_report(ckpt_path, args.split)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
