"""
Unified experiment reports for thesis / paper tables.
Implemented by me
Augmented with Cursor IDE 

After ``evaluate.py``, each run writes a ``run_report_<split>.json``
next to ``metrics_*.json`` with training + evaluation + normalized keys for CSV aggregation.

See ``analyze_experiment_results.py`` for batch export and plots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

SCHEMA_VERSION = 1


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _compact_meta_classifier(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: meta[k] for k in ("variant", "num_classes", "clip_model") if k in meta}
    return out


def evaluation_flat_classifier(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Stable column names for CSV / analysis across classifier runs."""
    return {
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "accuracy": metrics.get("accuracy"),
        "f1_macro": metrics.get("f1_macro"),
        "f1_micro": metrics.get("f1_micro"),
        "mrr": metrics.get("mrr"),
        "ece": metrics.get("ece"),
        "inference_seconds": metrics.get("test_inference_seconds"),
        "in_vocab_examples": metrics.get("in_vocab_examples"),
        "total_examples": metrics.get("total_examples"),
    }


def build_classifier_run_report(ckpt_path: Path, split: str) -> Dict[str, Any]:
    parent = ckpt_path.parent
    meta = _read_json(parent / "meta.json")
    timing = _read_json(parent / "train_timing.json") or {}
    metrics_path = parent / f"metrics_{split}.json"
    metrics = _read_json(metrics_path)
    if not meta or not metrics:
        raise FileNotFoundError(
            f"Need meta.json and {metrics_path.name} under {parent}"
        )
    optimizer = timing.get("optimizer", {})
    fine_tuning = timing.get("fine_tuning", {})
    training_args = timing.get("training_args", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "family": "classifier",
        "variant": meta.get("variant"),
        "checkpoint_dir": str(parent.resolve()),
        "split": split,
        "meta": _compact_meta_classifier(meta),
        "training": {
            "train_seconds": timing.get("train_seconds"),
            "epochs_requested": timing.get("epochs"),
            "epochs_ran": timing.get("epochs_ran"),
            "stopped_early": timing.get("stopped_early"),
            "early_stopping": timing.get("early_stopping"),
            "optimizer": {
                "name": optimizer.get("name"),
                "learning_rate": optimizer.get("learning_rate"),
                "weight_decay": optimizer.get("weight_decay"),
            },
            "fine_tuning": {
                "strategy": fine_tuning.get("strategy"),
                "last_n_trainable_layers": fine_tuning.get("last_n_trainable_layers"),
                "trainable_layers_count": fine_tuning.get("trainable_layers_count"),
                "trainable_layer_names": fine_tuning.get("trainable_layer_names"),
                "trainable_parameter_tensors_count": fine_tuning.get("trainable_parameter_tensors_count"),
                "trainable_parameters_count": fine_tuning.get("trainable_parameters_count"),
                "total_parameters_count": fine_tuning.get("total_parameters_count"),
                "trainable_parameters_ratio": fine_tuning.get("trainable_parameters_ratio"),
            },
            "training_args": {
                "variant": training_args.get("variant"),
                "batch_size": training_args.get("batch_size"),
                "num_workers": training_args.get("num_workers"),
            },
        },
        "evaluation": metrics,
        "evaluation_flat": evaluation_flat_classifier(metrics),
    }


def write_classifier_run_report(ckpt_path: Path, split: str) -> Path:
    report = build_classifier_run_report(ckpt_path, split)
    out = ckpt_path.parent / f"run_report_{split}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out
