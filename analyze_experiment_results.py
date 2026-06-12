#!/usr/bin/env python3
"""
Aggregate ``run_report_*.json`` files from training/eval runs and produce tables + figures for papers.
Implemented by me
Augmented with Cursor IDE 

**How reporting works in this repo**

1. ``train.py`` writes ``train_timing.json`` (wall time, epochs_ran, early stopping flags).
2. ``evaluate.py`` writes ``metrics_*.json`` and a unified ``run_report_<split>.json``
   (training + full evaluation metrics + ``evaluation_flat`` for stable CSV columns).

3. ``experiment_agent.py`` copies the same ``run_report_test.json`` into each history entry as ``run_report``.

**Usage**

  .venv/bin/python analyze_experiment_results.py --roots checkpoints checkpoints_smoke --out-dir reporting

Outputs under ``reporting/``:

- ``results_table.json`` — list of flattened dicts
- ``results_table.csv`` — same for Excel / LaTeX
- ``figures/time_share_training.png`` — pie chart of training time by run
- ``figures/metrics_by_variant.png`` — bar chart (score + inference time)
- ``summary.txt`` — mean/std of key metrics when multiple runs share a label

Requires: matplotlib (``pip install matplotlib``).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def discover_reports(roots: List[Path], name_glob: str) -> List[Path]:
    found: List[Path] = []
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            continue
        found.extend(sorted(root.rglob(name_glob)))
    # de-dupe same file
    seen = set()
    out: List[Path] = []
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def agent_experiment_id(path: Path) -> Optional[str]:
    parts = path.parts
    if "agent" in parts:
        i = parts.index("agent")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def flatten_report(path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "report_path": str(path),
        "family": data.get("family"),
        "variant": data.get("variant"),
        "split": data.get("split"),
        "schema_version": data.get("schema_version"),
        "agent_experiment_id": agent_experiment_id(path),
    }
    tr = data.get("training") or {}
    for k, v in tr.items():
        row[f"training_{k}"] = v
    ef = data.get("evaluation_flat") or {}
    for k, v in ef.items():
        row[f"eval_{k}"] = v
    row["score_for_compare"], row["score_label"] = unified_compare_score(data.get("family"), ef)
    row["inference_seconds"] = _inference_seconds(data.get("family"), ef)
    return row


def _inference_seconds(family: Optional[str], ef: Dict[str, Any]) -> Optional[float]:
    if family == "classifier":
        v = ef.get("inference_seconds")
        return float(v) if v is not None else None
    v = ef.get("inference_seconds")
    return float(v) if v is not None else None


def unified_compare_score(
    family: Optional[str], ef: Dict[str, Any]
) -> Tuple[Optional[float], str]:
    """One number per row for bar charts (family-specific)."""
    if family == "classifier":
        v = ef.get("balanced_accuracy")
        return (float(v) if v is not None else None, "balanced_accuracy")
    return None, "none"


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _csv_val(r.get(k)) for k in keys})


def _csv_val(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return str(v)


def text_summary(rows: List[Dict[str, Any]]) -> str:
    by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        lab = str(r.get("variant") or r.get("family") or "unknown")
        by_variant[lab].append(r)

    lines: List[str] = []
    for lab in sorted(by_variant):
        grp = by_variant[lab]
        lines.append(f"=== {lab} (n={len(grp)}) ===")

        def stat(key: str) -> str:
            xs = [float(r[key]) for r in grp if r.get(key) is not None]
            if not xs:
                return "n/a"
            if len(xs) == 1:
                return f"{xs[0]:.6f}"
            return f"mean={statistics.mean(xs):.6f} std={statistics.stdev(xs):.6f} min={min(xs):.6f} max={max(xs):.6f}"

        for metric in (
            "score_for_compare",
            "training_train_seconds",
            "training_epochs_ran",
            "inference_seconds",
            "eval_balanced_accuracy",
            "eval_f1_macro",
            "eval_balanced_accuracy_parsed",
            "eval_exact_match_normalized",
        ):
            if any(metric in r for r in grp):
                lines.append(f"  {metric}: {stat(metric)}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def plot_figures(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Pie: training time by variant
    labels: List[str] = []
    times: List[float] = []
    for r in rows:
        t = r.get("training_train_seconds")
        if t is None:
            continue
        lab = str(r.get("variant") or r.get("family") or "unknown")
        # shorten path noise: prefer agent id if many same variant
        aid = r.get("agent_experiment_id")
        if aid:
            lab = f"{lab}\n({aid})"
        labels.append(lab)
        times.append(float(t))
    if times and sum(times) > 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(times, labels=labels, autopct="%1.1f%%")
        ax.set_title("Share of total training time by run")
        fig.tight_layout()
        fig.savefig(fig_dir / "time_share_training.png", dpi=150)
        plt.close(fig)

    # Bar: score_for_compare and inference time (two subplots)
    variants: List[str] = []
    scores: List[float] = []
    infer: List[float] = []
    for r in rows:
        if r.get("score_for_compare") is None:
            continue
        variants.append(str(r.get("variant") or "?")[:24])
        scores.append(float(r["score_for_compare"]))
        inf = r.get("inference_seconds")
        infer.append(float(inf) if inf is not None else 0.0)

    if variants:
        x = range(len(variants))
        fig, axes = plt.subplots(2, 1, figsize=(max(10, len(variants) * 0.4), 8))
        axes[0].bar(x, scores, color="steelblue")
        axes[0].set_xticks(list(x))
        axes[0].set_xticklabels(variants, rotation=45, ha="right")
        axes[0].set_ylabel("score_for_compare")
        axes[0].set_title("Primary comparison score by run (see CSV score_label)")

        axes[1].bar(x, infer, color="coral")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(variants, rotation=45, ha="right")
        axes[1].set_ylabel("seconds")
        axes[1].set_title("Inference time (test split)")
        fig.tight_layout()
        fig.savefig(fig_dir / "metrics_by_variant.png", dpi=150)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate run_report JSON files into CSV + figures")
    p.add_argument(
        "--roots",
        nargs="+",
        default=["checkpoints", "checkpoints_smoke"],
        help="Directories to search recursively",
    )
    p.add_argument(
        "--glob",
        dest="name_glob",
        default="run_report_test.json",
        help="Filename glob (default: run_report_test.json)",
    )
    p.add_argument("--out-dir", type=str, default="reporting", help="Output directory")
    p.add_argument("--no-plots", action="store_true", help="Skip matplotlib figures")
    args = p.parse_args()

    roots = [Path(r) for r in args.roots]
    reports = discover_reports(roots, args.name_glob)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for path in reports:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows.append(flatten_report(path, data))

    rows.sort(key=lambda r: (str(r.get("family")), str(r.get("variant")), str(r.get("report_path"))))

    with open(out_dir / "results_table.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    write_csv(rows, out_dir / "results_table.csv")

    summary_path = out_dir / "summary.txt"
    summary_path.write_text(text_summary(rows), encoding="utf-8")

    if not args.no_plots and rows:
        try:
            plot_figures(rows, out_dir)
        except ImportError:
            summary_path.write_text(
                summary_path.read_text(encoding="utf-8")
                + "\n[plots skipped: install matplotlib]\n",
                encoding="utf-8",
            )

    print(f"Found {len(reports)} report(s). Wrote {out_dir}/results_table.json and .csv")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
