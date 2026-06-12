#!/usr/bin/env python3
"""Print a comparison table from checkpoints/*/metrics_test.json (for report tables).
Implemented by me
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints_root", type=str, default="checkpoints")
    p.add_argument("--split", type=str, default="test")
    args = p.parse_args()
    root = Path(args.checkpoints_root)
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        mf = d / f"metrics_{args.split}.json"
        if not mf.is_file():
            continue
        with open(mf, encoding="utf-8") as f:
            m = json.load(f)
        m["variant"] = d.name
        rows.append(m)
    if not rows:
        print("No metrics files found. Run evaluate.py per variant first.")
        return
    keys = [
        "variant",
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_micro",
        "mrr",
        "ece",
        "in_vocab_examples",
        "total_examples",
        "test_inference_seconds",
    ]
    header = "\t".join(keys)
    print(header)
    for r in rows:
        print("\t".join(str(r.get(k, "")) for k in keys))


if __name__ == "__main__":
    main()
