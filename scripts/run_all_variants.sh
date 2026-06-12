#!/usr/bin/env bash
# Train all assignment variants (fine-tuned). Uses project venv — run from repo root:
#   chmod +x scripts/run_all_variants.sh && ./scripts/run_all_variants.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
export HF_HOME="${ROOT}/.hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
cd "$ROOT"
for v in vit_bert resnet_bert clip cross_attn vit_bert_focal; do
  echo "=== training $v ==="
  "$PY" train.py --data_root SLAKE --variant "$v" --epochs 5 --batch_size 8 --out_dir checkpoints
done
echo "Done. Evaluate each with:"
echo "  .venv/bin/python evaluate.py --checkpoint checkpoints/<variant>/model.pt --split test"
