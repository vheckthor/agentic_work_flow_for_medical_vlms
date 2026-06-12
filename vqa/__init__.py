"""SLAKE closed-set VQA: multimodal classifiers over training answer vocabulary."""

import os
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_hf = _repo_root / ".hf_cache"
_hf.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf))
os.environ.setdefault("HF_HUB_CACHE", str(_hf / "hub"))
