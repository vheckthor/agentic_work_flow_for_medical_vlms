"""Evaluation metrics: BAcc, F1, MRR, ECE (and timing recorded in train/eval scripts).
Implemented by me
Augmented with Cursor IDE 
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def compute_mrr(logits: np.ndarray, labels: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Mean reciprocal rank of the true class under descending score order."""
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    ranks = []
    for i in np.where(mask)[0]:
        order = np.argsort(-logits[i])
        rank = int(np.where(order == labels[i])[0][0]) + 1
        ranks.append(1.0 / rank)
    return float(np.mean(ranks)) if ranks else 0.0


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    mask: Optional[np.ndarray] = None,
    n_bins: int = 15,
) -> float:
    """ECE with fixed-width bins on max predicted probability (multi-class)."""
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    acc = (predictions == labels).astype(np.float64)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = mask
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        in_bin = idx & (confidences > lo) & (confidences <= hi if b < n_bins - 1 else confidences <= hi + 1e-9)
        prop = in_bin.mean()
        if prop > 0:
            acc_bin = acc[in_bin].mean()
            conf_bin = confidences[in_bin].mean()
            ece += prop * abs(acc_bin - conf_bin)
    return float(ece)


@torch.no_grad()
def gather_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns logits [N,C], labels [N] (-1 allowed), mask valid [N]."""
    model.eval()
    use_cuda_amp = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16 if use_cuda_amp and torch.cuda.is_bf16_supported() else torch.float16
    )
    logits_list = []
    labels_list = []
    for batch in loader:
        labels = batch["labels"].to(device)
        pixel = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        valid = labels >= 0
        if not valid.any():
            continue
        if use_cuda_amp:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(pixel[valid], ids[valid], am[valid])
        else:
            out = model(pixel[valid], ids[valid], am[valid])
        logits_list.append(out.float().cpu().numpy())
        labels_list.append(labels[valid].cpu().numpy())
    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    mask = labels >= 0
    return logits, labels, mask


def metrics_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if mask is None:
        mask = labels >= 0
    probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
    pred = logits.argmax(axis=1)
    y_true = labels[mask]
    y_pred = pred[mask]
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "mrr": compute_mrr(logits, labels, mask),
        "ece": expected_calibration_error(probs, labels, mask),
    }
    return out
