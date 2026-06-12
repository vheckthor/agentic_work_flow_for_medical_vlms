"""Shared experiment agent constants and policy choice type.
Implemented by me
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Minimum epochs for any training run launched by the agent or requested by a policy.
MIN_TRAINING_EPOCHS = 40
MIN_LEARNING_RATE = 1e-6
MAX_LEARNING_RATE = 1e-2
MIN_WEIGHT_DECAY = 0.0
MAX_WEIGHT_DECAY = 1.0


@dataclass
class ExperimentChoice:
    """Policy output: chosen experiment with optional training hyperparameter overrides."""

    experiment_id: str
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None


def clamp_epochs(n: int) -> int:
    return max(MIN_TRAINING_EPOCHS, int(n))


def clamp_learning_rate(x: float) -> float:
    return min(MAX_LEARNING_RATE, max(MIN_LEARNING_RATE, float(x)))


def clamp_weight_decay(x: float) -> float:
    return min(MAX_WEIGHT_DECAY, max(MIN_WEIGHT_DECAY, float(x)))
