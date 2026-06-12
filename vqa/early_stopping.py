"""Early stopping: patience on a validation metric.
Implemented by Cursor AI IDE
"""

from __future__ import annotations

from typing import Literal

Mode = Literal["min", "max"]

#This class is used to implement early stopping
class EarlyStopping:
    """
    Stop when the validation metric does not *improve* for ``patience`` consecutive
    epochs.

    - ``mode="min"``: lower is better (e.g. validation loss).
    - ``mode="max"``: higher is better (e.g. balanced accuracy).

    Improvement uses ``min_delta`` like ``val < best - min_delta`` (min) or
    ``val > best + min_delta`` (max), similar to ``val_loss < best_loss - 1e-3``.
    """

    def __init__(
        self,
        patience: int = 3,
        *,
        min_delta: float = 0.0,
        mode: Mode = "min",
    ):
        self.patience = patience
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best_value: float = float("inf") if mode == "min" else float("-inf")
        self.patience_count = 0
        self.last_improved = False

    def reset(self) -> None:
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.patience_count = 0
        self.last_improved = False

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def step(
        self,
        epoch_1based: int,
        value: float,
        *,
        metric_name: str = "val",
        verbose: bool = True,
    ) -> bool:
        """
        Call once per epoch after computing the validation metric.
        Updates internal best / patience. Sets ``last_improved`` if this step is a new best.

        Returns:
            True if training should stop now.
        """
        improved = self._is_improvement(value)
        self.last_improved = improved
        if improved:
            self.best_value = value
            self.patience_count = 0
        else:
            self.patience_count += 1

        if verbose:
            print(
                "%s=%.5f, improved=%s, patience=%s/%s (epoch=%s)"
                % (
                    metric_name,
                    value,
                    improved,
                    self.patience_count,
                    self.patience,
                    epoch_1based,
                )
            )

        if self.patience_count >= self.patience:
            print("Stopping early!")
            return True
        return False
