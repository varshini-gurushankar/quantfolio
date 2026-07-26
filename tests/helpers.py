"""Shared test doubles.

Kept out of the test modules so more than one file can use them without
importing across test modules, which depends on pytest's import mode.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quantfolio.models.base import Model
from quantfolio.models.datasets import TARGET, transform


class RecordingModel(Model):
    """A least-squares fit that records what each fold was shown.

    Deterministic and exact, so harness behaviour (fold independence, target
    scaling, prediction alignment) can be asserted precisely rather than
    approximately as it would be with a real network.
    """

    name = "recording"
    framework = "test"

    def __init__(self) -> None:
        self.coef: np.ndarray | None = None
        self.fit_calls: list[dict] = []
        self.reset_calls = 0

    def params(self) -> dict[str, Any]:
        return {"model": self.name}

    def prepare(self, frame, feature_columns, mean, std):
        x = transform(frame, feature_columns, mean, std)
        y = frame[TARGET].to_numpy(dtype=np.float64)
        return x, y, frame[["ticker", "date"]].reset_index(drop=True)

    def fit(self, x, y, validation_split: float = 0.1):
        self.fit_calls.append({"n": len(x), "y_mean": float(np.mean(y)), "y_std": float(np.std(y))})
        self.coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        return {"epochs_run": 1}

    def predict(self, x):
        return x @ self.coef

    def save(self, path: str) -> str:
        return path

    def reset(self) -> None:
        self.reset_calls += 1
        self.coef = None
