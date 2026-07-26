"""The interface both frameworks implement.

The Keras and PyTorch models are only comparable if they see identical data,
identical folds and identical metrics. That is what this interface enforces: the
training harness never touches a framework API directly, so the difference
between the two runs is the model, not the plumbing around it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class Model(ABC):
    """A return predictor.

    Implementations decide their own input shape — the MLP wants a flat feature
    matrix, the LSTM wants sequences — so each one builds its own arrays from
    the same (train, test) frames via ``prepare``.
    """

    name: str
    framework: str

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Hyperparameters, logged to MLflow so a run is reproducible."""

    @abstractmethod
    def prepare(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        mean: np.ndarray,
        std: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Turn a fold's frame into (X, y, metadata) using train-fold scaling."""

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, validation_split: float = 0.1) -> dict[str, Any]:
        """Train on one fold. Returns a small history dict for logging."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict next-day log returns."""

    @abstractmethod
    def save(self, path: str) -> str:
        """Persist the fitted model; returns the path written."""

    def reset(self) -> None:
        """Discard learned weights before the next fold.

        Walk-forward folds must be independent: carrying weights from fold 3
        into fold 4 means fold 4's model has effectively seen its own past
        through a warm start, which quietly flatters later folds.
        """
        raise NotImplementedError
