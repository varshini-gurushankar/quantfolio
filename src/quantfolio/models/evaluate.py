"""Evaluation metrics for return prediction.

The headline metric is out-of-sample MSE, always reported next to the
**zero-prediction baseline** — the MSE you get by predicting 0.0 for every day,
which is very nearly the variance of returns.

That comparison is the whole story. Daily equity returns are dominated by noise,
so a model can look impressive in isolation and still be worse than predicting
nothing. Any claim of the form "X% lower MSE" is meaningless without saying
lower *than what*, and this module makes sure the answer is always on hand.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    """Prediction quality for one fold or one model."""

    mse: float
    rmse: float
    mae: float
    baseline_mse: float
    improvement_over_baseline: float  # fraction, positive means better
    directional_accuracy: float
    information_coefficient: float
    n_samples: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    @property
    def beats_baseline(self) -> bool:
        return self.mse < self.baseline_mse

    def summary(self) -> str:
        verdict = "beats" if self.beats_baseline else "LOSES TO"
        return (
            f"MSE {self.mse:.3e} vs baseline {self.baseline_mse:.3e} "
            f"({self.improvement_over_baseline:+.2%}, {verdict} zero-prediction), "
            f"DA {self.directional_accuracy:.1%}, IC {self.information_coefficient:+.4f}, "
            f"n={self.n_samples}"
        )


def zero_prediction_mse(y_true: np.ndarray) -> float:
    """MSE of predicting exactly zero — the bar every model has to clear."""
    y = np.asarray(y_true, dtype=np.float64)
    return float(np.mean(y**2))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of days where the sign was called correctly.

    Days with a zero prediction or a zero actual are excluded rather than
    counted as correct, which would inflate the number for a model that has
    learned to predict nothing.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = (y_pred != 0) & (y_true != 0)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask])))


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Correlation between prediction and realised return.

    The quantity a portfolio actually cares about: a model with a poor MSE but a
    reliably positive IC can still be tradeable, and one with a good MSE and a
    zero IC cannot.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) < 3 or np.std(y_pred) < 1e-15 or np.std(y_true) < 1e-15:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """Score predictions against the truth and the zero-prediction baseline."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("cannot evaluate an empty prediction set")

    errors = y_true - y_pred
    mse = float(np.mean(errors**2))
    baseline = zero_prediction_mse(y_true)

    # Positive means the model reduced error relative to predicting nothing.
    improvement = (baseline - mse) / baseline if baseline > 0 else 0.0

    metrics = Metrics(
        mse=mse,
        rmse=float(np.sqrt(mse)),
        mae=float(np.mean(np.abs(errors))),
        baseline_mse=baseline,
        improvement_over_baseline=float(improvement),
        directional_accuracy=directional_accuracy(y_true, y_pred),
        information_coefficient=information_coefficient(y_true, y_pred),
        n_samples=int(y_true.size),
    )

    if not metrics.beats_baseline:
        logger.warning("model does not beat zero-prediction: %s", metrics.summary())
    return metrics


def aggregate(fold_metrics: list[Metrics]) -> Metrics:
    """Combine per-fold metrics into one headline number.

    Averages are weighted by fold size so a short final fold does not count as
    much as a long one.
    """
    if not fold_metrics:
        raise ValueError("no folds to aggregate")

    weights = np.array([m.n_samples for m in fold_metrics], dtype=np.float64)
    weights /= weights.sum()

    def weighted(attr: str) -> float:
        values = np.array([getattr(m, attr) for m in fold_metrics], dtype=np.float64)
        mask = ~np.isnan(values)
        if not mask.any():
            return float("nan")
        return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))

    mse = weighted("mse")
    baseline = weighted("baseline_mse")

    return Metrics(
        mse=mse,
        rmse=float(np.sqrt(mse)),
        mae=weighted("mae"),
        baseline_mse=baseline,
        improvement_over_baseline=float((baseline - mse) / baseline) if baseline > 0 else 0.0,
        directional_accuracy=weighted("directional_accuracy"),
        information_coefficient=weighted("information_coefficient"),
        n_samples=int(sum(m.n_samples for m in fold_metrics)),
    )
