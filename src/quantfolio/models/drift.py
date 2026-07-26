"""Model drift detection — the trigger for automated retraining.

A model trained on 2019-2023 is not automatically wrong in 2025, but it is not
automatically right either. Volatility regimes shift, correlations break, and a
model's error quietly grows. The question this module answers is: *has recent
out-of-sample error deteriorated enough, relative to what the model achieved at
training time, to justify retraining?*

Two guards against retraining on noise:

* The comparison is against the model's **own** training-time OOS MSE, not a
  fixed constant. A model with intrinsically high error should not be
  perpetually "drifting".
* A breach must persist for ``min_consecutive_breaches`` days. Daily return
  error is noisy enough that a single bad day means nothing, and a retraining
  loop that fires on one is worse than no sensor at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

logger = logging.getLogger(__name__)

DEFAULT_WINDOW = 21  # about a trading month
DEFAULT_THRESHOLD = 1.5  # retrain when recent MSE is 50% above the reference
DEFAULT_MIN_BREACHES = 3


@dataclass
class DriftReport:
    """The sensor's verdict, with the numbers behind it."""

    model_name: str
    breached: bool
    current_mse: float
    reference_mse: float
    ratio: float
    threshold: float
    window: int
    consecutive_breaches: int
    n_observations: int
    as_of: date | None = None
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "breached": self.breached,
            "current_mse": self.current_mse,
            "reference_mse": self.reference_mse,
            "ratio": self.ratio,
            "threshold": self.threshold,
            "consecutive_breaches": self.consecutive_breaches,
            "n_observations": self.n_observations,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "reason": self.reason,
        }

    def summary(self) -> str:
        verdict = "DRIFT DETECTED" if self.breached else "within tolerance"
        return (
            f"{self.model_name}: {verdict} — recent MSE {self.current_mse:.3e} vs "
            f"reference {self.reference_mse:.3e} (ratio {self.ratio:.2f}, "
            f"threshold {self.threshold:.2f}), {self.consecutive_breaches} consecutive "
            f"breach(es) over {self.n_observations} days. {self.reason}"
        )


def rolling_mse_by_date(predictions: pd.DataFrame, window: int = DEFAULT_WINDOW) -> pd.Series:
    """Rolling mean squared error, one value per date.

    Errors are averaged across tickers within a date first, so a day with more
    symbols reporting does not dominate the window.
    """
    if predictions.empty:
        return pd.Series(dtype="float64")

    scored = predictions.dropna(subset=["y_true", "y_pred"]).copy()
    if scored.empty:
        return pd.Series(dtype="float64")

    scored["sq_error"] = (scored["y_true"] - scored["y_pred"]) ** 2
    daily = scored.groupby("date")["sq_error"].mean().sort_index()
    return daily.rolling(window, min_periods=max(2, window // 2)).mean()


def detect_drift(
    predictions: pd.DataFrame,
    reference_mse: float,
    model_name: str = "unknown",
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    min_consecutive_breaches: int = DEFAULT_MIN_BREACHES,
) -> DriftReport:
    """Decide whether recent error justifies retraining."""
    empty = DriftReport(
        model_name=model_name,
        breached=False,
        current_mse=float("nan"),
        reference_mse=reference_mse,
        ratio=float("nan"),
        threshold=threshold,
        window=window,
        consecutive_breaches=0,
        n_observations=0,
    )

    if reference_mse is None or not np.isfinite(reference_mse) or reference_mse <= 0:
        empty.reason = "no valid reference MSE; cannot judge drift"
        return empty

    rolling = rolling_mse_by_date(predictions, window).dropna()
    if rolling.empty:
        empty.reason = f"no scored predictions in the last {window} days"
        return empty

    limit = reference_mse * threshold
    breaches = rolling > limit

    # Count the breach streak ending at the most recent observation.
    consecutive = 0
    for value in reversed(breaches.tolist()):
        if value:
            consecutive += 1
        else:
            break

    current = float(rolling.iloc[-1])
    breached = consecutive >= min_consecutive_breaches

    report = DriftReport(
        model_name=model_name,
        breached=breached,
        current_mse=current,
        reference_mse=reference_mse,
        ratio=current / reference_mse,
        threshold=threshold,
        window=window,
        consecutive_breaches=consecutive,
        n_observations=len(rolling),
        as_of=pd.Timestamp(rolling.index[-1]).date(),
        reason=(
            f"{consecutive} consecutive day(s) above {limit:.3e}"
            if breached
            else f"needs {min_consecutive_breaches} consecutive breaches, has {consecutive}"
        ),
    )
    logger.info("%s", report.summary())
    return report


def load_recent_predictions(
    model_name: str,
    lookback_days: int = 90,
    as_of: date | None = None,
    engine=None,
) -> pd.DataFrame:
    """Read scored predictions for one model from Postgres."""
    from quantfolio.storage.db import get_engine
    from quantfolio.storage.schema import predictions_daily

    engine = engine or get_engine()
    as_of = as_of or date.today()
    start = as_of - timedelta(days=lookback_days)

    stmt = (
        select(predictions_daily)
        .where(predictions_daily.c.model_name == model_name)
        .where(predictions_daily.c.date >= start)
        .where(predictions_daily.c.date <= as_of)
        .order_by(predictions_daily.c.date)
    )
    with engine.connect() as conn:
        return pd.read_sql(stmt, conn)


def load_reference_mse(model_name: str, engine=None) -> float | None:
    """The model's training-time OOS MSE — the bar recent error is judged against."""
    from quantfolio.storage.db import get_engine
    from quantfolio.storage.schema import model_metrics

    engine = engine or get_engine()
    stmt = (
        select(model_metrics.c.mse)
        .where(model_metrics.c.model_name == model_name)
        .order_by(model_metrics.c.logged_at.desc())
        .limit(1)
    )
    with engine.connect() as conn:
        row = conn.execute(stmt).first()
    return float(row[0]) if row else None


def store_predictions(
    predictions: pd.DataFrame,
    model_name: str,
    run_id: str | None = None,
    engine=None,
) -> int:
    """Persist out-of-sample predictions, upserted on (model, ticker, date)."""
    from quantfolio.storage.db import upsert
    from quantfolio.storage.schema import predictions_daily

    if predictions.empty:
        return 0

    frame = predictions.copy()
    frame["model_name"] = model_name
    frame["run_id"] = run_id
    return upsert(frame, predictions_daily, engine=engine)


def store_reference_metrics(
    model_name: str,
    run_id: str,
    mse: float,
    baseline_mse: float,
    window_end: date,
    engine=None,
) -> int:
    """Record the training-time OOS MSE the drift sensor will compare against."""
    from quantfolio.storage.db import upsert
    from quantfolio.storage.schema import model_metrics

    frame = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "window_end": window_end,
                "model_name": model_name,
                "mse": mse,
                "baseline_mse": baseline_mse,
            }
        ]
    )
    return upsert(frame, model_metrics, engine=engine)
