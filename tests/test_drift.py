"""Drift detection.

Two failure modes matter equally: a sensor that never fires is useless, and one
that fires on noise causes retraining loops. Both directions are tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfolio.models.drift import detect_drift, rolling_mse_by_date

REFERENCE_MSE = 1.0e-4


def predictions(
    n_days: int = 90,
    drift_days: int = 0,
    error_multiple: float = 1.0,
    seed: int = 42,
    tickers: tuple[str, ...] = ("AAA", "BBB", "CCC"),
) -> pd.DataFrame:
    """Synthetic scored predictions whose final ``drift_days`` are degraded."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2024-06-28"), periods=n_days)
    base = np.sqrt(REFERENCE_MSE)

    rows = []
    for i, day in enumerate(dates):
        drifting = drift_days > 0 and i >= len(dates) - drift_days
        scale = base * (np.sqrt(error_multiple) if drifting else 1.0)
        for ticker in tickers:
            truth = rng.normal(0.0, 0.011)
            rows.append(
                {
                    "ticker": ticker,
                    "date": day.date(),
                    "y_true": truth,
                    "y_pred": truth + rng.normal(0.0, scale),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# rolling error
# --------------------------------------------------------------------------- #
def test_rolling_mse_tracks_the_reference_on_clean_data() -> None:
    """On clean data the rolling error must never reach the drift threshold.

    Sampling noise means individual windows wander either side of the reference,
    so the meaningful assertion is that none of them crosses the line that would
    trigger a retrain, not that they sit in some narrow band.
    """
    rolling = rolling_mse_by_date(predictions(), window=21).dropna()

    assert not rolling.empty
    assert rolling.max() < REFERENCE_MSE * 1.5, "clean data should never look like drift"
    assert rolling.mean() == pytest.approx(REFERENCE_MSE, rel=0.35)


def test_rolling_mse_is_one_value_per_date() -> None:
    frame = predictions(n_days=40)
    rolling = rolling_mse_by_date(frame, window=10)
    assert len(rolling) == frame["date"].nunique()


def test_rolling_mse_of_empty_input_is_empty() -> None:
    assert rolling_mse_by_date(pd.DataFrame()).empty


def test_unscored_predictions_are_ignored() -> None:
    """Rows with no realised value yet cannot contribute to an error estimate."""
    frame = predictions(n_days=30)
    frame.loc[frame.index[:20], "y_true"] = np.nan
    assert not rolling_mse_by_date(frame, window=5).dropna().empty


# --------------------------------------------------------------------------- #
# the sensor stays quiet
# --------------------------------------------------------------------------- #
def test_clean_data_does_not_trigger() -> None:
    report = detect_drift(predictions(), REFERENCE_MSE, "m", window=21, threshold=1.5)
    assert not report.breached
    assert report.ratio == pytest.approx(1.0, abs=0.4)


def test_a_brief_spike_does_not_trigger() -> None:
    """One bad day is noise. Requiring persistence is what prevents a retrain loop."""
    report = detect_drift(
        predictions(drift_days=1, error_multiple=6.0),
        REFERENCE_MSE,
        "m",
        window=21,
        threshold=1.5,
        min_consecutive_breaches=3,
    )
    assert not report.breached


def test_mild_degradation_below_threshold_does_not_trigger() -> None:
    report = detect_drift(
        predictions(drift_days=30, error_multiple=1.2),
        REFERENCE_MSE,
        "m",
        window=21,
        threshold=1.5,
    )
    assert not report.breached


# --------------------------------------------------------------------------- #
# the sensor fires
# --------------------------------------------------------------------------- #
def test_sustained_degradation_triggers() -> None:
    """The headline behaviour: real drift must be caught."""
    report = detect_drift(
        predictions(drift_days=25, error_multiple=3.0),
        REFERENCE_MSE,
        "m",
        window=21,
        threshold=1.5,
    )
    assert report.breached
    assert report.ratio > 1.5
    assert report.consecutive_breaches >= 3
    assert "DRIFT DETECTED" in report.summary()


def test_worse_drift_gives_a_higher_ratio() -> None:
    mild = detect_drift(predictions(drift_days=25, error_multiple=2.0), REFERENCE_MSE, "m")
    severe = detect_drift(predictions(drift_days=25, error_multiple=5.0), REFERENCE_MSE, "m")
    assert severe.ratio > mild.ratio


def test_a_lower_threshold_is_more_sensitive() -> None:
    frame = predictions(drift_days=25, error_multiple=1.8)
    strict = detect_drift(frame, REFERENCE_MSE, "m", threshold=1.2)
    lenient = detect_drift(frame, REFERENCE_MSE, "m", threshold=4.0)

    assert strict.breached
    assert not lenient.breached


# --------------------------------------------------------------------------- #
# degenerate inputs
# --------------------------------------------------------------------------- #
def test_no_predictions_reports_rather_than_crashing() -> None:
    report = detect_drift(pd.DataFrame(), REFERENCE_MSE, "m")
    assert not report.breached
    assert "no scored predictions" in report.reason


def test_missing_reference_reports_rather_than_crashing() -> None:
    """A model that was never scored cannot be judged to have drifted."""
    report = detect_drift(predictions(), float("nan"), "m")
    assert not report.breached
    assert "no valid reference" in report.reason


def test_zero_reference_is_rejected() -> None:
    report = detect_drift(predictions(), 0.0, "m")
    assert not report.breached


def test_report_serializes_for_xcom() -> None:
    payload = detect_drift(
        predictions(drift_days=25, error_multiple=3.0), REFERENCE_MSE, "m"
    ).as_dict()

    assert payload["breached"] is True
    assert isinstance(payload["as_of"], str)
    assert all(not isinstance(v, (pd.Series, pd.DataFrame)) for v in payload.values())


# --------------------------------------------------------------------------- #
# the demo script
# --------------------------------------------------------------------------- #
def test_injection_script_demonstrates_both_directions() -> None:
    """`scripts/inject_drift.py --dry-run` must actually prove the sensor works."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from inject_drift import main

    assert main(["--dry-run"]) == 0
