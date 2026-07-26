"""Allocation drift — turnover between consecutive optimizer runs.

This is the metric that says whether the optimizer is tracking a genuinely
changing covariance or just chasing estimation noise, so it needs to be right in
the same one-way units the backtest charges costs in.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantfolio.research_pipeline import allocation_drift


def _stored(weights: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_date": pd.Timestamp("2024-06-01").date(),
            "ticker": list(weights),
            "weight": list(weights.values()),
            "method": "markowitz_ledoit_wolf",
        }
    )


@pytest.fixture
def patch_previous(monkeypatch):
    def _apply(frame: pd.DataFrame):
        monkeypatch.setattr(
            "quantfolio.serving.predictor.latest_weights", lambda engine=None: frame
        )

    return _apply


def test_identical_allocations_have_no_drift(patch_previous) -> None:
    patch_previous(_stored({"A": 0.5, "B": 0.5}))
    new = pd.Series({"A": 0.5, "B": 0.5})

    assert allocation_drift(new) == pytest.approx(0.0)


def test_drift_is_one_way_turnover(patch_previous) -> None:
    """Moving 10% from A to B is 10% turnover, matching the backtest's convention."""
    patch_previous(_stored({"A": 0.5, "B": 0.5}))
    new = pd.Series({"A": 0.4, "B": 0.6})

    assert allocation_drift(new) == pytest.approx(0.10)


def test_a_complete_reshuffle_is_full_drift(patch_previous) -> None:
    patch_previous(_stored({"A": 1.0, "B": 0.0}))
    new = pd.Series({"A": 0.0, "B": 1.0})

    assert allocation_drift(new) == pytest.approx(1.0)


def test_first_ever_allocation_is_full_drift(patch_previous) -> None:
    """Building the book from cash is a full turnover, not a free move."""
    patch_previous(pd.DataFrame())
    assert allocation_drift(pd.Series({"A": 0.5, "B": 0.5})) == pytest.approx(1.0)


def test_a_new_ticker_counts_as_movement(patch_previous) -> None:
    """A name entering the universe is a real trade, not a missing value."""
    patch_previous(_stored({"A": 0.5, "B": 0.5}))
    new = pd.Series({"A": 0.4, "B": 0.4, "C": 0.2})

    assert allocation_drift(new) == pytest.approx(0.2)


def test_a_dropped_ticker_counts_as_movement(patch_previous) -> None:
    patch_previous(_stored({"A": 0.4, "B": 0.4, "C": 0.2}))
    new = pd.Series({"A": 0.5, "B": 0.5})

    assert allocation_drift(new) == pytest.approx(0.2)


def test_an_unreadable_database_does_not_break_the_pipeline(monkeypatch) -> None:
    """Drift is a metric, not a gate — losing it must not fail the DAG."""
    import numpy as np

    def boom(engine=None):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("quantfolio.serving.predictor.latest_weights", boom)

    assert np.isnan(allocation_drift(pd.Series({"A": 1.0})))
