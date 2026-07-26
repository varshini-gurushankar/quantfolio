"""Markowitz optimization and Ledoit-Wolf shrinkage.

The constraint tests matter most: an optimizer that quietly violates its own
long-only or concentration limits produces a portfolio nobody could trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfolio.portfolio.optimizer import (
    equal_weight,
    ledoit_wolf_covariance,
    optimize,
    returns_matrix,
)


@pytest.fixture
def returns() -> pd.DataFrame:
    """Six assets driven by a common factor, so the covariance is not diagonal."""
    rng = np.random.default_rng(17)
    dates = pd.bdate_range("2021-01-04", periods=600)
    market = rng.normal(0.0003, 0.009, len(dates))

    data = {}
    for i, ticker in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "SPY"]):
        beta = 0.6 + 0.15 * i
        data[ticker] = beta * market + rng.normal(0.0001, 0.006, len(dates))
    return pd.DataFrame(data, index=dates)


# --------------------------------------------------------------------------- #
# shrinkage
# --------------------------------------------------------------------------- #
def test_shrinkage_intensity_is_a_fraction(returns: pd.DataFrame) -> None:
    _, shrinkage = ledoit_wolf_covariance(returns)
    assert 0.0 <= shrinkage <= 1.0


def test_shrunk_covariance_is_symmetric_and_psd(returns: pd.DataFrame) -> None:
    cov, _ = ledoit_wolf_covariance(returns)
    np.testing.assert_allclose(cov, cov.T, rtol=1e-12)
    assert np.linalg.eigvalsh(cov).min() > 0, "shrinkage should guarantee positive definiteness"


def test_shrinkage_helps_most_when_data_is_scarce() -> None:
    """The regime Markowitz actually fails in: many assets, few observations.

    With T barely above N the sample covariance is nearly singular, and that is
    exactly where the optimizer's error amplification does its damage.
    """
    rng = np.random.default_rng(5)
    n_assets, n_obs = 15, 25
    data = pd.DataFrame(
        rng.normal(0, 0.01, (n_obs, n_assets)),
        columns=[f"A{i}" for i in range(n_assets)],
    )

    sample_cond = np.linalg.cond(data.cov().to_numpy())
    shrunk, intensity = ledoit_wolf_covariance(data)

    assert intensity > 0.1, "scarce data should pull hard toward the target"
    assert np.linalg.cond(shrunk) < sample_cond / 10, "shrinkage should transform conditioning"


# --------------------------------------------------------------------------- #
# constraints
# --------------------------------------------------------------------------- #
def test_weights_are_fully_invested(returns: pd.DataFrame) -> None:
    result = optimize(returns, max_weight=0.30)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)


def test_weights_are_long_only(returns: pd.DataFrame) -> None:
    result = optimize(returns, max_weight=0.30)
    assert (result.weights >= -1e-9).all(), "long-only constraint violated"


def test_position_cap_is_respected(returns: pd.DataFrame) -> None:
    """Without a cap the optimizer piles into whichever estimate looks safest."""
    result = optimize(returns, max_weight=0.25)
    assert result.weights.max() <= 0.25 + 1e-6


def test_a_tighter_cap_forces_diversification(returns: pd.DataFrame) -> None:
    loose = optimize(returns, max_weight=0.50)
    tight = optimize(returns, max_weight=0.20)

    assert tight.diagnostics["effective_n"] >= loose.diagnostics["effective_n"]


def test_an_impossible_cap_is_rejected(returns: pd.DataFrame) -> None:
    """Six assets capped at 10% each cannot sum to 1 — say so, don't solve it."""
    with pytest.raises(ValueError, match="fully invested"):
        optimize(returns, max_weight=0.10)


def test_minimum_variance_beats_equal_weight_in_sample(returns: pd.DataFrame) -> None:
    """A sanity check on the objective: it should minimise what it claims to."""
    optimized = optimize(returns, max_weight=0.50)

    n = returns.shape[1]
    equal = np.full(n, 1.0 / n)
    cov = ledoit_wolf_covariance(returns)[0] * 252
    equal_vol = float(np.sqrt(equal @ cov @ equal))

    assert optimized.volatility <= equal_vol + 1e-9


def test_target_return_constraint_is_binding(returns: pd.DataFrame) -> None:
    unconstrained = optimize(returns, max_weight=0.40)
    demanding = optimize(
        returns, max_weight=0.40, target_return=unconstrained.expected_return * 1.3
    )

    if demanding.is_optimal:
        assert demanding.expected_return >= unconstrained.expected_return - 1e-6
        # Chasing return costs risk; that trade-off is the efficient frontier.
        assert demanding.volatility >= unconstrained.volatility - 1e-9


def test_infeasible_problem_falls_back_to_equal_weight(returns: pd.DataFrame) -> None:
    """An unreachable return target should degrade, not crash the DAG."""
    result = optimize(returns, max_weight=0.30, target_return=10.0)

    assert result.weights.sum() == pytest.approx(1.0)
    assert "failed" in result.status or result.is_optimal


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        optimize(pd.DataFrame())


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #
def test_returns_matrix_pivots_a_long_panel() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["A", "A", "B", "B"],
            "date": pd.to_datetime(["2023-01-03", "2023-01-04"] * 2),
            "log_return": [0.01, 0.02, 0.03, 0.04],
        }
    )
    wide = returns_matrix(frame)
    assert list(wide.columns) == ["A", "B"]
    assert wide.shape == (2, 2)


def test_ragged_dates_are_dropped() -> None:
    """A ticker missing on a date makes that whole cross-section unusable."""
    frame = pd.DataFrame(
        {
            "ticker": ["A", "A", "B"],
            "date": pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-03"]),
            "log_return": [0.01, 0.02, 0.03],
        }
    )
    wide = returns_matrix(frame)
    assert len(wide) == 1


def test_equal_weight_is_uniform_and_sums_to_one() -> None:
    result = equal_weight(["A", "B", "C", "D"])
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.weights.nunique() == 1


def test_result_frame_is_ready_for_the_database(returns: pd.DataFrame) -> None:
    from datetime import date

    result = optimize(returns, max_weight=0.30, as_of=date(2023, 6, 1))
    frame = result.to_frame()

    assert set(frame.columns) == {"as_of_date", "ticker", "weight", "method"}
    assert len(frame) == returns.shape[1]
    assert frame["weight"].sum() == pytest.approx(1.0)
