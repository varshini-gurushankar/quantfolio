"""Backtest correctness.

The two tests that would catch the classic disasters: charging no transaction
costs, and applying weights on the same day they were computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfolio.portfolio.backtest import (
    backtest,
    compare_to_equal_weight,
    compute_turnover,
    max_drawdown,
    rolling_rebalance,
    sharpe_ratio,
)

TICKERS = ["AAA", "BBB", "CCC"]


@pytest.fixture
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(23)
    dates = pd.bdate_range("2021-01-04", periods=500)
    market = rng.normal(0.0004, 0.008, len(dates))
    return pd.DataFrame(
        {
            t: market * (0.8 + 0.2 * i) + rng.normal(0, 0.005, len(dates))
            for i, t in enumerate(TICKERS)
        },
        index=dates,
    )


@pytest.fixture
def static_weights(returns: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(1.0 / len(TICKERS), index=returns.index, columns=TICKERS)


# --------------------------------------------------------------------------- #
# Sharpe and drawdown
# --------------------------------------------------------------------------- #
def test_sharpe_is_annualized() -> None:
    daily = pd.Series([0.001] * 300)
    daily.iloc[::2] = -0.0005

    expected = daily.mean() / daily.std() * np.sqrt(252)
    assert sharpe_ratio(daily) == pytest.approx(expected)


def test_sharpe_of_a_constant_series_is_zero() -> None:
    assert sharpe_ratio(pd.Series([0.001] * 100)) == 0.0


def test_max_drawdown_is_negative_and_bounded() -> None:
    equity = pd.Series([1.0, 1.2, 0.9, 1.1, 0.8])
    assert max_drawdown(equity) == pytest.approx(0.8 / 1.2 - 1.0)


def test_max_drawdown_of_a_rising_curve_is_zero() -> None:
    assert max_drawdown(pd.Series([1.0, 1.1, 1.2, 1.3])) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# turnover and costs
# --------------------------------------------------------------------------- #
def test_holding_a_static_portfolio_incurs_no_ongoing_turnover(
    static_weights: pd.DataFrame,
) -> None:
    turnover = compute_turnover(static_weights)
    assert turnover.iloc[0] == pytest.approx(1.0), "the first day builds the position from cash"
    assert turnover.iloc[1:].sum() == pytest.approx(0.0)


def test_turnover_is_one_way() -> None:
    """Selling 10% of A to buy 10% of B is 10% turnover, not 20%."""
    weights = pd.DataFrame(
        {"A": [0.5, 0.4], "B": [0.5, 0.6]},
        index=pd.bdate_range("2023-01-02", periods=2),
    )
    assert compute_turnover(weights).iloc[1] == pytest.approx(0.10)


def test_costs_reduce_returns(returns: pd.DataFrame, static_weights: pd.DataFrame) -> None:
    result = backtest(static_weights, returns, cost_bps=10.0)
    assert result.sharpe_net <= result.sharpe_gross
    assert result.total_cost > 0


def test_zero_cost_makes_gross_and_net_identical(
    returns: pd.DataFrame, static_weights: pd.DataFrame
) -> None:
    result = backtest(static_weights, returns, cost_bps=0.0)
    assert result.sharpe_net == pytest.approx(result.sharpe_gross)
    assert result.total_cost == pytest.approx(0.0)


def test_high_turnover_is_punished_by_costs(returns: pd.DataFrame) -> None:
    """The failure mode a gross-only backtest hides completely."""
    rng = np.random.default_rng(31)
    thrashing = pd.DataFrame(
        rng.dirichlet(np.ones(len(TICKERS)), len(returns)),
        index=returns.index,
        columns=TICKERS,
    )

    cheap = backtest(thrashing, returns, cost_bps=0.0)
    expensive = backtest(thrashing, returns, cost_bps=50.0)

    assert expensive.sharpe_net < cheap.sharpe_net
    assert expensive.average_turnover > 0.2


def test_costs_scale_with_the_rate(returns: pd.DataFrame, static_weights: pd.DataFrame) -> None:
    ten = backtest(static_weights, returns, cost_bps=10.0)
    twenty = backtest(static_weights, returns, cost_bps=20.0)
    assert twenty.total_cost == pytest.approx(2 * ten.total_cost)


# --------------------------------------------------------------------------- #
# the lag
# --------------------------------------------------------------------------- #
def test_weights_are_applied_with_a_one_day_lag(returns: pd.DataFrame) -> None:
    """Trading on weights computed from today's close is a lookahead bug.

    A portfolio that goes all-in on whichever asset rises tomorrow would post an
    impossible Sharpe if the lag were missing.
    """
    clairvoyant = pd.DataFrame(0.0, index=returns.index, columns=TICKERS)
    best = returns.idxmax(axis=1)
    for day, ticker in best.items():
        clairvoyant.loc[day, ticker] = 1.0

    lagged = backtest(clairvoyant, returns, cost_bps=0.0, lag_days=1)
    same_day = backtest(clairvoyant, returns, cost_bps=0.0, lag_days=0)

    # Without the lag this portfolio earns a Sharpe no real strategy reaches,
    # purely from knowing the answer. With it, the foresight is worthless.
    assert same_day.sharpe_gross > 5, "the no-lag case should be implausibly good"
    assert lagged.sharpe_gross < 2.0, "one day of lag should leave no free lunch"
    assert lagged.sharpe_gross < same_day.sharpe_gross / 5


def test_returns_series_is_shorter_than_the_weight_series(
    returns: pd.DataFrame, static_weights: pd.DataFrame
) -> None:
    """The lag necessarily costs the first observation."""
    result = backtest(static_weights, returns, lag_days=1)
    assert result.n_periods == len(returns) - 1


# --------------------------------------------------------------------------- #
# rolling rebalance
# --------------------------------------------------------------------------- #
def test_rolling_rebalance_only_uses_trailing_data(returns: pd.DataFrame) -> None:
    """Weights dated t must be reproducible from data up to t alone."""
    full = rolling_rebalance(returns, lookback=252, rebalance_every=63, max_weight=0.5)

    cut = 400
    truncated = rolling_rebalance(
        returns.iloc[:cut], lookback=252, rebalance_every=63, max_weight=0.5
    )

    shared = full.index.intersection(truncated.index)
    assert len(shared) > 0
    pd.testing.assert_frame_equal(
        full.loc[shared], truncated.loc[shared], check_exact=False, atol=1e-6
    )


def test_weights_are_held_between_rebalances(returns: pd.DataFrame) -> None:
    weights = rolling_rebalance(returns, lookback=252, rebalance_every=63, max_weight=0.5)
    changed_days = int((weights.diff().abs().sum(axis=1) > 1e-9).sum())
    assert changed_days < len(weights) / 10, "weights should be piecewise constant"


def test_rebalanced_weights_stay_valid(returns: pd.DataFrame) -> None:
    weights = rolling_rebalance(returns, lookback=252, rebalance_every=63, max_weight=0.5)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    assert (weights >= -1e-9).all().all()


def test_too_little_history_is_rejected(returns: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="need more than"):
        rolling_rebalance(returns.iloc[:100], lookback=252, max_weight=0.5)


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
def test_comparison_reports_both_strategies(returns: pd.DataFrame) -> None:
    weights = rolling_rebalance(returns, lookback=252, rebalance_every=63, max_weight=0.5)
    table = compare_to_equal_weight(returns, weights, cost_bps=10.0)

    assert set(table["strategy"]) == {"markowitz_lw", "equal_weight"}
    assert {"sharpe_gross", "sharpe_net"}.issubset(table.columns)
    assert (table["sharpe_net"] <= table["sharpe_gross"]).all()


def test_backtest_rejects_disjoint_inputs(returns: pd.DataFrame) -> None:
    wrong = pd.DataFrame(0.5, index=returns.index, columns=["XXX", "YYY"])
    with pytest.raises(ValueError, match="share no tickers"):
        backtest(wrong, returns)


def test_backtest_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="requires both"):
        backtest(pd.DataFrame(), pd.DataFrame())
