"""Portfolio backtest with transaction costs.

Two things here are deliberately pessimistic, because a backtest's job is to
avoid flattering you.

**Costs are charged on turnover.** A strategy that rebalances daily to a
frequently-changing optimum can look excellent gross and lose money net. At 10
basis points per side, a portfolio that turns over 50% a day gives up roughly
25% a year. Reporting only gross Sharpe is the single easiest way to produce a
strategy that does not survive contact with a broker.

**Weights are applied with a one-day lag.** Weights computed from data through
close on day *t* cannot be traded until day *t+1*, so returns are earned from
*t+1* onward. Skipping this lag is a lookahead bug that reliably manufactures a
spectacular Sharpe out of nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
DEFAULT_COST_BPS = 10.0  # per side, per unit turnover


@dataclass
class BacktestResult:
    """Gross and net performance. Both are always reported, never just gross."""

    returns_gross: pd.Series = field(repr=False)
    returns_net: pd.Series = field(repr=False)
    equity_curve: pd.Series = field(repr=False)
    turnover: pd.Series = field(repr=False)

    sharpe_gross: float = 0.0
    sharpe_net: float = 0.0
    annual_return_gross: float = 0.0
    annual_return_net: float = 0.0
    volatility: float = 0.0
    max_drawdown: float = 0.0
    total_cost: float = 0.0
    average_turnover: float = 0.0
    n_periods: int = 0
    cost_bps: float = DEFAULT_COST_BPS

    def as_dict(self) -> dict[str, float]:
        return {
            "sharpe_gross": self.sharpe_gross,
            "sharpe_net": self.sharpe_net,
            "annual_return_gross": self.annual_return_gross,
            "annual_return_net": self.annual_return_net,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "total_cost": self.total_cost,
            "average_turnover": self.average_turnover,
            "n_periods": float(self.n_periods),
            "cost_bps": self.cost_bps,
        }

    def summary(self) -> str:
        drag = self.sharpe_gross - self.sharpe_net
        return (
            f"Sharpe {self.sharpe_net:.2f} net / {self.sharpe_gross:.2f} gross "
            f"(cost drag {drag:.2f}), return {self.annual_return_net:.2%} net, "
            f"vol {self.volatility:.2%}, max DD {self.max_drawdown:.2%}, "
            f"turnover {self.average_turnover:.1%}/day, {self.n_periods} days"
        )


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio from a series of daily returns."""
    if len(returns) < 2:
        return float("nan")

    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    std = excess.std()
    if std < 1e-12:
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Worst peak-to-trough decline, as a negative fraction."""
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    return float((equity_curve / running_max - 1.0).min())


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """One-way turnover per rebalance: half the sum of absolute weight changes.

    Halved because selling 10% of A to buy 10% of B is a 10% turnover, not 20%.
    Costs are then charged per side against this figure.
    """
    if weights.empty:
        return pd.Series(dtype="float64")

    changes = weights.diff().abs().sum(axis=1) / 2.0
    # The first rebalance builds the whole position from cash.
    changes.iloc[0] = weights.iloc[0].abs().sum()
    return changes


def backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = DEFAULT_COST_BPS,
    risk_free_rate: float = 0.0,
    lag_days: int = 1,
) -> BacktestResult:
    """Run a daily-rebalance backtest.

    ``weights`` is (date x ticker) target weights; ``returns`` is (date x ticker)
    simple returns. Weights on date *t* are traded into effect on *t + lag_days*,
    which is the earliest a real desk could act on them.
    """
    if weights.empty or returns.empty:
        raise ValueError("backtest requires both weights and returns")

    tickers = [t for t in weights.columns if t in returns.columns]
    if not tickers:
        raise ValueError("weights and returns share no tickers")

    weights = weights[tickers].sort_index()
    returns = returns[tickers].sort_index()

    # The lag: today's weights earn tomorrow's returns.
    effective = weights.shift(lag_days).dropna(how="all")
    aligned_dates = effective.index.intersection(returns.index)
    if len(aligned_dates) == 0:
        raise ValueError("no overlapping dates between lagged weights and returns")

    effective = effective.loc[aligned_dates].fillna(0.0)
    period_returns = returns.loc[aligned_dates]

    gross = (effective * period_returns).sum(axis=1)

    turnover = compute_turnover(effective)
    # Charged per side: a 10 bps cost on 30% turnover is 3 bps that day.
    costs = turnover * (cost_bps / 10_000.0)
    net = gross - costs

    equity = (1.0 + net).cumprod()

    result = BacktestResult(
        returns_gross=gross,
        returns_net=net,
        equity_curve=equity,
        turnover=turnover,
        sharpe_gross=sharpe_ratio(gross, risk_free_rate),
        sharpe_net=sharpe_ratio(net, risk_free_rate),
        annual_return_gross=float(gross.mean() * TRADING_DAYS_PER_YEAR),
        annual_return_net=float(net.mean() * TRADING_DAYS_PER_YEAR),
        volatility=float(net.std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
        max_drawdown=max_drawdown(equity),
        total_cost=float(costs.sum()),
        average_turnover=float(turnover.mean()),
        n_periods=len(net),
        cost_bps=cost_bps,
    )
    logger.info("%s", result.summary())
    return result


def rolling_rebalance(
    returns: pd.DataFrame,
    lookback: int = 252,
    rebalance_every: int = 21,
    max_weight: float = 0.30,
    use_shrinkage: bool = True,
) -> pd.DataFrame:
    """Walk forward, re-optimizing periodically on trailing data only.

    Each optimization at date *t* sees returns from ``[t-lookback, t]`` and
    nothing after, which is what makes the resulting weight series usable in a
    backtest without leaking. Between rebalances the previous weights are held.

    Monthly rebalancing rather than daily is itself a cost decision: re-solving
    every day chases noise in the covariance estimate and pays for the privilege.
    """
    from quantfolio.portfolio.optimizer import optimize

    if len(returns) <= lookback:
        raise ValueError(f"need more than {lookback} observations, got {len(returns)}")

    # Check the cap once, up front. It is a property of the universe size, not
    # of any particular window, so letting the loop discover it repeatedly would
    # turn one clear configuration error into N identical warnings and a vague
    # "no successful optimizations" at the end.
    n_assets = returns.shape[1]
    if n_assets * max_weight < 1.0 - 1e-9:
        raise ValueError(
            f"max_weight {max_weight} across {n_assets} assets cannot reach a fully "
            f"invested portfolio; need at least {1.0 / n_assets:.3f}"
        )

    weights_by_date: dict = {}
    dates = returns.index

    for i in range(lookback, len(dates), rebalance_every):
        window = returns.iloc[i - lookback : i]
        try:
            result = optimize(
                window,
                max_weight=max_weight,
                use_shrinkage=use_shrinkage,
                as_of=dates[i],
            )
            weights_by_date[dates[i]] = result.weights
        except Exception as exc:  # noqa: BLE001 - a bad window should not end the run
            logger.warning("optimization failed at %s: %s", dates[i], exc)

    if not weights_by_date:
        raise RuntimeError("no successful optimizations across the whole period")

    weights = pd.DataFrame(weights_by_date).T.sort_index()
    # Hold weights between rebalances; ffill only ever carries the past forward.
    return weights.reindex(dates).ffill().dropna(how="all")


def compare_to_equal_weight(
    returns: pd.DataFrame,
    optimized_weights: pd.DataFrame,
    cost_bps: float = DEFAULT_COST_BPS,
) -> pd.DataFrame:
    """Backtest the optimizer against 1/N — the baseline that estimates nothing.

    If shrinkage-based optimization cannot beat equal weight net of costs, that
    is the honest finding and it belongs in the results table.
    """
    equal = pd.DataFrame(
        1.0 / len(returns.columns),
        index=optimized_weights.index,
        columns=returns.columns,
    )

    rows = []
    for name, w in (("markowitz_lw", optimized_weights), ("equal_weight", equal)):
        result = backtest(w, returns, cost_bps=cost_bps)
        rows.append({"strategy": name, **result.as_dict()})

    return pd.DataFrame(rows)
