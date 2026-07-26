"""Markowitz mean-variance optimization with Ledoit-Wolf shrinkage.

Markowitz's problem in practice is not the mathematics, it is the covariance
estimate. The sample covariance matrix of N assets from T observations has
N(N+1)/2 parameters estimated from NT numbers, so its extreme eigenvalues are
badly biased: the smallest ones are too small. The optimizer then does exactly
what it was asked to do and piles into whatever combination looks least risky,
which is usually the direction where the estimate is most wrong. That is the
"error maximisation" critique, and it is why naively optimized portfolios so
often underperform equal weighting out of sample.

Ledoit-Wolf shrinkage is the standard answer: pull the sample covariance toward
a structured target (here, scaled identity) by an analytically chosen intensity.
The result is better conditioned, always invertible, and much more stable
period-to-period — which also means less turnover and lower transaction costs.

The long-only and per-asset-cap constraints do similar work from a different
direction: they bound how badly a single bad estimate can distort the portfolio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


@dataclass
class OptimizationResult:
    """Weights plus the diagnostics needed to judge whether to trust them."""

    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float
    status: str
    method: str = "markowitz_ledoit_wolf"
    shrinkage: float | None = None
    as_of: date | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def is_optimal(self) -> bool:
        return self.status in ("optimal", "optimal_inaccurate")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "as_of_date": self.as_of,
                "ticker": self.weights.index,
                "weight": self.weights.to_numpy(),
                "method": self.method,
            }
        )

    def summary(self) -> str:
        top = self.weights.sort_values(ascending=False).head(3)
        holdings = ", ".join(f"{t} {w:.1%}" for t, w in top.items())
        return (
            f"{self.method} [{self.status}]: vol {self.volatility:.2%}, "
            f"E[r] {self.expected_return:.2%}, Sharpe {self.sharpe:.2f} | top: {holdings}"
        )


def returns_matrix(features: pd.DataFrame, value_column: str = "log_return") -> pd.DataFrame:
    """Pivot a long feature panel into a (date x ticker) return matrix.

    Dates where any ticker is missing are dropped: a covariance estimated from
    ragged data silently weights assets by how much history they happen to have.
    """
    wide = features.pivot_table(index="date", columns="ticker", values=value_column)
    before = len(wide)
    wide = wide.dropna(how="any")
    if before != len(wide):
        logger.info("dropped %d dates with incomplete cross-sections", before - len(wide))
    return wide.sort_index()


def ledoit_wolf_covariance(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Shrunk covariance matrix and the shrinkage intensity chosen for it.

    The intensity is derived analytically from the data, not tuned: it is the
    value minimising expected squared error against the true covariance.
    """
    from sklearn.covariance import LedoitWolf

    estimator = LedoitWolf().fit(returns.to_numpy(dtype=np.float64))
    return estimator.covariance_, float(estimator.shrinkage_)


def optimize(
    returns: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    target_return: float | None = None,
    max_weight: float = 0.30,
    min_weight: float = 0.0,
    use_shrinkage: bool = True,
    as_of: date | None = None,
) -> OptimizationResult:
    """Minimum-variance portfolio, optionally subject to a return floor.

    Constraints: fully invested (weights sum to 1), long only by default, and no
    single asset above ``max_weight``. The cap is what stops the optimizer
    answering "put 94% in the one asset whose estimated variance came out
    lowest", which is rarely a statement about the world.

    Returns and covariance are annualized so the numbers are readable.
    """
    import cvxpy as cp

    if returns.empty or returns.shape[1] == 0:
        raise ValueError("cannot optimize an empty return matrix")

    tickers = list(returns.columns)
    n = len(tickers)

    if n * max_weight < 1.0 - 1e-9:
        raise ValueError(
            f"max_weight {max_weight} across {n} assets cannot reach a fully invested "
            f"portfolio; need at least {1.0 / n:.3f}"
        )

    if use_shrinkage:
        cov_daily, shrinkage = ledoit_wolf_covariance(returns)
    else:
        cov_daily, shrinkage = returns.cov().to_numpy(dtype=np.float64), None

    cov = cov_daily * TRADING_DAYS_PER_YEAR
    # cvxpy needs exact symmetry; floating-point noise can break the PSD check.
    cov = (cov + cov.T) / 2.0

    if expected_returns is None:
        mu = returns.mean().reindex(tickers).to_numpy(dtype=np.float64) * TRADING_DAYS_PER_YEAR
    else:
        mu = expected_returns.reindex(tickers).to_numpy(dtype=np.float64) * TRADING_DAYS_PER_YEAR

    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= min_weight, w <= max_weight]

    if target_return is not None:
        constraints.append(mu @ w >= target_return)

    # psd_wrap tells cvxpy to trust that the shrunk matrix is PSD, which Ledoit-
    # Wolf guarantees, instead of re-verifying it at every solve.
    problem = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), constraints)

    try:
        problem.solve(solver=cp.CLARABEL)
    except Exception as exc:  # noqa: BLE001 - fall back rather than fail the DAG
        logger.warning("CLARABEL failed (%s); retrying with SCS", exc)
        problem.solve(solver=cp.SCS)

    if w.value is None:
        # Infeasible almost always means the return floor is unreachable given
        # the caps. Equal weight is a defensible, honest fallback.
        logger.error(
            "optimization failed with status %s; falling back to equal weight", problem.status
        )
        weights = pd.Series(np.full(n, 1.0 / n), index=tickers)
        status = f"failed_{problem.status}"
    else:
        raw = np.asarray(w.value).ravel()
        # The solver returns tiny negatives like -1e-12; clip and renormalise so
        # the weights that get stored are exactly valid.
        raw = np.clip(raw, min_weight, max_weight)
        weights = pd.Series(raw / raw.sum(), index=tickers)
        status = problem.status

    variance = float(weights.to_numpy() @ cov @ weights.to_numpy())
    volatility = float(np.sqrt(max(variance, 0.0)))
    expected = float(mu @ weights.to_numpy())

    result = OptimizationResult(
        weights=weights,
        expected_return=expected,
        volatility=volatility,
        sharpe=expected / volatility if volatility > 0 else 0.0,
        status=status,
        method="markowitz_ledoit_wolf" if use_shrinkage else "markowitz_sample_cov",
        shrinkage=shrinkage,
        as_of=as_of or (returns.index[-1] if len(returns) else None),
        diagnostics={
            "n_assets": n,
            "n_observations": len(returns),
            "max_weight": max_weight,
            "target_return": target_return,
            "condition_number": float(np.linalg.cond(cov)),
            "effective_n": float(1.0 / np.sum(weights.to_numpy() ** 2)),
        },
    )
    logger.info("%s", result.summary())
    return result


def equal_weight(tickers: list[str], as_of: date | None = None) -> OptimizationResult:
    """The benchmark Markowitz has to beat.

    Naive 1/N is a famously hard baseline precisely because it estimates
    nothing, so it has no estimation error to amplify.
    """
    n = len(tickers)
    weights = pd.Series(np.full(n, 1.0 / n), index=tickers)
    return OptimizationResult(
        weights=weights,
        expected_return=float("nan"),
        volatility=float("nan"),
        sharpe=float("nan"),
        status="optimal",
        method="equal_weight",
        as_of=as_of,
    )


def store_weights(result: OptimizationResult, engine=None) -> int:
    """Persist weights to Postgres, upserted on ``(as_of_date, ticker)``."""
    from quantfolio.storage.db import upsert
    from quantfolio.storage.schema import portfolio_weights

    frame = result.to_frame()
    if frame["as_of_date"].isna().any():
        raise ValueError("cannot store weights without an as_of date")
    return upsert(frame, portfolio_weights, engine=engine)
