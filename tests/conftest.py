"""Shared fixtures.

Tests run against synthetic series rather than live market data: a test that
needs the internet is a test that fails for reasons unrelated to the code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfolio.config import FeatureParams

N_DAYS = 400
SEED = 20240101


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture
def sessions() -> pd.DatetimeIndex:
    """Business days, standing in for exchange sessions."""
    return pd.bdate_range("2022-01-03", periods=N_DAYS)


@pytest.fixture
def price_frame(sessions: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """A single ticker's OHLCV history: geometric random walk with drift."""
    returns = rng.normal(loc=0.0004, scale=0.012, size=len(sessions))
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.002, len(sessions)))

    # High and low are derived from the open/close pair rather than drawn
    # independently, so every bar satisfies low <= {open, close} <= high — the
    # same invariant check_ohlc_consistency enforces on real data.
    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)

    return pd.DataFrame(
        {
            "ticker": "TEST",
            "date": sessions,
            "open": open_,
            "high": body_high * (1 + np.abs(rng.normal(0, 0.004, len(sessions)))),
            "low": body_low * (1 - np.abs(rng.normal(0, 0.004, len(sessions)))),
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1_000_000, 5_000_000, len(sessions)),
        }
    )


@pytest.fixture
def multi_ticker_frame(price_frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Two tickers with different levels, to catch cross-ticker window bleed."""
    other = price_frame.copy()
    other["ticker"] = "OTHER"
    scale = np.exp(np.cumsum(rng.normal(0.0002, 0.015, len(other))))
    for col in ("open", "high", "low", "close", "adj_close"):
        other[col] = 250.0 * scale
    return pd.concat([price_frame, other], ignore_index=True)


@pytest.fixture
def feature_params() -> FeatureParams:
    return FeatureParams()
