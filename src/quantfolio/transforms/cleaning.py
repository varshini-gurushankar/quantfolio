"""Causal outlier handling.

The tempting version of this file computes a mean and standard deviation over
the whole series and clips anything beyond N sigma. That leaks: the bound
applied at 2016-03-01 would depend on prices from 2024, so a backtest run over
the cleaned series is quietly reading the future.

Everything here uses trailing windows only. The bound at time *t* is built from
observations at *t* and earlier, which is exactly what a live system would have
had. Median and MAD rather than mean and standard deviation because the point is
to be robust to the very outliers being detected.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# MAD * this constant estimates the standard deviation for normal data.
_MAD_TO_SIGMA = 1.4826


def rolling_mad(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Trailing median absolute deviation.

    ``.rolling()`` is right-closed, so the value at *t* uses ``[t-window+1, t]``
    and nothing later.
    """
    min_periods = min_periods or max(2, window // 4)
    med = series.rolling(window, min_periods=min_periods).median()
    return (series - med).abs().rolling(window, min_periods=min_periods).median()


def clip_outliers(
    series: pd.Series,
    window: int = 63,
    scale: float = 8.0,
    min_periods: int | None = None,
) -> pd.Series:
    """Clip to ``rolling_median +/- scale * sigma_hat`` using trailing windows only.

    Early observations, where the window has not filled, are left untouched
    rather than clipped against a bound estimated from two points.
    """
    min_periods = min_periods or max(2, window // 4)
    med = series.rolling(window, min_periods=min_periods).median()
    sigma = _MAD_TO_SIGMA * rolling_mad(series, window, min_periods)

    lower = med - scale * sigma
    upper = med + scale * sigma

    # Where the bound is undefined (warm-up, or a flat window with MAD == 0),
    # fall back to the observation itself: no bound is better than a fake one.
    lower = lower.where(sigma > 0)
    upper = upper.where(sigma > 0)

    clipped = series.clip(lower=lower, upper=upper)
    # NaN != NaN, so the notna mask keeps warm-up rows out of the count.
    n_clipped = int(((clipped != series) & series.notna()).sum())
    if n_clipped:
        logger.info(
            "clipped %d outlier observations (window=%d, scale=%.1f)", n_clipped, window, scale
        )
    return clipped


def clean_prices(
    frame: pd.DataFrame,
    window: int = 63,
    scale: float = 8.0,
    column: str = "adj_close",
) -> pd.DataFrame:
    """Clip the price column of one ticker's frame, causally.

    Clipping is applied to *returns* rather than the price level: a price series
    trends, so a rolling median of the level flags legitimate drift as an
    outlier, while a return series is roughly stationary and a bad tick shows up
    as exactly the spike this is meant to catch.
    """
    if frame.empty:
        return frame

    out = frame.sort_values("date").copy()
    price = out[column].astype(float)

    raw_ret = np.log(price / price.shift(1))
    clipped_ret = clip_outliers(raw_ret, window=window, scale=scale)

    was_clipped = (clipped_ret != raw_ret) & raw_ret.notna()
    if was_clipped.any():
        # Apply the clipping as a cumulative multiplicative adjustment rather
        # than rebuilding the level from scratch. Before the first clipped
        # return the adjustment is exp(0) == 1.0 exactly, so untouched prices
        # are preserved bit-for-bit instead of being re-derived through a
        # cumsum — which would perturb the past at the 1e-15 level and make a
        # causality test fail for purely numerical reasons.
        adjustment = np.exp((clipped_ret - raw_ret).fillna(0.0).cumsum())
        out[f"{column}_raw"] = price
        out[column] = (price * adjustment).where(price.notna())
        out["is_outlier_adjusted"] = was_clipped.fillna(False)
    else:
        out["is_outlier_adjusted"] = False

    return out
