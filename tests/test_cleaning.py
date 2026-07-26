"""Outlier handling: robust, causal, and conservative during warm-up."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfolio.transforms.cleaning import clean_prices, clip_outliers, rolling_mad


def test_rolling_mad_is_zero_on_a_constant_series() -> None:
    constant = pd.Series([5.0] * 100)
    assert rolling_mad(constant, 20).dropna().eq(0.0).all()


def test_a_single_spike_is_clipped(price_frame: pd.DataFrame) -> None:
    returns = np.log(price_frame["adj_close"] / price_frame["adj_close"].shift(1))
    spiked = returns.copy()
    spiked.iloc[250] = 0.9  # a 145% one-day move: a bad tick, not a market event

    clipped = clip_outliers(spiked, window=63, scale=8.0)
    assert clipped.iloc[250] < spiked.iloc[250]


def test_ordinary_moves_survive_untouched(price_frame: pd.DataFrame) -> None:
    """Clipping must be rare. A filter that trims normal days destroys signal."""
    returns = np.log(price_frame["adj_close"] / price_frame["adj_close"].shift(1))
    clipped = clip_outliers(returns, window=63, scale=8.0)

    changed = int(((clipped != returns) & returns.notna()).sum())
    assert changed == 0, f"{changed} ordinary returns were clipped from a clean series"


def test_warmup_period_is_left_alone() -> None:
    """With too few observations the bound is meaningless, so nothing is clipped."""
    series = pd.Series([0.01, -0.02, 0.5, 0.01])
    clipped = clip_outliers(series, window=63, scale=8.0)
    assert clipped.iloc[2] == pytest.approx(0.5)


def test_clean_prices_preserves_the_series_when_nothing_is_extreme(
    price_frame: pd.DataFrame,
) -> None:
    result = clean_prices(price_frame, window=63, scale=8.0)
    pd.testing.assert_series_equal(result["adj_close"], price_frame["adj_close"], check_names=False)
    assert not result["is_outlier_adjusted"].any()


def test_clean_prices_damps_a_bad_tick(price_frame: pd.DataFrame) -> None:
    corrupted = price_frame.copy()
    corrupted.loc[250, "adj_close"] *= 4.0

    result = clean_prices(corrupted, window=63, scale=8.0)

    assert result["is_outlier_adjusted"].any(), "a 4x one-day jump should be flagged"
    assert result.loc[250, "adj_close"] < corrupted.loc[250, "adj_close"]
    # The original value is kept alongside — clipping is recorded, not hidden.
    assert result.loc[250, "adj_close_raw"] == pytest.approx(corrupted.loc[250, "adj_close"])


def test_prices_before_a_bad_tick_are_bit_identical(price_frame: pd.DataFrame) -> None:
    """The adjustment must leave the untouched prefix exactly as it was."""
    corrupted = price_frame.copy()
    corrupted.loc[250, "adj_close"] *= 4.0

    result = clean_prices(corrupted, window=63, scale=8.0)

    pd.testing.assert_series_equal(
        result.loc[:249, "adj_close"],
        price_frame.loc[:249, "adj_close"],
        check_names=False,
        rtol=0,
        atol=0,
    )


def test_empty_frame_is_returned_unchanged() -> None:
    empty = pd.DataFrame({"date": [], "adj_close": []})
    assert clean_prices(empty).empty
