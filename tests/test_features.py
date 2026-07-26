"""Feature correctness — values, not just causality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfolio.transforms.features import (
    FEATURE_COLUMNS,
    compute_features,
    compute_features_by_ticker,
    ema,
    macd,
    rolling_volatility,
    rsi,
    simple_returns,
    sma,
)


def test_simple_and_log_returns_agree_for_small_moves() -> None:
    price = pd.Series([100.0, 100.5, 101.0, 100.2])
    simple = simple_returns(price)
    log = np.log(price / price.shift(1))
    pd.testing.assert_series_equal(simple, np.expm1(log), check_names=False)


def test_sma_matches_a_hand_computed_window() -> None:
    price = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(price, 3)

    assert result.iloc[:2].isna().all(), "window must not emit a value before it is full"
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_ema_is_the_recursive_form() -> None:
    price = pd.Series(np.arange(1.0, 21.0))
    span = 5
    alpha = 2.0 / (span + 1)
    result = ema(price, span)

    # Once warmed up, each value must satisfy e_t = alpha*p_t + (1-alpha)*e_{t-1}.
    for t in range(span + 1, len(price)):
        expected = alpha * price.iloc[t] + (1 - alpha) * result.iloc[t - 1]
        assert result.iloc[t] == pytest.approx(expected)


def test_macd_histogram_is_line_minus_signal(price_frame: pd.DataFrame) -> None:
    line, signal, hist = macd(price_frame["adj_close"])
    pd.testing.assert_series_equal(hist, line - signal, check_names=False)


def test_rsi_stays_within_bounds(price_frame: pd.DataFrame) -> None:
    values = rsi(price_frame["adj_close"], 14).dropna()
    assert not values.empty
    assert values.min() >= 0.0
    assert values.max() <= 100.0


def test_rsi_saturates_on_a_monotonic_series() -> None:
    rising = pd.Series(np.linspace(100, 200, 60))
    falling = pd.Series(np.linspace(200, 100, 60))

    assert rsi(rising, 14).dropna().iloc[-1] == pytest.approx(100.0)
    assert rsi(falling, 14).dropna().iloc[-1] == pytest.approx(0.0)


def test_rsi_on_a_flat_series_is_neutral() -> None:
    """0/0 is undefined, not extreme — a flat market is neither overbought nor oversold."""
    flat = pd.Series([100.0] * 60)
    assert rsi(flat, 14).dropna().iloc[-1] == pytest.approx(50.0)


def test_volatility_is_annualized() -> None:
    returns = pd.Series(np.full(100, 0.01))
    returns.iloc[::2] = -0.01  # alternating, so the sample std is stable

    daily = rolling_volatility(returns, 20, annualize=False).iloc[-1]
    annual = rolling_volatility(returns, 20, annualize=True).iloc[-1]
    assert annual == pytest.approx(daily * np.sqrt(252))


def test_warmup_rows_are_nan_not_backfilled(price_frame: pd.DataFrame) -> None:
    """A partially filled window must produce NaN rather than a value computed from less data."""
    features = compute_features(price_frame)

    assert features["sma_20"].iloc[:19].isna().all()
    assert features["sma_60"].iloc[:59].isna().all()
    assert features["sma_20"].iloc[19:].notna().all()


def test_all_declared_feature_columns_are_produced(price_frame: pd.DataFrame) -> None:
    features = compute_features(price_frame)
    missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
    assert not missing, f"compute_features did not produce {missing}"


def test_per_ticker_grouping_preserves_row_count(multi_ticker_frame: pd.DataFrame) -> None:
    result = compute_features_by_ticker(multi_ticker_frame)
    assert len(result) == len(multi_ticker_frame)
    assert set(result["ticker"]) == {"TEST", "OTHER"}


def test_features_are_independent_of_input_row_order(price_frame: pd.DataFrame) -> None:
    """Sources do not guarantee ordering; the transform sorts before computing."""
    shuffled = price_frame.sample(frac=1.0, random_state=7).reset_index(drop=True)

    expected = compute_features(price_frame)[FEATURE_COLUMNS]
    actual = compute_features(shuffled)[FEATURE_COLUMNS]

    pd.testing.assert_frame_equal(expected, actual)


def test_empty_input_returns_empty_frame_with_columns() -> None:
    empty = pd.DataFrame(
        {"ticker": [], "date": [], "adj_close": []},
    )
    result = compute_features(empty)
    assert result.empty
