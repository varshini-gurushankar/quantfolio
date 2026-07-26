"""Proof that no feature reads the future.

The argument these tests make is empirical rather than by inspection: take a
price history, compute every feature, then reach into the series and change a
value at some date *t*. If any feature dated before *t* moves, that feature
depended on information that did not exist yet.

This is the property that makes a backtest on this data meaningful, and it is
the first thing worth checking after any change to `transforms/`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfolio.transforms.cleaning import clean_prices, clip_outliers
from quantfolio.transforms.features import (
    FEATURE_COLUMNS,
    compute_features,
    compute_features_by_ticker,
    next_day_log_return,
    rolling_volatility,
    rsi,
    sma,
)

PERTURBATION_POINTS = [120, 200, 305, 399]


def _perturb(frame: pd.DataFrame, index: int, factor: float = 1.35) -> pd.DataFrame:
    """Move every price on one date, as a bad tick or a late correction would."""
    out = frame.copy()
    for col in ("open", "high", "low", "close", "adj_close"):
        out.loc[index, col] = out.loc[index, col] * factor
    return out


@pytest.mark.parametrize("cut", PERTURBATION_POINTS)
def test_features_before_a_perturbation_are_unchanged(price_frame: pd.DataFrame, cut: int) -> None:
    """The core assertion: changing day t cannot move any feature dated before t."""
    baseline = compute_features(price_frame)
    perturbed = compute_features(_perturb(price_frame, cut))

    before_baseline = baseline.loc[: cut - 1, FEATURE_COLUMNS]
    before_perturbed = perturbed.loc[: cut - 1, FEATURE_COLUMNS]

    pd.testing.assert_frame_equal(
        before_baseline,
        before_perturbed,
        check_exact=False,
        rtol=0,
        atol=0,
        obj=f"features strictly before the perturbation at index {cut}",
    )


@pytest.mark.parametrize("cut", PERTURBATION_POINTS)
def test_perturbation_does_reach_the_present(price_frame: pd.DataFrame, cut: int) -> None:
    """Guard against a vacuous pass.

    If the previous test passed because the features were all NaN or constant,
    it would prove nothing. The perturbed day itself must actually change.
    """
    baseline = compute_features(price_frame)
    perturbed = compute_features(_perturb(price_frame, cut))

    changed = [
        col
        for col in FEATURE_COLUMNS
        if not np.allclose(
            baseline.loc[cut, col], perturbed.loc[cut, col], equal_nan=True, rtol=1e-12
        )
    ]
    assert changed, f"perturbation at index {cut} changed nothing — the test would be vacuous"


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(lambda s: sma(s, 20), id="sma_20"),
        pytest.param(lambda s: rsi(s, 14), id="rsi_14"),
        pytest.param(lambda s: rolling_volatility(np.log(s / s.shift(1)), 20), id="volatility_20"),
        pytest.param(lambda s: clip_outliers(s, window=63, scale=8.0), id="clip_outliers"),
    ],
)
def test_individual_transforms_are_causal(price_frame: pd.DataFrame, func) -> None:
    """Each transform in isolation, so a failure points at one function."""
    series = price_frame["adj_close"].astype(float)
    cut = 250

    perturbed_series = series.copy()
    perturbed_series.iloc[cut] *= 1.5

    baseline = func(series)
    perturbed = func(perturbed_series)

    pd.testing.assert_series_equal(
        baseline.iloc[:cut],
        perturbed.iloc[:cut],
        check_exact=False,
        rtol=0,
        atol=0,
    )


def test_truncating_the_future_does_not_change_the_past(price_frame: pd.DataFrame) -> None:
    """A live system only ever has data up to today.

    Computing features on a truncated history must reproduce exactly what the
    full-history run produced for those same dates — otherwise yesterday's
    feature value depends on data that arrives tomorrow.
    """
    cut = 300
    full = compute_features(price_frame)
    truncated = compute_features(price_frame.iloc[:cut].copy())

    pd.testing.assert_frame_equal(
        full.loc[: cut - 1, FEATURE_COLUMNS],
        truncated.loc[: cut - 1, FEATURE_COLUMNS],
        check_exact=False,
        rtol=0,
        atol=0,
        obj="features computed on truncated vs full history",
    )


def test_cleaning_is_causal(price_frame: pd.DataFrame) -> None:
    """Outlier clipping must not use a bound estimated from future prices."""
    cut = 220
    baseline = clean_prices(price_frame, window=63, scale=8.0)
    perturbed = clean_prices(_perturb(price_frame, cut, factor=3.0), window=63, scale=8.0)

    pd.testing.assert_series_equal(
        baseline.loc[: cut - 1, "adj_close"],
        perturbed.loc[: cut - 1, "adj_close"],
        check_exact=False,
        rtol=0,
        atol=0,
    )


def test_multi_ticker_windows_do_not_bleed(multi_ticker_frame: pd.DataFrame) -> None:
    """One ticker's prices must not enter another ticker's rolling window."""
    computed = compute_features_by_ticker(multi_ticker_frame)

    perturbed_input = multi_ticker_frame.copy()
    other_rows = perturbed_input.index[perturbed_input["ticker"] == "OTHER"]
    perturbed_input.loc[other_rows, ["close", "adj_close"]] *= 2.0
    perturbed = compute_features_by_ticker(perturbed_input)

    baseline_test = computed[computed["ticker"] == "TEST"][FEATURE_COLUMNS].reset_index(drop=True)
    perturbed_test = perturbed[perturbed["ticker"] == "TEST"][FEATURE_COLUMNS].reset_index(
        drop=True
    )

    pd.testing.assert_frame_equal(baseline_test, perturbed_test, check_exact=False, rtol=0, atol=0)


def test_target_is_the_only_forward_looking_column(price_frame: pd.DataFrame) -> None:
    """The label looks forward by exactly one day; the last row must be unknowable."""
    features = compute_features(price_frame)
    features["ticker"] = "TEST"
    target = next_day_log_return(features)

    assert pd.isna(target.iloc[-1]), "the final target must be NaN — tomorrow has not happened"

    # target[t] must equal the realised return at t+1, not at t.
    aligned = target.iloc[:-1].reset_index(drop=True)
    realised_next = features["log_return"].iloc[1:].reset_index(drop=True)
    pd.testing.assert_series_equal(aligned, realised_next, check_names=False)


def test_no_feature_column_correlates_with_its_own_future(price_frame: pd.DataFrame) -> None:
    """A blunt smoke test for the classic mistake of shifting the wrong way.

    A feature that accidentally contains tomorrow's price correlates almost
    perfectly with the next-day return. Real features correlate weakly at best.
    """
    features = compute_features(price_frame)
    features["ticker"] = "TEST"
    target = next_day_log_return(features)

    for col in FEATURE_COLUMNS:
        pair = pd.concat([features[col], target], axis=1).dropna()
        if len(pair) < 50 or pair.iloc[:, 0].std() == 0:
            continue
        corr = abs(pair.corr().iloc[0, 1])
        assert corr < 0.5, f"{col} correlates {corr:.3f} with the next-day return — likely leakage"
