"""Walk-forward splitting and dataset construction.

The tests that matter here are the ones that would catch a leak: train folds
strictly preceding test folds, scalers fit on training data only, and a target
that is genuinely the *next* day's return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfolio.models.datasets import (
    MODEL_FEATURES,
    TARGET,
    apply_split,
    build_supervised,
    fit_scaler,
    make_sequences,
    normalize_price_features,
    transform,
    walk_forward_splits,
)
from quantfolio.transforms.features import compute_features_by_ticker


@pytest.fixture
def panel(multi_ticker_frame: pd.DataFrame) -> pd.DataFrame:
    return compute_features_by_ticker(multi_ticker_frame)


@pytest.fixture
def dataset(panel: pd.DataFrame) -> pd.DataFrame:
    return build_supervised(panel)


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def test_train_always_precedes_test(dataset: pd.DataFrame) -> None:
    """The defining property: no fold may train on data after its test window."""
    for split in walk_forward_splits(dataset["date"], n_splits=3, test_size=40, min_train_size=100):
        assert split.train_end < split.test_start, f"{split} trains on its own future"
        assert split.train_start <= split.train_end
        assert split.test_start <= split.test_end


def test_expanding_window_grows(dataset: pd.DataFrame) -> None:
    splits = walk_forward_splits(dataset["date"], n_splits=3, test_size=40, min_train_size=100)
    starts = [s.train_start for s in splits]
    ends = [s.train_end for s in splits]

    assert len(set(starts)) == 1, "an expanding window keeps the same start"
    assert ends == sorted(ends), "each fold must train on strictly more history"


def test_rolling_window_moves(dataset: pd.DataFrame) -> None:
    splits = walk_forward_splits(
        dataset["date"], n_splits=3, test_size=40, min_train_size=100, expanding=False
    )
    starts = [s.train_start for s in splits]
    assert starts == sorted(starts) and len(set(starts)) > 1, "a rolling window drops old data"


def test_test_windows_do_not_overlap(dataset: pd.DataFrame) -> None:
    """Overlapping test folds would count the same day twice in the aggregate."""
    splits = walk_forward_splits(dataset["date"], n_splits=3, test_size=40, min_train_size=100)
    for earlier, later in zip(splits, splits[1:], strict=False):
        assert earlier.test_end < later.test_start


def test_final_fold_tests_the_most_recent_data(dataset: pd.DataFrame) -> None:
    splits = walk_forward_splits(dataset["date"], n_splits=3, test_size=40, min_train_size=100)
    last_date = pd.to_datetime(dataset["date"]).max().date()
    assert splits[-1].test_end == last_date


def test_too_little_history_raises() -> None:
    dates = pd.bdate_range("2023-01-02", periods=30)
    with pytest.raises(ValueError, match="need at least"):
        walk_forward_splits(dates, n_splits=5, test_size=63, min_train_size=252)


def test_split_count_is_reduced_rather_than_failing() -> None:
    """Slightly short history should degrade gracefully, not blow up."""
    dates = pd.bdate_range("2022-01-03", periods=400)
    splits = walk_forward_splits(dates, n_splits=5, test_size=63, min_train_size=252)
    assert 0 < len(splits) < 5


def test_apply_split_partitions_by_date(dataset: pd.DataFrame) -> None:
    split = walk_forward_splits(dataset["date"], n_splits=2, test_size=40, min_train_size=100)[0]
    train, test = apply_split(dataset, split)

    assert not train.empty and not test.empty
    assert pd.to_datetime(train["date"]).max().date() <= split.train_end
    assert pd.to_datetime(test["date"]).min().date() >= split.test_start
    # No date may appear in both halves — the multi-ticker leak.
    assert not set(train["date"]) & set(test["date"])


def test_split_keeps_all_tickers_on_both_sides(dataset: pd.DataFrame) -> None:
    """Splitting by date, not by row, keeps the cross-section intact."""
    split = walk_forward_splits(dataset["date"], n_splits=2, test_size=40, min_train_size=100)[0]
    train, test = apply_split(dataset, split)
    assert set(train["ticker"]) == set(test["ticker"])


# --------------------------------------------------------------------------- #
# target construction
# --------------------------------------------------------------------------- #
def test_target_is_tomorrows_return(panel: pd.DataFrame) -> None:
    dataset = build_supervised(panel)
    one = dataset[dataset["ticker"] == "TEST"].sort_values("date").reset_index(drop=True)
    source = panel[panel["ticker"] == "TEST"].sort_values("date").reset_index(drop=True)

    lookup = dict(zip(source["date"], source["log_return"], strict=True))
    dates = list(source["date"])

    for _, row in one.head(20).iterrows():
        next_date = dates[dates.index(row["date"]) + 1]
        assert row[TARGET] == pytest.approx(lookup[next_date])


def test_target_never_equals_the_same_day_return(dataset: pd.DataFrame) -> None:
    """Guards against an off-by-one that would make the task trivially easy."""
    identical = np.isclose(dataset[TARGET], dataset["log_return"])
    assert identical.mean() < 0.05, "target looks like the same-day return"


def test_rows_without_a_known_target_are_dropped(dataset: pd.DataFrame) -> None:
    assert dataset[TARGET].notna().all()
    for _, group in dataset.groupby("ticker"):
        assert group[MODEL_FEATURES].notna().all().all()


def test_last_session_per_ticker_is_excluded(panel: pd.DataFrame) -> None:
    """The most recent day has no known tomorrow, so it cannot be a training row."""
    dataset = build_supervised(panel)
    for ticker, group in panel.groupby("ticker"):
        last = pd.to_datetime(group["date"]).max()
        kept = dataset[dataset["ticker"] == ticker]
        assert pd.to_datetime(kept["date"]).max() < last


# --------------------------------------------------------------------------- #
# feature normalization
# --------------------------------------------------------------------------- #
def test_price_features_become_scale_free(panel: pd.DataFrame) -> None:
    """SMA in dollars encodes the price level; as a ratio it does not."""
    normalized = normalize_price_features(panel.dropna(subset=["sma_20"]))
    ratios = normalized["sma_20"].dropna()
    assert ratios.between(0.5, 1.5).mean() > 0.95


def test_rsi_is_centred(panel: pd.DataFrame) -> None:
    normalized = normalize_price_features(panel.dropna(subset=["rsi_14"]))
    assert normalized["rsi_14"].dropna().between(-1.0, 1.0).all()


def test_normalization_uses_only_same_day_values(panel: pd.DataFrame) -> None:
    """Dividing by the same row's price introduces no time dependence at all."""
    clean = panel.dropna(subset=["sma_20"]).reset_index(drop=True)
    cut = len(clean) // 2

    full = normalize_price_features(clean)
    truncated = normalize_price_features(clean.iloc[:cut].copy())

    pd.testing.assert_series_equal(
        full.loc[: cut - 1, "sma_20"], truncated["sma_20"], check_names=False
    )


# --------------------------------------------------------------------------- #
# scaling
# --------------------------------------------------------------------------- #
def test_scaler_is_fit_on_training_data_only(dataset: pd.DataFrame) -> None:
    """The most common silent leak in a walk-forward setup."""
    split = walk_forward_splits(dataset["date"], n_splits=2, test_size=40, min_train_size=100)[0]
    train, test = apply_split(dataset, split)

    train_mean, _ = fit_scaler(train, MODEL_FEATURES)
    full_mean, _ = fit_scaler(dataset, MODEL_FEATURES)

    assert not np.allclose(train_mean, full_mean), (
        "training-fold scaler matches the full-panel scaler; it may have seen the test set"
    )


def test_scaled_training_features_are_standardized(dataset: pd.DataFrame) -> None:
    mean, std = fit_scaler(dataset, MODEL_FEATURES)
    scaled = transform(dataset, MODEL_FEATURES, mean, std)

    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-9)


def test_constant_feature_does_not_produce_infinities() -> None:
    frame = pd.DataFrame({"a": [1.0] * 20, "b": np.arange(20.0)})
    mean, std = fit_scaler(frame, ["a", "b"])
    scaled = transform(frame, ["a", "b"], mean, std)
    assert np.isfinite(scaled).all()


# --------------------------------------------------------------------------- #
# sequences
# --------------------------------------------------------------------------- #
def test_sequences_have_the_expected_shape(dataset: pd.DataFrame) -> None:
    mean, std = fit_scaler(dataset, MODEL_FEATURES)
    x, y, meta = make_sequences(dataset, MODEL_FEATURES, mean, std, lookback=10)

    assert x.ndim == 3
    assert x.shape[1] == 10
    assert x.shape[2] == len(MODEL_FEATURES)
    assert len(x) == len(y) == len(meta)


def test_sequences_never_span_two_tickers(dataset: pd.DataFrame) -> None:
    """A window crossing a ticker boundary would blend two unrelated histories."""
    mean, std = fit_scaler(dataset, MODEL_FEATURES)
    lookback = 10
    _, _, meta = make_sequences(dataset, MODEL_FEATURES, mean, std, lookback=lookback)

    per_ticker = meta.groupby("ticker").size()
    for ticker, count in per_ticker.items():
        available = len(dataset[dataset["ticker"] == ticker])
        assert count == available - lookback + 1


def test_sequence_ends_on_the_labelled_row(dataset: pd.DataFrame) -> None:
    """Window t must cover [t-lookback+1, t] with the label from t+1."""
    lookback = 5
    one = dataset[dataset["ticker"] == "TEST"].sort_values("date").reset_index(drop=True)
    mean, std = fit_scaler(one, MODEL_FEATURES)
    x, y, meta = make_sequences(one, MODEL_FEATURES, mean, std, lookback=lookback)

    expected_first_date = one["date"].iloc[lookback - 1]
    assert meta["date"].iloc[0] == expected_first_date
    assert y[0] == pytest.approx(one[TARGET].iloc[lookback - 1], rel=1e-5)

    last_row = transform(one.iloc[[lookback - 1]], MODEL_FEATURES, mean, std)
    np.testing.assert_allclose(x[0][-1], last_row[0], rtol=1e-5)


def test_too_short_a_history_yields_no_sequences() -> None:
    frame = pd.DataFrame(
        {
            "ticker": "X",
            "date": pd.bdate_range("2023-01-02", periods=5),
            TARGET: 0.001,
            **{c: 0.5 for c in MODEL_FEATURES},
        }
    )
    mean, std = fit_scaler(frame, MODEL_FEATURES)
    x, y, meta = make_sequences(frame, MODEL_FEATURES, mean, std, lookback=20)

    assert len(x) == 0 and len(y) == 0 and meta.empty
