"""Supervised dataset construction and walk-forward splitting.

Two decisions here carry most of the statistical weight of Phase 2.

**The target is the next-day log return, not the price.** Predicting price
levels produces a flattering, meaningless MSE: prices are near-random walks, so
"predict today's price for tomorrow" scores extremely well and means nothing. A
return target has no such autocorrelation to lean on, which is why the numbers
it produces are small, honest, and comparable across models.

**Splits are walk-forward, never random.** A random split trains on 2024 and
tests on 2019, which is not a thing any deployed model can do. Every split here
trains on a contiguous past and tests on the future immediately after it.

The subtler trap is feature scaling. Fitting a scaler on the full series leaks
the test period's mean and variance into training, so ``fit_scaler`` is always
fit on the training fold alone and then applied to the test fold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Features the models consume. Deliberately excludes raw price levels: an
# absolute price is not comparable across tickers or across time, and a model
# handed one tends to learn the ticker rather than the signal.
MODEL_FEATURES = [
    "log_return",
    "sma_20",
    "sma_60",
    "ema_20",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi_14",
    "volatility_20",
]

TARGET = "target_next_log_return"


@dataclass(frozen=True)
class Split:
    """One walk-forward fold, identified by the dates at its boundaries."""

    index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date

    def __str__(self) -> str:
        return (
            f"fold {self.index}: train {self.train_start}..{self.train_end}, "
            f"test {self.test_start}..{self.test_end}"
        )


def normalize_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert price-level features into scale-free ratios.

    SMA and EMA are in dollars, so their raw values encode the ticker's price
    level rather than anything about its behaviour. Dividing by the closing
    price turns "the 20-day average is $187.34" into "price is 1.02x its 20-day
    average", which is comparable across AAPL and SPY and across a decade.

    This uses only same-day values, so it introduces no lookahead.
    """
    out = frame.copy()
    price = out["adj_close"].astype(float)

    for col in ("sma_20", "sma_60", "ema_20"):
        if col in out.columns:
            out[col] = out[col] / price

    # MACD is a difference of two EMAs, also in dollars.
    for col in ("macd", "macd_signal", "macd_hist"):
        if col in out.columns:
            out[col] = out[col] / price

    # RSI is already bounded in [0, 100]; centre it near zero for the optimizer.
    if "rsi_14" in out.columns:
        out["rsi_14"] = (out["rsi_14"] - 50.0) / 50.0

    return out


def build_supervised(
    features: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Attach the next-day target and drop rows that cannot be used.

    The label is created with a ``shift(-1)`` *within each ticker*. This is the
    only forward-looking operation in the project, and it produces the target,
    never a feature. The final row of every ticker becomes NaN and is dropped:
    tomorrow has not happened yet.
    """
    feature_columns = feature_columns or MODEL_FEATURES
    if features.empty:
        return features

    out = features.sort_values(["ticker", "date"]).reset_index(drop=True)
    out = normalize_price_features(out)

    out[TARGET] = out.groupby("ticker")["log_return"].shift(-1)

    required = [*feature_columns, TARGET]
    before = len(out)
    out = out.dropna(subset=required).reset_index(drop=True)
    logger.info("supervised dataset: %d rows (dropped %d incomplete)", len(out), before - len(out))

    return out


def walk_forward_splits(
    dates: pd.Series | pd.DatetimeIndex,
    n_splits: int = 5,
    test_size: int = 63,
    min_train_size: int = 252,
    expanding: bool = True,
) -> list[Split]:
    """Generate expanding (or rolling) walk-forward folds over unique dates.

    Splitting on *dates* rather than rows matters for a multi-ticker panel: a
    row-based split would put AAPL's 3 March in training and MSFT's 3 March in
    test, which lets the model see the market regime it is being tested on.

    ``test_size`` defaults to 63 sessions — roughly a quarter, long enough for
    the MSE to mean something and short enough to give several folds.
    """
    unique = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).unique())).sort_values()
    n_dates = len(unique)

    needed = min_train_size + n_splits * test_size
    if n_dates < needed:
        max_splits = max(0, (n_dates - min_train_size) // test_size)
        if max_splits == 0:
            raise ValueError(
                f"need at least {min_train_size + test_size} sessions for one fold, got {n_dates}"
            )
        logger.warning(
            "only %d sessions available; reducing from %d to %d folds",
            n_dates,
            n_splits,
            max_splits,
        )
        n_splits = max_splits

    splits: list[Split] = []
    # Lay the test windows against the end of the series so the final fold tests
    # the most recent data — the period anyone reading the results cares about.
    first_test_start = n_dates - n_splits * test_size

    for i in range(n_splits):
        test_lo = first_test_start + i * test_size
        test_hi = test_lo + test_size
        train_lo = 0 if expanding else max(0, test_lo - min_train_size)

        splits.append(
            Split(
                index=i,
                train_start=unique[train_lo].date(),
                train_end=unique[test_lo - 1].date(),
                test_start=unique[test_lo].date(),
                test_end=unique[test_hi - 1].date(),
            )
        )

    return splits


def apply_split(frame: pd.DataFrame, split: Split) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice a panel into (train, test) for one fold, by date."""
    dates = pd.to_datetime(frame["date"])
    train_mask = (dates >= pd.Timestamp(split.train_start)) & (
        dates <= pd.Timestamp(split.train_end)
    )
    test_mask = (dates >= pd.Timestamp(split.test_start)) & (dates <= pd.Timestamp(split.test_end))
    return frame.loc[train_mask].copy(), frame.loc[test_mask].copy()


def fit_scaler(train: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of the *training fold only*.

    Fitting on the full panel is the most common silent leak in a walk-forward
    setup: it hands the model the test period's mean and variance before it has
    seen a single test row.
    """
    values = train[feature_columns].to_numpy(dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    # A constant feature has zero variance; dividing by it would produce inf.
    std[std < 1e-12] = 1.0
    return mean, std


def transform(
    frame: pd.DataFrame,
    feature_columns: list[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return (frame[feature_columns].to_numpy(dtype=np.float64) - mean) / std


def make_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    lookback: int = 20,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build (n, lookback, n_features) sequences for the LSTM.

    Sequences are built per ticker and never span a ticker boundary. Each
    sequence ends at the row whose target is being predicted, so window *t*
    covers sessions ``[t-lookback+1, t]`` and the label is the return on
    ``t+1`` — trailing input, forward label, no overlap.

    Returns the sequences, the targets, and the metadata rows (ticker, date)
    they correspond to, so predictions can be joined back to dates.
    """
    xs, ys, meta = [], [], []

    for ticker, group in frame.groupby("ticker", sort=True):
        group = group.sort_values("date")
        if len(group) <= lookback:
            logger.debug("%s: %d rows is too few for lookback %d", ticker, len(group), lookback)
            continue

        scaled = (group[feature_columns].to_numpy(dtype=np.float64) - mean) / std
        targets = group[TARGET].to_numpy(dtype=np.float64)

        for end in range(lookback - 1, len(group)):
            xs.append(scaled[end - lookback + 1 : end + 1])
            ys.append(targets[end])
            meta.append((ticker, group["date"].iloc[end]))

    if not xs:
        empty_meta = pd.DataFrame(columns=["ticker", "date"])
        n_features = len(feature_columns)
        return np.empty((0, lookback, n_features)), np.empty((0,)), empty_meta

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        pd.DataFrame(meta, columns=["ticker", "date"]),
    )
