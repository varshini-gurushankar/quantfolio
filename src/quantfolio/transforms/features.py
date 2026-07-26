"""Feature engineering — trailing windows only.

Every feature here is computed with ``.rolling()``, ``.ewm()`` or ``.shift(k)``
for positive *k*. No function in this module may call ``.mean()``, ``.std()``,
``.median()``, ``.min()`` or ``.max()`` over a full series, and none may use
``center=True`` or a negative shift. Those are the operations that make a
feature at time *t* depend on data after *t*.

``tests/test_no_lookahead.py`` proves the property empirically: it perturbs a
future observation and asserts that no earlier feature value moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantfolio.config import FeatureParams

FEATURE_COLUMNS = [
    "simple_return",
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

TRADING_DAYS_PER_YEAR = 252


def simple_returns(price: pd.Series) -> pd.Series:
    """(P_t / P_{t-1}) - 1. Uses only the previous observation."""
    return price.pct_change()


def log_returns(price: pd.Series) -> pd.Series:
    """log(P_t / P_{t-1}) — the modelling target, additive across time."""
    return np.log(price / price.shift(1))


def sma(price: pd.Series, window: int) -> pd.Series:
    return price.rolling(window, min_periods=window).mean()


def ema(price: pd.Series, span: int) -> pd.Series:
    # adjust=False gives the recursive form: today's value depends only on
    # today's price and yesterday's EMA.
    return price.ewm(span=span, adjust=False, min_periods=span).mean()


def macd(
    price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    ema_fast = price.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = price.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def rsi(price: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI, bounded in [0, 100].

    Wilder's smoothing is an EWM with alpha = 1/window, so each value depends
    only on the current change and the previous average.
    """
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))

    warm = avg_gain.notna() & avg_loss.notna()
    # No losses in the window -> rs is infinite -> RSI 100. A perfectly flat
    # window leaves 0/0, which is undefined rather than extreme: call it 50.
    out = out.mask(warm & (avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask(warm & (avg_gain == 0) & (avg_loss == 0), 50.0)
    return out


def rolling_volatility(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """Trailing standard deviation of returns, annualized by default."""
    vol = returns.rolling(window, min_periods=window).std()
    return vol * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else vol


def compute_features(
    frame: pd.DataFrame,
    params: FeatureParams | None = None,
    price_column: str = "adj_close",
) -> pd.DataFrame:
    """Compute the full feature set for one ticker's price history.

    Expects a single ticker sorted ascending by date; returns one row per input
    row with warm-up periods left as NaN rather than back-filled.
    """
    params = params or FeatureParams()
    if frame.empty:
        return frame.assign(**{c: pd.Series(dtype="float64") for c in FEATURE_COLUMNS})

    out = frame.sort_values("date").reset_index(drop=True).copy()
    price = out[price_column].astype(float)

    out["simple_return"] = simple_returns(price)
    out["log_return"] = log_returns(price)

    for window in params.sma_windows:
        out[f"sma_{window}"] = sma(price, window)

    out[f"ema_{params.ema_span}"] = ema(price, params.ema_span)

    line, sig, hist = macd(price, params.macd.fast, params.macd.slow, params.macd.signal)
    out["macd"], out["macd_signal"], out["macd_hist"] = line, sig, hist

    out[f"rsi_{params.rsi_window}"] = rsi(price, params.rsi_window)
    out[f"volatility_{params.volatility_window}"] = rolling_volatility(
        out["log_return"], params.volatility_window
    )

    return out


def compute_features_by_ticker(
    frame: pd.DataFrame,
    params: FeatureParams | None = None,
    price_column: str = "adj_close",
) -> pd.DataFrame:
    """Apply ``compute_features`` per ticker.

    Grouping matters: a single rolling window across a concatenated multi-ticker
    frame would blend one symbol's history into another's.
    """
    if frame.empty:
        return frame

    parts = [
        compute_features(group, params=params, price_column=price_column)
        for _, group in frame.groupby("ticker", sort=True)
    ]
    return pd.concat(parts, ignore_index=True)


def next_day_log_return(frame: pd.DataFrame, return_column: str = "log_return") -> pd.Series:
    """The supervised target: tomorrow's log return, aligned to today's features.

    This is the one intentional forward shift in the project. It creates the
    label, never a feature, and the last row is NaN by construction — if it were
    not, the target would be knowable today.
    """
    return frame.groupby("ticker")[return_column].shift(-1)
