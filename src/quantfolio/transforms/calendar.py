"""Exchange-calendar-aware gap handling.

Blind forward-filling over a date range invents bars for weekends, holidays and
half-days, which then feed rolling windows and quietly corrupt every downstream
statistic. Instead we ask the exchange calendar which sessions *should* exist,
reindex onto exactly those, and forward-fill only the sessions a source actually
missed — flagging each one so imputed rows can be excluded or inspected later.
"""

from __future__ import annotations

import functools
import logging
from datetime import date

import pandas as pd
import pandas_market_calendars as mcal

logger = logging.getLogger(__name__)

PRICE_FIELDS = ["open", "high", "low", "close", "adj_close", "volume"]


@functools.lru_cache(maxsize=8)
def _calendar(exchange: str):
    return mcal.get_calendar(exchange)


def trading_sessions(start: date, end: date, exchange: str = "NYSE") -> pd.DatetimeIndex:
    """Sessions the exchange was open for, inclusive of both endpoints."""
    sched = _calendar(exchange).schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(sched.index).normalize()


def align_to_sessions(
    frame: pd.DataFrame,
    start: date,
    end: date,
    exchange: str = "NYSE",
    max_gap: int = 5,
) -> pd.DataFrame:
    """Reindex one ticker's bars onto the exchange calendar and fill real gaps.

    * Rows on non-session dates are dropped (a source occasionally emits them).
    * Missing sessions are forward-filled from the previous session and marked
      ``is_imputed``. Forward-fill only ever reads the past, so it is causal.
    * Gaps longer than ``max_gap`` sessions are left as NaN: a two-week hole is
      a data problem for the quality check to fail on, not something to paper
      over with a stale price.
    """
    if frame.empty:
        return frame.assign(is_imputed=pd.Series(dtype=bool))

    ticker = frame["ticker"].iloc[0]
    sessions = trading_sessions(start, end, exchange)

    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"], keep="last").set_index("date").sort_index()

    off_session = df.index.difference(sessions)
    if len(off_session):
        logger.warning("%s: dropping %d rows on non-session dates", ticker, len(off_session))
        df = df.loc[df.index.intersection(sessions)]

    aligned = df.reindex(sessions)
    aligned["is_imputed"] = aligned["close"].isna()

    present = [c for c in PRICE_FIELDS if c in aligned.columns]
    aligned[present] = aligned[present].ffill(limit=max_gap)

    # Volume on an imputed session is zero, not the previous day's volume —
    # carrying it forward would fabricate traded shares.
    if "volume" in aligned.columns:
        aligned.loc[aligned["is_imputed"], "volume"] = 0

    still_missing = int(aligned["close"].isna().sum())
    if still_missing:
        logger.warning(
            "%s: %d sessions still missing after ffill (gap longer than %d sessions)",
            ticker,
            still_missing,
            max_gap,
        )

    aligned["ticker"] = ticker
    return aligned.rename_axis("date").reset_index()


def missing_sessions(
    frame: pd.DataFrame, start: date, end: date, exchange: str = "NYSE"
) -> pd.DatetimeIndex:
    """Sessions with no row at all — the input to the date-continuity quality check."""
    sessions = trading_sessions(start, end, exchange)
    have = pd.DatetimeIndex(pd.to_datetime(frame["date"]).unique()).normalize()
    return sessions.difference(have)
