"""Shared contract for price sources.

Every source returns the same canonical frame, so downstream transforms never
branch on provenance. Sources are responsible for their own retry/backoff; the
caller is responsible for tolerating a source that gives up.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import pandas as pd
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Canonical schema for every source. Order matters: raw parquet is written with
# these columns so a replay from S3 needs no per-source knowledge.
PRICE_COLUMNS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]


class TransientSourceError(RuntimeError):
    """Rate limit, timeout, or 5xx — worth retrying."""


class PermanentSourceError(RuntimeError):
    """Bad ticker, bad key, or malformed response — retrying will not help."""


@dataclass
class FetchResult:
    """Outcome of one (source, ticker) fetch. Failures are data, not exceptions.

    The DAG's fetch task collects these and continues; a partial failure marks
    the ticker in ``ingestion_log`` and leaves every other ticker's state intact.
    """

    ticker: str
    source: str
    frame: pd.DataFrame | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.frame is not None and not self.frame.empty

    @property
    def row_count(self) -> int:
        return 0 if self.frame is None else len(self.frame)


def _log_retry(state: RetryCallState) -> None:
    logger.warning(
        "retrying %s (attempt %d) after %s",
        state.fn.__qualname__ if state.fn else "fetch",
        state.attempt_number,
        state.outcome.exception() if state.outcome else "unknown error",
    )


def with_backoff(max_attempts: int = 5, initial_wait: float = 2.0, max_wait: float = 60.0):
    """Exponential backoff for transient failures only.

    Alpha Vantage's free tier allows 5 requests/minute, so the ceiling is set
    above a minute of waiting rather than the usual few seconds.
    """
    return retry(
        retry=retry_if_exception_type(TransientSourceError),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=initial_wait, max=max_wait),
        before_sleep=_log_retry,
        reraise=True,
    )


def normalize_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Coerce a source frame into the canonical schema, sorted and deduplicated."""
    out = df.copy()
    out["ticker"] = ticker
    missing = [c for c in PRICE_COLUMNS if c not in out.columns]
    if missing:
        raise PermanentSourceError(f"{ticker}: source frame missing columns {missing}")

    out = out[PRICE_COLUMNS]
    out["date"] = pd.to_datetime(out["date"]).dt.date
    for col in ("open", "high", "low", "close", "adj_close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").astype("Int64")

    # A duplicated date from a source is always a source bug; keep the last.
    out = out.drop_duplicates(subset=["ticker", "date"], keep="last")
    return out.sort_values("date").reset_index(drop=True)


class PriceSource(ABC):
    """A daily OHLCV source."""

    name: str

    @abstractmethod
    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Return canonical daily bars for ``ticker`` over ``[start, end]`` inclusive."""

    def fetch_safe(self, ticker: str, start: date, end: date) -> FetchResult:
        """``fetch`` that converts any failure into a ``FetchResult``.

        This is what the DAG calls: one bad ticker must not abort the run or
        leave the other tickers half-written.
        """
        try:
            frame = self.fetch(ticker, start, end)
            return FetchResult(ticker=ticker, source=self.name, frame=frame)
        except Exception as exc:  # noqa: BLE001 - deliberate: failures are recorded, not raised
            logger.exception("fetch failed for %s from %s", ticker, self.name)
            return FetchResult(ticker=ticker, source=self.name, frame=None, error=str(exc))
