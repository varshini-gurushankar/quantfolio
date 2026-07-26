"""Primary source: yfinance. No API key, generous limits."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from quantfolio.ingestion.base import (
    PermanentSourceError,
    PriceSource,
    TransientSourceError,
    normalize_frame,
    with_backoff,
)

logger = logging.getLogger(__name__)


class YFinanceSource(PriceSource):
    name = "yfinance"

    @with_backoff()
    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        try:
            # yfinance treats `end` as exclusive; the caller's range is inclusive.
            raw = yf.download(
                ticker,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:  # network/HTTP problems surface as generic exceptions
            raise TransientSourceError(f"{ticker}: yfinance download failed: {exc}") from exc

        if raw is None or raw.empty:
            # An empty window is legitimate (holidays, halted ticker); an empty
            # response for a long window is not, but we cannot tell them apart
            # here, so return empty and let the quality check decide.
            logger.info("yfinance returned no rows for %s over %s..%s", ticker, start, end)
            return normalize_frame(_empty_frame(), ticker)

        # A single-ticker download still returns a MultiIndex on recent versions.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        if "adj_close" not in raw.columns:
            raise PermanentSourceError(f"{ticker}: yfinance response has no adjusted close")

        return normalize_frame(raw, ticker)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "adj_close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
        }
    )
