"""Secondary source: Alpha Vantage.

The free tier is 25 requests/day and 5/minute, and it signals throttling with a
200 response containing a ``Note``/``Information`` key rather than a 429. That
quirk is the reason this client inspects the payload before parsing it.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import requests

from quantfolio.ingestion.base import (
    PermanentSourceError,
    PriceSource,
    TransientSourceError,
    normalize_frame,
    with_backoff,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://www.alphavantage.co/query"
_SERIES_KEY = "Time Series (Daily)"
_FIELD_MAP = {
    "1. open": "open",
    "2. high": "high",
    "3. low": "low",
    "4. close": "close",
    "5. adjusted close": "adj_close",
    "6. volume": "volume",
}


class AlphaVantageSource(PriceSource):
    name = "alpha_vantage"

    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        if not api_key:
            raise PermanentSourceError("alpha_vantage: no API key configured")
        self.api_key = api_key
        self.timeout = timeout

    @with_backoff()
    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            # "full" reaches back 20+ years; "compact" is the last 100 bars.
            "outputsize": "full",
            "apikey": self.api_key,
        }
        try:
            resp = requests.get(BASE_URL, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise TransientSourceError(f"{ticker}: alpha vantage request failed: {exc}") from exc

        if resp.status_code >= 500 or resp.status_code == 429:
            raise TransientSourceError(f"{ticker}: alpha vantage HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise PermanentSourceError(f"{ticker}: alpha vantage HTTP {resp.status_code}")

        payload = resp.json()

        # Throttling and quota exhaustion both arrive as HTTP 200.
        if "Note" in payload or "Information" in payload:
            reason = payload.get("Note") or payload.get("Information")
            raise TransientSourceError(f"{ticker}: alpha vantage throttled: {reason}")
        if "Error Message" in payload:
            raise PermanentSourceError(
                f"{ticker}: alpha vantage rejected: {payload['Error Message']}"
            )
        if _SERIES_KEY not in payload:
            raise PermanentSourceError(
                f"{ticker}: unexpected alpha vantage payload keys {list(payload)}"
            )

        frame = (
            pd.DataFrame.from_dict(payload[_SERIES_KEY], orient="index")
            .rename(columns=_FIELD_MAP)
            .rename_axis("date")
            .reset_index()
        )
        frame["date"] = pd.to_datetime(frame["date"])
        mask = (frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))
        return normalize_frame(frame.loc[mask], ticker)
