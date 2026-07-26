"""Engineered features.

Every value served here was computed from a trailing window, so a row dated
2020-03-16 contains only what was knowable on 2020-03-16. That is the property
that makes this endpoint safe to build a backtest on.
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from quantfolio.api.deps import date_range, normalize_ticker, records
from quantfolio.storage.db import read_features

router = APIRouter(prefix="/features", tags=["features"])


class FeatureRow(BaseModel):
    ticker: str
    date: date
    adj_close: float | None = None
    simple_return: float | None = None
    log_return: float | None = None
    sma_20: float | None = None
    sma_60: float | None = None
    ema_20: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    rsi_14: float | None = None
    volatility_20: float | None = None
    is_imputed: bool | None = None


class FeatureResponse(BaseModel):
    ticker: str
    count: int
    start: date | None
    end: date | None
    rows: list[FeatureRow]


@router.get("/{ticker}", response_model=FeatureResponse)
def get_features(
    ticker: str,
    window: tuple[date | None, date | None] = Depends(date_range),
    limit: int = Query(1000, ge=1, le=10_000, description="Most recent N rows"),
) -> FeatureResponse:
    symbol = normalize_ticker(ticker)
    start, end = window

    frame = read_features(tickers=[symbol], start=start, end=end)
    if frame.empty:
        raise HTTPException(status_code=404, detail=f"no features for {symbol}")

    frame = frame.sort_values("date").tail(limit)
    rows = [FeatureRow(**row) for row in records(frame)]

    return FeatureResponse(
        ticker=symbol,
        count=len(rows),
        start=rows[0].date if rows else None,
        end=rows[-1].date if rows else None,
        rows=rows,
    )
