"""Daily price history."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from quantfolio.api.deps import date_range, normalize_ticker, records
from quantfolio.storage.db import read_prices

router = APIRouter(prefix="/prices", tags=["prices"])


class PriceBar(BaseModel):
    ticker: str
    date: date
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    adj_close: float | None = None
    volume: int | None = None
    source: str | None = None


class PriceResponse(BaseModel):
    ticker: str
    count: int
    start: date | None
    end: date | None
    bars: list[PriceBar]


@router.get("/{ticker}", response_model=PriceResponse)
def get_prices(
    ticker: str,
    window: tuple[date | None, date | None] = Depends(date_range),
    limit: int = Query(1000, ge=1, le=10_000, description="Most recent N bars"),
) -> PriceResponse:
    symbol = normalize_ticker(ticker)
    start, end = window

    frame = read_prices(tickers=[symbol], start=start, end=end)
    if frame.empty:
        raise HTTPException(status_code=404, detail=f"no price data for {symbol}")

    frame = frame.sort_values("date").tail(limit)
    bars = [PriceBar(**row) for row in records(frame)]

    return PriceResponse(
        ticker=symbol,
        count=len(bars),
        start=bars[0].date if bars else None,
        end=bars[-1].date if bars else None,
        bars=bars,
    )
