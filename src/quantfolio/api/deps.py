"""Shared API dependencies."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd
from fastapi import HTTPException, Query
from sqlalchemy.engine import Engine

from quantfolio.storage.db import get_engine

MAX_RANGE_DAYS = 3650


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Frame -> JSON-safe row dicts.

    Warm-up rows carry NaN, which is a float in Python but not valid JSON, so
    every non-finite value becomes null before it reaches the response model.
    """
    rows = frame.to_dict(orient="records")
    for row in rows:
        for key, value in row.items():
            if value is None or value is pd.NaT or value is pd.NA:
                row[key] = None
            elif isinstance(value, float) and not math.isfinite(value):
                row[key] = None
            elif hasattr(value, "item") and not isinstance(value, (str, bytes)):
                row[key] = value.item()
    return rows


def engine() -> Engine:
    return get_engine()


def normalize_ticker(ticker: str) -> str:
    cleaned = ticker.strip().upper()
    if not cleaned.isalnum() and "." not in cleaned and "-" not in cleaned:
        raise HTTPException(status_code=400, detail=f"invalid ticker: {ticker!r}")
    return cleaned


def date_range(
    start: date | None = Query(None, description="Inclusive start date (YYYY-MM-DD)"),
    end: date | None = Query(None, description="Inclusive end date (YYYY-MM-DD)"),
) -> tuple[date | None, date | None]:
    if start and end:
        if start > end:
            raise HTTPException(status_code=400, detail="start must not be after end")
        if (end - start).days > MAX_RANGE_DAYS:
            raise HTTPException(
                status_code=400, detail=f"range exceeds {MAX_RANGE_DAYS} days; narrow the window"
            )
    return start, end
