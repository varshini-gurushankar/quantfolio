"""Liveness and readiness.

``/health`` reports database reachability and data freshness together: a service
that can serve requests but is backed by a feature store that stopped updating a
week ago is not actually healthy, and that distinction is what the freshness
field surfaces.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import func, select, text

from quantfolio.storage.db import get_engine
from quantfolio.storage.schema import features_daily

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    database: str
    latest_feature_date: date | None = None
    data_age_days: int | None = None
    checked_at: datetime


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    now = datetime.now(UTC)
    database = "unreachable"
    latest: date | None = None

    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            database = "ok"
            latest = conn.execute(select(func.max(features_daily.c.date))).scalar()
    except Exception as exc:  # noqa: BLE001 - health must report failure, not raise it
        logger.warning("health check could not reach the database: %s", exc)

    age = (now.date() - latest).days if latest else None
    return HealthResponse(
        status="ok" if database == "ok" else "degraded",
        database=database,
        latest_feature_date=latest,
        data_age_days=age,
        checked_at=now,
    )
