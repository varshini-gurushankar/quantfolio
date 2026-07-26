"""Current portfolio allocation."""

from __future__ import annotations

import logging
from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError

from quantfolio.api.deps import records
from quantfolio.serving.predictor import latest_weights

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/portfolio", tags=["portfolio"])


class Holding(BaseModel):
    ticker: str
    weight: float


class Portfolio(BaseModel):
    as_of_date: date
    method: str
    n_holdings: int
    total_weight: float
    concentration: float
    holdings: list[Holding]


@router.get("", response_model=Portfolio)
def get_portfolio() -> Portfolio:
    """The most recent weights the optimizer produced."""
    try:
        frame = latest_weights()
    except SQLAlchemyError as exc:
        # The feature store being unreachable is a dependency failure, not a bug
        # in this service — 503 points at the right layer, 500 does not.
        logger.warning("could not read portfolio weights: %s", exc)
        raise HTTPException(status_code=503, detail="feature store unavailable") from exc

    if frame.empty:
        raise HTTPException(
            status_code=503,
            detail="no portfolio weights stored — run the training pipeline",
        )

    rows = records(frame)
    holdings = [Holding(ticker=r["ticker"], weight=r["weight"]) for r in rows]
    weights = [h.weight for h in holdings]

    return Portfolio(
        as_of_date=rows[0]["as_of_date"],
        method=rows[0].get("method") or "unknown",
        n_holdings=len(holdings),
        total_weight=sum(weights),
        # Herfindahl index: 1/N for an equal-weight book, 1.0 for a single
        # position. A quick read on whether the optimizer concentrated.
        concentration=sum(w**2 for w in weights),
        holdings=holdings,
    )
