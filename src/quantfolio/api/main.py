"""FastAPI service over the feature store."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from quantfolio.api.routers import features, health, prices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(
    title="QuantFolio API",
    version="0.1.0",
    description="Read access to point-in-time-correct prices and engineered features.",
)

app.include_router(health.router)
app.include_router(prices.router)
app.include_router(features.router)
