"""FastAPI service over the feature store, models and portfolio."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from quantfolio.api import metrics
from quantfolio.api.routers import features, health, portfolio, predict, prices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(
    title="QuantFolio API",
    version="0.3.0",
    description=(
        "Point-in-time-correct prices and features, model predictions from the "
        "MLflow registry, and the current optimized portfolio."
    ),
)

# Middleware first, so every route below is instrumented.
metrics.install(app)

app.include_router(health.router)
app.include_router(prices.router)
app.include_router(features.router)
app.include_router(predict.router)
app.include_router(portfolio.router)
