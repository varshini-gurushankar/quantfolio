"""Model inference.

The response deliberately carries ``beats_baseline`` alongside the number. A
prediction from a model that loses to predicting zero is still a prediction, and
the caller is entitled to know which kind they are getting.
"""

from __future__ import annotations

import logging
import time
from datetime import date

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError

from quantfolio.api.deps import normalize_ticker
from quantfolio.api.metrics import MODEL_INFO, PREDICTION_LATENCY
from quantfolio.serving.predictor import ModelNotAvailable, load_model, predict_ticker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["predictions"])


class Prediction(BaseModel):
    ticker: str
    as_of: date
    predicted_log_return: float
    predicted_simple_return: float
    model_name: str
    model_version: str
    framework: str
    beats_baseline: bool
    inference_seconds: float

    model_config = {"protected_namespaces": ()}


class ModelStatus(BaseModel):
    loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    framework: str | None = None
    oos_mse: float | None = None
    baseline_mse: float | None = None
    beats_baseline: bool | None = None
    trained_through: str | None = None
    detail: str | None = None

    model_config = {"protected_namespaces": ()}


@router.get("/model/status", response_model=ModelStatus)
def model_status() -> ModelStatus:
    """What is currently servable — checked before blaming the endpoint."""
    try:
        model = load_model()
    except ModelNotAvailable as exc:
        return ModelStatus(loaded=False, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - status must report, not raise
        logger.warning("could not load model: %s", exc)
        return ModelStatus(loaded=False, detail=f"{type(exc).__name__}: {exc}")

    MODEL_INFO.labels(model.name, model.version, model.framework).set(1)

    return ModelStatus(
        loaded=True,
        model_name=model.name,
        model_version=model.version,
        framework=model.framework,
        oos_mse=model.metadata.get("oos_mse"),
        baseline_mse=model.metadata.get("baseline_mse"),
        beats_baseline=model.metadata.get("beats_baseline"),
        trained_through=model.metadata.get("trained_through"),
    )


@router.get("/{ticker}", response_model=Prediction)
def predict(
    ticker: str,
    as_of: date | None = Query(None, description="Predict as of this date (default: today)"),
    version: str | None = Query(None, description="Pin a registry version"),
) -> Prediction:
    """Predict the next session's return for one ticker."""
    symbol = normalize_ticker(ticker)

    start = time.perf_counter()
    try:
        result = predict_ticker(symbol, as_of=as_of, version=version)
    except ModelNotAvailable as exc:
        # 503, not 500: the service is fine, there is just nothing to serve yet.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        # Same reasoning for the feature store: a dependency being down is not a
        # bug in this service, and 500 would send you debugging the wrong layer.
        logger.warning("could not read features for %s: %s", symbol, exc)
        raise HTTPException(status_code=503, detail="feature store unavailable") from exc

    elapsed = time.perf_counter() - start
    PREDICTION_LATENCY.labels(result["model_name"]).observe(elapsed)

    return Prediction(**result, inference_seconds=elapsed)
