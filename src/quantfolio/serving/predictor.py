"""Load a registered model and serve predictions from it.

Serving a model is more than loading weights. The three things that make this
work are all here:

* **The bundle is self-describing.** ``metadata.json`` travels with the weights
  and carries the feature ordering, the scaler statistics and the architecture
  hyperparameters. Reconstructing those from memory at deploy time is how
  serving skew happens.
* **Loading is cached.** Pulling from the registry and rebuilding a network
  takes seconds; doing it per request would dominate the latency this phase
  exists to measure.
* **Features are read, never recomputed.** The API serves the same rows the
  pipeline wrote, so a prediction cannot disagree with the feature store.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantfolio.config import get_settings
from quantfolio.models.datasets import MODEL_FEATURES, normalize_price_features
from quantfolio.models.train import Preprocessing

logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "quantfolio_return_predictor"

# Enough trailing history to fill the longest lookback with room for holidays.
FEATURE_LOOKBACK_DAYS = 120


class ModelNotAvailable(RuntimeError):
    """Raised when no usable model can be loaded from the registry."""


@dataclass
class LoadedModel:
    """A model plus everything needed to feed it."""

    name: str
    framework: str
    version: str
    preprocessing: Preprocessing
    params: dict[str, Any]
    predict_fn: Any
    metadata: dict[str, Any]

    @property
    def lookback(self) -> int:
        """Sequence length, or 1 for a model that sees a single day."""
        return int(self.params.get("lookback", 1) or 1)

    @property
    def is_sequential(self) -> bool:
        return self.lookback > 1


_cache: LoadedModel | None = None
_cache_lock = threading.Lock()


def _download_bundle(version: str | None = None) -> tuple[Path, str]:
    """Fetch the registered model's artifact folder from MLflow."""
    import mlflow
    from mlflow.artifacts import download_artifacts

    tracking_uri = get_settings().mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    try:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    except Exception as exc:  # noqa: BLE001 - any registry failure means "cannot serve"
        # An unreachable registry is a dependency outage, and callers above turn
        # ModelNotAvailable into a 503. Letting the raw MlflowException escape
        # would surface as a 500 and point at the wrong layer.
        raise ModelNotAvailable(f"model registry at {tracking_uri} unavailable: {exc}") from exc

    if not versions:
        raise ModelNotAvailable(
            f"no versions registered under '{REGISTERED_MODEL_NAME}' — run the training pipeline"
        )

    if version:
        chosen = next((v for v in versions if str(v.version) == str(version)), None)
        if chosen is None:
            raise ModelNotAvailable(f"version {version} not found")
    else:
        chosen = max(versions, key=lambda v: int(v.version))

    try:
        local = download_artifacts(artifact_uri=f"{chosen.source}")
    except Exception as exc:  # noqa: BLE001 - artifact store down is also "cannot serve"
        raise ModelNotAvailable(
            f"could not download artifacts for version {chosen.version}: {exc}"
        ) from exc

    return Path(local), str(chosen.version)


def _rebuild_keras(bundle: Path, metadata: dict):
    from tensorflow import keras

    model = keras.models.load_model(bundle / metadata["weights_file"])
    return lambda x: model.predict(x, verbose=0).ravel()


def _rebuild_torch(bundle: Path, metadata: dict):
    import torch

    from quantfolio.models.torch_lstm import TorchLSTM

    params = metadata["model_params"]
    shell = TorchLSTM(
        lookback=int(params["lookback"]),
        hidden_size=int(params["hidden_size"]),
        num_layers=int(params["num_layers"]),
        dropout=float(params["dropout"]),
    )
    n_features = len(metadata["preprocessing"]["feature_columns"])

    # The architecture is rebuilt from the logged hyperparameters, then the
    # saved weights are loaded into it — a state dict alone does not describe
    # the network it came from.
    net = shell._build(n_features)
    net.load_state_dict(torch.load(bundle / metadata["weights_file"], map_location="cpu"))
    net.eval()

    def predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return net(torch.from_numpy(x).float()).cpu().numpy().ravel()

    return predict


def load_model(version: str | None = None, force: bool = False) -> LoadedModel:
    """Load the registered model, caching it across requests."""
    global _cache

    with _cache_lock:
        if _cache is not None and not force and version is None:
            return _cache

        bundle, resolved_version = _download_bundle(version)
        metadata_path = bundle / "metadata.json"
        if not metadata_path.exists():
            raise ModelNotAvailable(
                "registered artifact has no metadata.json; it was logged by an older "
                "training run and cannot be served"
            )

        metadata = json.loads(metadata_path.read_text())
        framework = metadata["framework"]

        if framework == "tensorflow":
            predict_fn = _rebuild_keras(bundle, metadata)
        elif framework == "pytorch":
            predict_fn = _rebuild_torch(bundle, metadata)
        else:
            raise ModelNotAvailable(f"unsupported framework: {framework}")

        loaded = LoadedModel(
            name=metadata["model_name"],
            framework=framework,
            version=resolved_version,
            preprocessing=Preprocessing.from_dict(metadata["preprocessing"]),
            params=metadata.get("model_params", {}),
            predict_fn=predict_fn,
            metadata=metadata,
        )
        logger.info("loaded %s v%s (%s)", loaded.name, loaded.version, framework)

        if version is None:
            _cache = loaded
        return loaded


def clear_cache() -> None:
    """Drop the cached model so the next request reloads from the registry."""
    global _cache
    with _cache_lock:
        _cache = None


def _prepare_input(features: pd.DataFrame, model: LoadedModel) -> np.ndarray:
    """Scale the most recent rows into the shape the model expects."""
    prep = model.preprocessing
    mean = np.asarray(prep.feature_mean, dtype=np.float64)
    std = np.asarray(prep.feature_std, dtype=np.float64)

    normalized = normalize_price_features(features.sort_values("date"))
    usable = normalized.dropna(subset=prep.feature_columns)

    if usable.empty:
        raise ModelNotAvailable("no complete feature rows available for this ticker")

    if model.is_sequential:
        if len(usable) < model.lookback:
            raise ModelNotAvailable(
                f"{model.name} needs {model.lookback} sessions, only {len(usable)} are complete"
            )
        window = usable.tail(model.lookback)
        scaled = (window[prep.feature_columns].to_numpy(dtype=np.float64) - mean) / std
        return scaled[np.newaxis, :, :].astype(np.float32)

    latest = usable.tail(1)
    scaled = (latest[prep.feature_columns].to_numpy(dtype=np.float64) - mean) / std
    return scaled.astype(np.float32)


def predict_ticker(
    ticker: str,
    as_of: date | None = None,
    version: str | None = None,
) -> dict:
    """Predict the next session's log return for one ticker."""
    from quantfolio.storage.db import read_features

    model = load_model(version=version)
    as_of = as_of or date.today()
    start = as_of - timedelta(days=FEATURE_LOOKBACK_DAYS)

    features = read_features(tickers=[ticker], start=start, end=as_of)
    if features.empty:
        raise ModelNotAvailable(f"no features stored for {ticker}")

    x = _prepare_input(features, model)
    scaled_prediction = float(np.asarray(model.predict_fn(x)).ravel()[0])

    # Invert the target scaling so the answer is a return, not a z-score.
    prep = model.preprocessing
    predicted_log_return = scaled_prediction * prep.target_std + prep.target_mean

    feature_date = pd.to_datetime(features["date"]).max().date()

    return {
        "ticker": ticker,
        "as_of": feature_date,
        "predicted_log_return": predicted_log_return,
        "predicted_simple_return": float(np.expm1(predicted_log_return)),
        "model_name": model.name,
        "model_version": model.version,
        "framework": model.framework,
        "beats_baseline": bool(model.metadata.get("beats_baseline", False)),
    }


def latest_weights(engine=None) -> pd.DataFrame:
    """Most recent stored portfolio allocation."""
    from sqlalchemy import select

    from quantfolio.storage.db import get_engine
    from quantfolio.storage.schema import portfolio_weights

    engine = engine or get_engine()
    with engine.connect() as conn:
        latest_date = conn.execute(
            select(portfolio_weights.c.as_of_date)
            .order_by(portfolio_weights.c.as_of_date.desc())
            .limit(1)
        ).scalar()

        if latest_date is None:
            return pd.DataFrame(columns=["as_of_date", "ticker", "weight", "method"])

        stmt = (
            select(portfolio_weights)
            .where(portfolio_weights.c.as_of_date == latest_date)
            .order_by(portfolio_weights.c.weight.desc())
        )
        return pd.read_sql(stmt, conn)


def default_feature_columns() -> list[str]:
    return list(MODEL_FEATURES)
