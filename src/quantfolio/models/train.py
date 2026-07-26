"""Walk-forward training harness with MLflow tracking.

One function runs any model through the same folds, so a Keras-vs-PyTorch
comparison differs in the model and nothing else. Each fold gets a fresh model
(`reset()`) and a scaler fit on that fold's training data alone.

MLflow structure: one parent run per model, one nested run per fold. Per-fold
MSE lives on the child runs; the aggregate lands on the parent, which is what
the registry promotes.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from quantfolio.config import get_settings
from quantfolio.models.base import Model
from quantfolio.models.datasets import (
    MODEL_FEATURES,
    Split,
    apply_split,
    build_supervised,
    fit_scaler,
    walk_forward_splits,
)
from quantfolio.models.evaluate import Metrics, aggregate, evaluate

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "quantfolio_return_prediction"


def _bundle_metadata(result: TrainingResult, weights_filename: str) -> dict:
    """The manifest that makes a saved model loadable by the serving layer."""
    return {
        "model_name": result.model_name,
        "framework": result.framework,
        "weights_file": weights_filename,
        "model_params": result.model_params,
        "preprocessing": result.preprocessing.as_dict() if result.preprocessing else None,
        "oos_mse": result.overall.mse,
        "baseline_mse": result.overall.baseline_mse,
        "beats_baseline": result.overall.beats_baseline,
        "trained_through": str(result.folds[-1].split.test_end) if result.folds else None,
    }


def _target_scaling(y_train: np.ndarray) -> tuple[float, float]:
    """Mean and standard deviation of the training fold's target, only.

    Using test-fold statistics here would leak the very thing being predicted.
    """
    mu = float(np.mean(y_train))
    sigma = float(np.std(y_train))
    return mu, sigma if sigma > 1e-12 else 1.0


@dataclass
class FoldResult:
    split: Split
    metrics: Metrics
    history: dict
    predictions: pd.DataFrame = field(repr=False)


@dataclass
class Preprocessing:
    """Everything needed to turn raw features into model input at serving time.

    Weights alone are not a servable model. Without the exact feature ordering
    and the scaler statistics the model was trained against, inference silently
    produces nonsense — so this travels with the artifact rather than being
    reconstructed from memory later.

    The values come from the *final* fold, which is the one trained on the most
    history and therefore the one worth deploying.
    """

    feature_columns: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    target_mean: float
    target_std: float

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> Preprocessing:
        return cls(**payload)


@dataclass
class TrainingResult:
    model_name: str
    framework: str
    folds: list[FoldResult]
    overall: Metrics
    run_id: str | None = None
    preprocessing: Preprocessing | None = None
    model_params: dict = field(default_factory=dict)

    def predictions(self) -> pd.DataFrame:
        """Out-of-sample predictions across every fold, in date order."""
        if not self.folds:
            return pd.DataFrame(columns=["ticker", "date", "y_true", "y_pred", "fold"])
        return pd.concat([f.predictions for f in self.folds], ignore_index=True).sort_values(
            ["date", "ticker"]
        )

    def summary(self) -> str:
        return f"{self.model_name} ({self.framework}): {self.overall.summary()}"


def walk_forward_train(
    dataset: pd.DataFrame,
    model: Model,
    feature_columns: list[str] | None = None,
    n_splits: int = 5,
    test_size: int = 63,
    min_train_size: int = 252,
    expanding: bool = True,
) -> TrainingResult:
    """Train and evaluate ``model`` across walk-forward folds.

    No MLflow involvement, so this is directly unit-testable; ``run_experiment``
    wraps it with tracking.
    """
    feature_columns = feature_columns or MODEL_FEATURES
    if dataset.empty:
        raise ValueError("cannot train on an empty dataset")

    splits = walk_forward_splits(
        dataset["date"],
        n_splits=n_splits,
        test_size=test_size,
        min_train_size=min_train_size,
        expanding=expanding,
    )

    results: list[FoldResult] = []

    for split in splits:
        train_frame, test_frame = apply_split(dataset, split)
        if train_frame.empty or test_frame.empty:
            logger.warning("skipping %s: empty train or test slice", split)
            continue

        # Fit the scaler on training data only — the classic silent leak.
        mean, std = fit_scaler(train_frame, feature_columns)

        model.reset()
        x_train, y_train, _ = model.prepare(train_frame, feature_columns, mean, std)
        x_test, y_test, meta = model.prepare(test_frame, feature_columns, mean, std)

        if len(x_train) == 0 or len(x_test) == 0:
            logger.warning("skipping %s: no usable samples after preparation", split)
            continue

        # Daily returns are order 1e-2, while a freshly initialised linear head
        # emits values of order 1 — so an unscaled target spends most of its
        # training budget shrinking the output rather than learning. Scaling the
        # target by the training fold's own statistics puts the loss surface in
        # a sane range for both frameworks. Predictions are inverted immediately
        # afterwards, so every reported metric is in raw return units.
        y_mu, y_sigma = _target_scaling(y_train)

        history = model.fit(x_train, (y_train - y_mu) / y_sigma)
        y_pred = model.predict(x_test) * y_sigma + y_mu
        metrics = evaluate(y_test, y_pred)

        logger.info("%s | %s", split, metrics.summary())

        predictions = meta.copy()
        predictions["y_true"] = y_test
        predictions["y_pred"] = y_pred
        predictions["fold"] = split.index

        results.append(
            FoldResult(split=split, metrics=metrics, history=history, predictions=predictions)
        )

        # Overwritten each fold, so this ends up describing the final one — the
        # fold with the most training history, and the model left in memory.
        preprocessing = Preprocessing(
            feature_columns=list(feature_columns),
            feature_mean=mean.tolist(),
            feature_std=std.tolist(),
            target_mean=y_mu,
            target_std=y_sigma,
        )

    if not results:
        raise RuntimeError("no fold produced a usable result")

    overall = aggregate([r.metrics for r in results])
    result = TrainingResult(
        model_name=model.name,
        framework=model.framework,
        folds=results,
        overall=overall,
        preprocessing=preprocessing,
        model_params=model.params(),
    )
    logger.info("OVERALL %s", result.summary())
    return result


def run_experiment(
    dataset: pd.DataFrame,
    model: Model,
    feature_columns: list[str] | None = None,
    experiment_name: str = EXPERIMENT_NAME,
    tracking_uri: str | None = None,
    register: bool = False,
    **split_kwargs,
) -> TrainingResult:
    """``walk_forward_train`` plus MLflow tracking and optional registration."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri or get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model.name) as parent:
        mlflow.log_params(model.params())
        mlflow.log_params({f"split_{k}": v for k, v in split_kwargs.items()})
        mlflow.set_tag("framework", model.framework)

        result = walk_forward_train(dataset, model, feature_columns=feature_columns, **split_kwargs)
        result.run_id = parent.info.run_id

        for fold in result.folds:
            with mlflow.start_run(run_name=f"{model.name}_fold{fold.split.index}", nested=True):
                mlflow.log_params(
                    {
                        "fold": fold.split.index,
                        "train_start": str(fold.split.train_start),
                        "train_end": str(fold.split.train_end),
                        "test_start": str(fold.split.test_start),
                        "test_end": str(fold.split.test_end),
                    }
                )
                mlflow.log_metrics(fold.metrics.as_dict())
                mlflow.log_metrics(
                    {k: v for k, v in fold.history.items() if isinstance(v, (int, float))}
                )

        # Aggregate metrics on the parent run: this is what model comparison and
        # the registry read.
        mlflow.log_metrics({f"overall_{k}": v for k, v in result.overall.as_dict().items()})
        mlflow.log_metric("beats_baseline", 1.0 if result.overall.beats_baseline else 0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            preds_path = Path(tmpdir) / "oos_predictions.parquet"
            result.predictions().to_parquet(preds_path, index=False)
            mlflow.log_artifact(str(preds_path))

            model_path = model.save(str(Path(tmpdir) / model.name))
            mlflow.log_artifact(model_path, artifact_path="model")

            # The weights are useless without the feature order and scaler that
            # produced them, so the metadata ships in the same artifact folder.
            meta_path = Path(tmpdir) / "metadata.json"
            meta_path.write_text(
                json.dumps(_bundle_metadata(result, Path(model_path).name), indent=2)
            )
            mlflow.log_artifact(str(meta_path), artifact_path="model")

        if register:
            _register(model.name, parent.info.run_id, result)

    return result


def _register(model_name: str, run_id: str, result: TrainingResult) -> None:
    """Register the trained model, tagged with the numbers that justify it."""
    import mlflow

    try:
        registered = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name="quantfolio_return_predictor",
        )
        client = mlflow.MlflowClient()
        client.set_model_version_tag(registered.name, registered.version, "model_type", model_name)
        client.set_model_version_tag(
            registered.name, registered.version, "oos_mse", f"{result.overall.mse:.6e}"
        )
        client.set_model_version_tag(
            registered.name,
            registered.version,
            "baseline_mse",
            f"{result.overall.baseline_mse:.6e}",
        )
        client.set_model_version_tag(
            registered.name,
            registered.version,
            "beats_baseline",
            str(result.overall.beats_baseline),
        )
        logger.info("registered %s version %s", registered.name, registered.version)
    except Exception as exc:  # noqa: BLE001 - registration is optional, training is not
        logger.warning("model registration failed: %s", exc)


def compare(results: list[TrainingResult]) -> pd.DataFrame:
    """Side-by-side comparison table — the honest version of the résumé claim.

    ``vs_baseline`` is the number that can be quoted, and ``vs_best_other`` is
    what "X% lower MSE than the Keras baseline" actually means.
    """
    rows = []
    for result in results:
        rows.append(
            {
                "model": result.model_name,
                "framework": result.framework,
                "oos_mse": result.overall.mse,
                "baseline_mse": result.overall.baseline_mse,
                "vs_baseline": result.overall.improvement_over_baseline,
                "directional_accuracy": result.overall.directional_accuracy,
                "information_coefficient": result.overall.information_coefficient,
                "beats_baseline": result.overall.beats_baseline,
                "n_samples": result.overall.n_samples,
            }
        )

    table = pd.DataFrame(rows).sort_values("oos_mse").reset_index(drop=True)

    if len(table) > 1:
        best_other = table["oos_mse"].iloc[1]
        table["vs_best_other"] = np.where(
            table.index == 0, (best_other - table["oos_mse"]) / best_other, np.nan
        )

    return table


def load_training_data(
    start: str | None = None,
    end: str | None = None,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Read features from Postgres and turn them into a supervised dataset."""
    from quantfolio.storage.db import read_features

    features = read_features(
        tickers=tickers,
        start=pd.Timestamp(start).date() if start else None,
        end=pd.Timestamp(end).date() if end else None,
    )
    if features.empty:
        raise RuntimeError("no features found — run the data pipeline first")

    dataset = build_supervised(features)
    logger.info(
        "training data: %d rows, %d tickers, %s..%s",
        len(dataset),
        dataset["ticker"].nunique(),
        dataset["date"].min(),
        dataset["date"].max(),
    )
    return dataset


def rolling_oos_mse(predictions: pd.DataFrame, window: int = 21) -> pd.Series:
    """Rolling out-of-sample MSE by date — the input to the drift sensor."""
    if predictions.empty:
        return pd.Series(dtype="float64")

    daily = (
        predictions.assign(sq_error=lambda d: (d["y_true"] - d["y_pred"]) ** 2)
        .groupby("date")["sq_error"]
        .mean()
        .sort_index()
    )
    return daily.rolling(window, min_periods=max(2, window // 2)).mean()
