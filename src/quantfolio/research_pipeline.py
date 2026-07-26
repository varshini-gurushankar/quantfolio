"""Task bodies for the Phase 2 DAGs (training, optimization, drift).

Same convention as ``pipeline.py``: the logic lives here so it can be tested and
run without a scheduler, and the DAG files stay thin.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import pandas as pd

from quantfolio.config import get_settings
from quantfolio.metrics.push import push_sharpe
from quantfolio.models.drift import (
    detect_drift,
    load_recent_predictions,
    load_reference_mse,
    store_predictions,
    store_reference_metrics,
)
from quantfolio.models.keras_mlp import KerasMLP
from quantfolio.models.torch_lstm import TorchLSTM
from quantfolio.models.train import compare, load_training_data, run_experiment
from quantfolio.portfolio.backtest import backtest, rolling_rebalance
from quantfolio.portfolio.optimizer import optimize, returns_matrix, store_weights
from quantfolio.storage import db

logger = logging.getLogger(__name__)

# Enough history for a meaningful walk-forward: five folds of a quarter each,
# on top of a year of initial training data.
DEFAULT_TRAINING_YEARS = 5

# The model the drift sensor watches. The challenger rather than the baseline,
# because it is the one that would be serving predictions.
DEFAULT_MONITORED_MODEL = "torch_lstm"


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)[:10]).date()


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def task_train_models(
    execution_date: str | date,
    years: int = DEFAULT_TRAINING_YEARS,
    n_splits: int = 5,
    test_size: int = 63,
    register_best: bool = True,
    epochs: int = 50,
) -> dict:
    """Train both frameworks on identical folds and register the better one.

    Both models are always trained, never just the incumbent: "the LSTM beats
    the Keras baseline" is only a claim if the baseline was retrained on the
    same data at the same time.
    """
    exec_date = _as_date(execution_date)
    start = exec_date - timedelta(days=365 * years)

    dataset = load_training_data(start=str(start), end=str(exec_date))

    models = [
        KerasMLP(epochs=epochs),  # the named baseline
        TorchLSTM(epochs=epochs),  # the sequential challenger
    ]

    results = []
    for model in models:
        logger.info("training %s (%s)", model.name, model.framework)
        result = run_experiment(
            dataset,
            model,
            n_splits=n_splits,
            test_size=test_size,
            register=False,  # registration happens once, for the winner
        )
        results.append(result)

        store_predictions(
            result.predictions()[["ticker", "date", "y_pred", "y_true"]],
            model_name=model.name,
            run_id=result.run_id,
        )
        store_reference_metrics(
            model_name=model.name,
            run_id=result.run_id or "unknown",
            mse=result.overall.mse,
            baseline_mse=result.overall.baseline_mse,
            window_end=exec_date,
        )

    table = compare(results)
    logger.info("model comparison:\n%s", table.to_string(index=False))

    best = min(results, key=lambda r: r.overall.mse)

    # Only promote a model that actually beats predicting nothing. Registering a
    # model worse than the zero baseline would be worse than having no model.
    if register_best and best.overall.beats_baseline:
        _register_best(best)
        registered = True
    else:
        registered = False
        if register_best:
            logger.warning(
                "best model (%s) does not beat the zero-prediction baseline; not registering",
                best.model_name,
            )

    return {
        "execution_date": exec_date.isoformat(),
        "best_model": best.model_name,
        "best_mse": best.overall.mse,
        "baseline_mse": best.overall.baseline_mse,
        "beats_baseline": best.overall.beats_baseline,
        "registered": registered,
        "comparison": table.to_dict(orient="records"),
    }


def _register_best(result) -> None:
    import mlflow

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    try:
        version = mlflow.register_model(
            model_uri=f"runs:/{result.run_id}/model",
            name="quantfolio_return_predictor",
        )
        client = mlflow.MlflowClient()
        for key, value in {
            "model_type": result.model_name,
            "framework": result.framework,
            "oos_mse": f"{result.overall.mse:.6e}",
            "baseline_mse": f"{result.overall.baseline_mse:.6e}",
            "improvement_over_baseline": f"{result.overall.improvement_over_baseline:.4f}",
        }.items():
            client.set_model_version_tag(version.name, version.version, key, value)
        logger.info("registered %s v%s (%s)", version.name, version.version, result.model_name)
    except Exception as exc:  # noqa: BLE001 - a registry outage must not fail training
        logger.warning("registration failed: %s", exc)


# --------------------------------------------------------------------------- #
# optimization + backtest
# --------------------------------------------------------------------------- #
def task_optimize_portfolio(
    execution_date: str | date,
    lookback_days: int = 504,
    max_weight: float = 0.30,
) -> dict:
    """Solve for target weights on trailing data and store them."""
    exec_date = _as_date(execution_date)
    start = exec_date - timedelta(days=lookback_days)

    features = db.read_features(start=start, end=exec_date)
    if features.empty:
        raise RuntimeError(f"no features available for {start}..{exec_date}")

    returns = returns_matrix(features)
    if returns.shape[1] < 2:
        raise RuntimeError(f"need at least 2 assets to optimize, got {returns.shape[1]}")

    result = optimize(returns, max_weight=max_weight, as_of=exec_date)
    rows = store_weights(result)

    return {
        "execution_date": exec_date.isoformat(),
        "n_assets": int(returns.shape[1]),
        "rows_written": rows,
        "volatility": result.volatility,
        "expected_return": result.expected_return,
        "status": result.status,
        "shrinkage": result.shrinkage,
    }


def task_backtest(
    execution_date: str | date,
    lookback_days: int = 1095,
    cost_bps: float = 10.0,
    rebalance_every: int = 21,
) -> dict:
    """Backtest the optimizer's allocations and publish Sharpe to Prometheus.

    Both gross and net Sharpe are reported. The claim this supports is that the
    system measures a strategy honestly, not that the strategy works.
    """
    exec_date = _as_date(execution_date)
    start = exec_date - timedelta(days=lookback_days)

    features = db.read_features(start=start, end=exec_date)
    if features.empty:
        raise RuntimeError(f"no features available for {start}..{exec_date}")

    log_returns = returns_matrix(features, value_column="log_return")
    # Backtest arithmetic compounds simple returns, not log returns.
    simple_returns = pd.DataFrame.expm1(log_returns)

    weights = rolling_rebalance(
        simple_returns,
        lookback=252,
        rebalance_every=rebalance_every,
        use_shrinkage=True,
    )
    result = backtest(weights, simple_returns, cost_bps=cost_bps)

    push_sharpe(
        gross=result.sharpe_gross,
        net=result.sharpe_net,
        volatility=result.volatility,
        as_of=exec_date.isoformat(),
    )

    return {"execution_date": exec_date.isoformat(), **result.as_dict()}


# --------------------------------------------------------------------------- #
# drift
# --------------------------------------------------------------------------- #
def task_check_drift(
    execution_date: str | date,
    model_name: str | None = None,
    window: int = 21,
    threshold: float = 1.5,
    min_consecutive_breaches: int = 3,
    lookback_days: int = 90,
) -> dict:
    """Compare recent OOS error against the model's training-time error.

    Returns a report; the DAG decides whether to trigger retraining. Keeping the
    decision and the action separate makes the sensor testable on its own.
    """
    exec_date = _as_date(execution_date)
    model_name = model_name or DEFAULT_MONITORED_MODEL

    reference = load_reference_mse(model_name)
    predictions = load_recent_predictions(model_name, lookback_days=lookback_days, as_of=exec_date)

    report = detect_drift(
        predictions,
        reference_mse=reference if reference is not None else float("nan"),
        model_name=model_name,
        window=window,
        threshold=threshold,
        min_consecutive_breaches=min_consecutive_breaches,
    )
    return report.as_dict()
