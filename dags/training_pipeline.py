"""Model training and portfolio construction.

train_models -> optimize_portfolio -> backtest

Runs weekly on a schedule, and is also the DAG the drift sensor triggers when
recent out-of-sample error deteriorates.

Both frameworks are retrained on every run. Comparing a freshly trained
challenger against a stale baseline would flatter the challenger for reasons
that have nothing to do with the model.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pendulum
from airflow.decorators import dag, task

from quantfolio.research_pipeline import (
    task_backtest,
    task_optimize_portfolio,
    task_train_models,
)

DEFAULT_ARGS = {
    "owner": "quantfolio",
    "retries": 1,
    # Training is expensive; a tight retry loop on a genuine failure just burns
    # an hour of CPU to fail the same way.
    "retry_delay": timedelta(minutes=15),
    "depends_on_past": False,
    "email_on_failure": False,
}


@dag(
    dag_id="training_pipeline",
    description="Walk-forward training, Markowitz optimization, cost-adjusted backtest",
    default_args=DEFAULT_ARGS,
    schedule="0 2 * * 6",  # Saturday 02:00 UTC, after the week's data has landed
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["quantfolio", "ml", "phase2"],
    doc_md=__doc__,
)
def training_pipeline():
    @task(task_id="train_models", execution_timeout=timedelta(hours=2))
    def train(logical_date: datetime | None = None, **_) -> dict:
        """Train Keras baseline and PyTorch LSTM on identical walk-forward folds."""
        exec_date = (logical_date or datetime.now(UTC)).date()
        return task_train_models(exec_date)

    @task(task_id="optimize_portfolio")
    def optimize(_trained: dict, logical_date: datetime | None = None, **_) -> dict:
        """Minimum-variance weights with Ledoit-Wolf shrinkage, stored to Postgres."""
        exec_date = (logical_date or datetime.now(UTC)).date()
        return task_optimize_portfolio(exec_date)

    @task(task_id="backtest")
    def run_backtest(_weights: dict, logical_date: datetime | None = None, **_) -> dict:
        """Cost-adjusted backtest; pushes gross and net Sharpe to Prometheus."""
        exec_date = (logical_date or datetime.now(UTC)).date()
        return task_backtest(exec_date)

    trained = train()
    weights = optimize(trained)
    run_backtest(weights)


training_pipeline()
