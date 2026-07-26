"""Daily market data pipeline: fetch -> validate -> transform -> load -> quality_check.

The DAG is deliberately thin. All logic lives in ``quantfolio.pipeline`` so it
can be tested without a scheduler; this file only wires the tasks together and
declares the schedule.

Backfill: every task is keyed on the logical date and every write is either
write-once (raw) or an upsert (Postgres), so
``airflow dags backfill -s 2024-01-01 -e 2024-03-01 market_data_pipeline``
is safe to run repeatedly and in any order.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pendulum
from airflow.decorators import dag, task

from quantfolio.pipeline import (
    task_fetch,
    task_load,
    task_quality_check,
    task_transform,
    task_validate,
)

DEFAULT_ARGS = {
    "owner": "quantfolio",
    "retries": 3,
    # Sources rate-limit; backing off for minutes rather than seconds is the
    # difference between a retry that helps and three that burn the quota.
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "depends_on_past": False,
    "email_on_failure": False,
}


@dag(
    dag_id="market_data_pipeline",
    description="Point-in-time-correct daily equity prices and features",
    default_args=DEFAULT_ARGS,
    # 23:00 UTC — after the US close and after the vendors have settled the
    # day's adjusted closes.
    schedule="0 23 * * 1-5",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["quantfolio", "data", "phase1"],
    doc_md=__doc__,
)
def market_data_pipeline():
    @task(task_id="fetch")
    def fetch(logical_date: datetime | None = None, run_id: str | None = None, **_) -> dict:
        """Pull every ticker from the primary source, falling back to the secondary.

        ``logical_date`` and ``run_id`` are reserved context keys, which Airflow
        injects at run time and refuses to let a task default to anything but
        None — hence the fallbacks in the body rather than in the signature.
        """
        exec_date = (logical_date or datetime.now(UTC)).date()
        return task_fetch(exec_date, run_id=run_id or "manual")

    @task(task_id="validate")
    def validate(summary: dict) -> dict:
        """Schema and null gate on raw data, before anything derived is computed."""
        return task_validate(summary)

    @task(task_id="transform")
    def transform(summary: dict, _validated: dict) -> dict:
        """Calendar alignment, causal outlier clipping, trailing-window features."""
        return task_transform(summary)

    @task(task_id="load")
    def load(staged: dict) -> dict:
        """Upsert into Postgres on (ticker, date) — idempotent under retry."""
        return task_load(staged)

    @task(task_id="quality_check")
    def quality_check(loaded: dict) -> dict:
        """Row counts, date continuity, feature ranges. Fails the DAG on breach."""
        return task_quality_check(loaded)

    fetched = fetch()
    validated = validate(fetched)
    staged = transform(fetched, validated)
    loaded = load(staged)
    quality_check(loaded)


market_data_pipeline()
