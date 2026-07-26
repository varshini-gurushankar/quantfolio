"""Metrics for ephemeral batch jobs.

Prometheus scrapes; Airflow tasks exit. A task that ran for forty seconds is
gone long before the next scrape interval, so there is nothing to pull from.
The Pushgateway exists for exactly this case: the job pushes its metrics on
completion and Prometheus scrapes the gateway instead.

Pushes are best-effort. A monitoring outage must never fail a data pipeline, so
every failure here is logged and swallowed.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import UTC, datetime

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from quantfolio.config import get_settings

logger = logging.getLogger(__name__)

JOB_NAME = "quantfolio_pipeline"


def _push(registry: CollectorRegistry, job: str, grouping: dict[str, str]) -> None:
    url = get_settings().pushgateway_url
    try:
        push_to_gateway(url, job=job, registry=registry, grouping_key=grouping)
        logger.info("pushed metrics to %s (job=%s, %s)", url, job, grouping)
    except Exception as exc:  # noqa: BLE001 - monitoring must never break the pipeline
        logger.warning("failed to push metrics to %s: %s", url, exc)


def push_ingestion_metrics(
    task: str,
    rows: int,
    duration_seconds: float,
    tickers_ok: int,
    tickers_failed: int,
    execution_date: str,
) -> None:
    registry = CollectorRegistry()

    Gauge("quantfolio_rows_ingested", "Rows ingested in this run", registry=registry).set(rows)
    Gauge("quantfolio_task_duration_seconds", "Task wall-clock duration", registry=registry).set(
        duration_seconds
    )
    Gauge("quantfolio_tickers_succeeded", "Tickers fetched successfully", registry=registry).set(
        tickers_ok
    )
    Gauge("quantfolio_tickers_failed", "Tickers that failed all sources", registry=registry).set(
        tickers_failed
    )
    Gauge(
        "quantfolio_last_run_timestamp", "Unix time of the last completed run", registry=registry
    ).set(time.time())

    _push(registry, JOB_NAME, {"task": task, "execution_date": execution_date})


def push_freshness(newest_row_date: datetime | None, task: str = "quality_check") -> None:
    """Age of the newest row in the feature store, in seconds.

    Freshness is the metric that catches a pipeline which is "green" but has
    silently stopped producing new data.
    """
    registry = CollectorRegistry()
    age_seconds = (
        (datetime.now(UTC) - newest_row_date).total_seconds()
        if newest_row_date is not None
        else -1.0
    )
    Gauge(
        "quantfolio_data_age_seconds",
        "Age of the newest row in the feature store",
        registry=registry,
    ).set(age_seconds)
    _push(registry, JOB_NAME, {"task": task})


def push_quality_result(passed: bool, n_failures: int, execution_date: str) -> None:
    registry = CollectorRegistry()
    Gauge("quantfolio_quality_passed", "1 if all blocking checks passed", registry=registry).set(
        1 if passed else 0
    )
    Gauge("quantfolio_quality_failures", "Number of failed checks", registry=registry).set(
        n_failures
    )
    _push(registry, JOB_NAME, {"task": "quality_check", "execution_date": execution_date})


def push_sharpe(gross: float, net: float, volatility: float, as_of: str) -> None:
    """Phase 3: strategy metrics, pushed after each backtest run."""
    registry = CollectorRegistry()
    Gauge("quantfolio_sharpe_gross", "Gross Sharpe ratio", registry=registry).set(gross)
    Gauge("quantfolio_sharpe_net", "Sharpe ratio net of transaction costs", registry=registry).set(
        net
    )
    Gauge(
        "quantfolio_portfolio_volatility", "Annualized portfolio volatility", registry=registry
    ).set(volatility)
    _push(registry, "quantfolio_strategy", {"as_of": as_of})


def push_allocation(
    drift: float,
    concentration: float,
    max_weight: float,
    n_holdings: int,
    as_of: str,
) -> None:
    """Portfolio composition metrics, pushed after each optimization.

    ``drift`` is one-way turnover against the previous allocation. It is the
    number that says whether the optimizer is tracking a genuinely changing
    covariance or just chasing noise — a min-variance portfolio that reshuffles
    30% of the book every month is paying costs for estimation error.
    """
    registry = CollectorRegistry()
    Gauge(
        "quantfolio_allocation_drift",
        "One-way turnover versus the previous allocation",
        registry=registry,
    ).set(drift)
    Gauge(
        "quantfolio_allocation_concentration",
        "Herfindahl index of the weights (1/N when equally weighted)",
        registry=registry,
    ).set(concentration)
    Gauge("quantfolio_allocation_max_weight", "Largest single position", registry=registry).set(
        max_weight
    )
    Gauge("quantfolio_allocation_holdings", "Number of positions held", registry=registry).set(
        n_holdings
    )
    _push(registry, "quantfolio_strategy", {"as_of": as_of})


@contextmanager
def timed(label: str):
    """Time a block and hand the elapsed seconds back to the caller."""
    elapsed = {}
    start = time.perf_counter()
    try:
        yield elapsed
    finally:
        elapsed["seconds"] = time.perf_counter() - start
        logger.info("%s took %.2fs", label, elapsed["seconds"])
