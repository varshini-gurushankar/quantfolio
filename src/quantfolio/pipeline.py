"""Task bodies for ``market_data_pipeline``.

These live in the package rather than in the DAG file so they can be unit-tested
and run from a script without an Airflow scheduler. The DAG is a thin wrapper.

Two properties hold for every function here:

* **Idempotent** — re-running for the same execution date produces the same end
  state. Raw partitions are write-once, staged parquet is overwritten wholesale,
  and Postgres writes are upserts on ``(ticker, date)``.
* **Self-contained** — each run fetches a full warm-up window rather than
  depending on what a previous run happened to leave behind, so any execution
  date can be re-run in isolation and a backfill needs no particular ordering.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from quantfolio.config import get_settings, get_universe
from quantfolio.ingestion.runner import fetch_universe
from quantfolio.metrics.push import (
    push_freshness,
    push_ingestion_metrics,
    push_quality_result,
)
from quantfolio.quality.checks import QualityReport, validate_features, validate_raw_prices
from quantfolio.storage import db
from quantfolio.storage.s3 import raw_key, s3_client, write_raw
from quantfolio.storage.schema import features_daily, ingestion_log, prices_daily
from quantfolio.transforms.calendar import align_to_sessions
from quantfolio.transforms.cleaning import clean_prices
from quantfolio.transforms.features import compute_features_by_ticker

logger = logging.getLogger(__name__)

# Calendar days of history fetched on every run. Long enough to warm up the
# 60-session SMA and the 63-session MAD window with room for holidays, so the
# newest rows are never computed from a partially filled window.
WARMUP_DAYS = 400

_STAGING_PREFIX = "staging"


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)[:10]).date()


def _staging_key(execution_date: date, name: str) -> str:
    return f"{_STAGING_PREFIX}/{execution_date.isoformat()}/{name}.parquet"


def _write_staging(frame: pd.DataFrame, execution_date: date, name: str) -> str:
    """Staged data is derived, so unlike raw it is overwritten on every re-run."""
    settings = get_settings()
    client = s3_client(settings)
    key = _staging_key(execution_date, name)
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
    client.put_object(Bucket=settings.s3_artifacts_bucket, Key=key, Body=buf.getvalue())
    logger.info("staged %d rows at s3://%s/%s", len(frame), settings.s3_artifacts_bucket, key)
    return key


def _read_staging(execution_date: date, name: str) -> pd.DataFrame:
    settings = get_settings()
    client = s3_client(settings)
    obj = client.get_object(
        Bucket=settings.s3_artifacts_bucket, Key=_staging_key(execution_date, name)
    )
    return pd.read_parquet(io.BytesIO(obj["Body"].read()), engine="pyarrow")


def _read_raw_partitions(execution_date: date, sources_by_ticker: dict[str, str]) -> pd.DataFrame:
    """Reassemble the run's raw partitions — the replay path, used every run."""
    settings = get_settings()
    client = s3_client(settings)
    frames = []
    for ticker, source in sources_by_ticker.items():
        obj = client.get_object(
            Bucket=settings.s3_raw_bucket, Key=raw_key(source, ticker, execution_date)
        )
        frames.append(pd.read_parquet(io.BytesIO(obj["Body"].read()), engine="pyarrow"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@dataclass
class FetchSummary:
    """Small, XCom-friendly result. Frames go to S3, never through XCom."""

    execution_date: str
    sources_by_ticker: dict[str, str]
    rows: int
    succeeded: list[str]
    failed: list[str]


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #
def task_fetch(
    execution_date: str | date, run_id: str = "manual", warmup_days: int = WARMUP_DAYS
) -> dict:
    """Fetch every ticker and write immutable raw partitions to S3.

    Per-ticker failures are recorded in ``ingestion_log`` and do not abort the
    run: a single delisted or rate-limited symbol must not cost the other
    thirteen their data.
    """
    exec_date = _as_date(execution_date)
    universe = get_universe()
    start = exec_date - timedelta(days=warmup_days)

    started = datetime.now(UTC)
    results = fetch_universe(list(universe.tickers), start, exec_date)
    duration = (datetime.now(UTC) - started).total_seconds()

    log_rows, sources_by_ticker, succeeded, failed = [], {}, [], []
    total_rows = 0

    for result in results:
        s3_key = None
        if result.ok:
            s3_key = write_raw(result.frame, result.source, result.ticker, exec_date)
            sources_by_ticker[result.ticker] = result.source
            succeeded.append(result.ticker)
            total_rows += result.row_count
            status = "success"
        else:
            failed.append(result.ticker)
            status = "empty" if result.error is None else "failed"

        log_rows.append(
            {
                "run_id": run_id,
                "ticker": result.ticker,
                "source": result.source,
                "execution_date": exec_date,
                "row_count": result.row_count,
                "status": status,
                "error": result.error,
                "s3_key": s3_key,
            }
        )

    db.append(pd.DataFrame(log_rows), ingestion_log)
    push_ingestion_metrics(
        task="fetch",
        rows=total_rows,
        duration_seconds=duration,
        tickers_ok=len(succeeded),
        tickers_failed=len(failed),
        execution_date=exec_date.isoformat(),
    )

    if not succeeded:
        raise RuntimeError(f"fetch produced no data for any ticker on {exec_date}: {failed}")

    logger.info("fetched %d rows for %d/%d tickers", total_rows, len(succeeded), len(results))
    return FetchSummary(
        execution_date=exec_date.isoformat(),
        sources_by_ticker=sources_by_ticker,
        rows=total_rows,
        succeeded=succeeded,
        failed=failed,
    ).__dict__


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #
def task_validate(summary: dict) -> dict:
    """Schema and null checks on raw, before anything derived is computed."""
    exec_date = _as_date(summary["execution_date"])
    raw = _read_raw_partitions(exec_date, summary["sources_by_ticker"])

    report = validate_raw_prices(raw, min_rows=1)
    report.raise_if_failed()

    logger.info("raw validation passed: %d rows, %d tickers", len(raw), raw["ticker"].nunique())
    return {"execution_date": exec_date.isoformat(), "rows": len(raw)}


# --------------------------------------------------------------------------- #
# transform
# --------------------------------------------------------------------------- #
def task_transform(summary: dict, warmup_days: int = WARMUP_DAYS) -> dict:
    """Align to the exchange calendar, clip outliers causally, compute features."""
    exec_date = _as_date(summary["execution_date"])
    universe = get_universe()
    start = exec_date - timedelta(days=warmup_days)

    raw = _read_raw_partitions(exec_date, summary["sources_by_ticker"])
    if raw.empty:
        raise RuntimeError(f"no raw data to transform for {exec_date}")

    cleaned_parts = []
    for _ticker, group in raw.groupby("ticker", sort=True):
        aligned = align_to_sessions(group, start, exec_date, universe.exchange)
        cleaned = clean_prices(
            aligned,
            window=universe.cleaning.mad_window,
            scale=universe.cleaning.mad_scale,
        )
        cleaned_parts.append(cleaned)

    prices = pd.concat(cleaned_parts, ignore_index=True)
    prices["source"] = prices["ticker"].map(summary["sources_by_ticker"]).fillna("unknown")

    features = compute_features_by_ticker(prices, params=universe.features)
    features["is_imputed"] = features.get("is_imputed", False)

    price_key = _write_staging(prices, exec_date, "prices")
    feature_key = _write_staging(features, exec_date, "features")

    return {
        "execution_date": exec_date.isoformat(),
        "price_key": price_key,
        "feature_key": feature_key,
        "price_rows": len(prices),
        "feature_rows": len(features),
    }


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def task_load(summary: dict) -> dict:
    """Upsert prices and features into Postgres. Safe to run any number of times."""
    exec_date = _as_date(summary["execution_date"])

    db.create_all()
    prices = _read_staging(exec_date, "prices")
    features = _read_staging(exec_date, "features")

    # Warm-up rows carry NaN features by construction; loading them would fill
    # the store with rows that exist only to be filtered out later.
    features = features.dropna(subset=["log_return"])

    n_prices = db.upsert(prices, prices_daily)
    n_features = db.upsert(features, features_daily)

    logger.info("loaded %d price rows and %d feature rows", n_prices, n_features)
    return {
        "execution_date": exec_date.isoformat(),
        "price_rows": n_prices,
        "feature_rows": n_features,
    }


# --------------------------------------------------------------------------- #
# quality_check
# --------------------------------------------------------------------------- #
def task_quality_check(
    summary: dict,
    lookback_days: int = 30,
    min_rows: int = 1,
) -> dict:
    """Verify what actually landed in Postgres, then publish the verdict.

    Runs against the database rather than the staged frame: the question is
    whether the feature store is correct, not whether an intermediate file was.
    """
    exec_date = _as_date(summary["execution_date"])
    universe = get_universe()
    start = exec_date - timedelta(days=lookback_days)

    stored = db.read_features(start=start, end=exec_date)
    report: QualityReport = validate_features(
        stored, start, exec_date, universe.exchange, min_rows=min_rows
    )

    newest = pd.to_datetime(stored["date"]).max() if not stored.empty else None
    push_freshness(newest.tz_localize(UTC).to_pydatetime() if newest is not None else None)
    push_quality_result(report.passed, len(report.failures), exec_date.isoformat())

    # Loud failure: the DAG turns red rather than leaving bad data unremarked.
    report.raise_if_failed()

    return {
        "execution_date": exec_date.isoformat(),
        "rows_checked": len(stored),
        "checks_run": len(report.results),
        "warnings": [str(r) for r in report.failures],
    }
