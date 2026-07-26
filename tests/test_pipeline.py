"""Pipeline wiring: the stages hand off correctly and stay idempotent.

Postgres is stubbed (the upsert guarantee has its own test file) while S3 is
real-but-in-memory via moto, because the raw/staging round trip is exactly the
handoff these tests are meant to cover.
"""

from __future__ import annotations

from datetime import date

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from quantfolio import pipeline
from quantfolio.config import Settings
from quantfolio.ingestion.base import PriceSource
from quantfolio.storage.s3 import raw_key

EXEC_DATE = date(2023, 3, 1)
TICKERS = ["AAA", "BBB"]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        s3_endpoint_url="",
        s3_raw_bucket="test-raw",
        s3_artifacts_bucket="test-artifacts",
    )


@pytest.fixture
def aws(settings, monkeypatch):
    with mock_aws():
        client = boto3.client("s3", region_name=settings.aws_default_region)
        client.create_bucket(Bucket=settings.s3_raw_bucket)
        client.create_bucket(Bucket=settings.s3_artifacts_bucket)
        monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
        monkeypatch.setattr("quantfolio.storage.s3.get_settings", lambda: settings)
        yield client


@pytest.fixture
def stub_universe(monkeypatch):
    """A two-ticker universe, so grouping logic is exercised without 14 fetches."""
    from quantfolio.config import Universe, get_universe

    real = get_universe()
    stub = Universe(
        tickers=tuple(TICKERS),
        benchmark="AAA",
        exchange=real.exchange,
        backfill_start="2022-01-01",
        features=real.features,
        cleaning=real.cleaning,
    )
    monkeypatch.setattr(pipeline, "get_universe", lambda: stub)
    return stub


class FakeSource(PriceSource):
    name = "fake"

    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        sessions = pd.bdate_range(start, end)
        base = 100.0 if ticker == "AAA" else 250.0
        close = base + pd.Series(range(len(sessions)), dtype=float) * 0.1
        return pd.DataFrame(
            {
                "ticker": ticker,
                "date": sessions,
                "open": close.values,
                "high": close.values + 1.0,
                "low": close.values - 1.0,
                "close": close.values,
                "adj_close": close.values,
                "volume": 1_000_000,
            }
        )


@pytest.fixture
def stub_infra(monkeypatch, stub_universe):
    """Silence the database and the metrics gateway; capture what was written."""
    written: dict[str, pd.DataFrame] = {}

    monkeypatch.setattr(
        pipeline,
        "fetch_universe",
        lambda tickers, start, end: [FakeSource().fetch_safe(t, start, end) for t in tickers],
    )
    monkeypatch.setattr(pipeline.db, "append", lambda frame, table: len(frame))
    monkeypatch.setattr(pipeline.db, "create_all", lambda *a, **k: None)

    def fake_upsert(frame, table, **kwargs):
        written[table.name] = frame
        return len(frame)

    monkeypatch.setattr(pipeline.db, "upsert", fake_upsert)
    monkeypatch.setattr(pipeline, "push_ingestion_metrics", lambda **kw: None)
    monkeypatch.setattr(pipeline, "push_freshness", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "push_quality_result", lambda *a, **k: None)
    return written


def test_fetch_writes_one_raw_partition_per_ticker(aws, stub_infra, settings) -> None:
    summary = pipeline.task_fetch(EXEC_DATE, run_id="test", warmup_days=120)

    assert set(summary["succeeded"]) == set(TICKERS)
    assert summary["failed"] == []

    for ticker in TICKERS:
        key = raw_key("fake", ticker, EXEC_DATE)
        assert aws.head_object(Bucket=settings.s3_raw_bucket, Key=key)["ContentLength"] > 0


def test_fetch_summary_is_small_enough_for_xcom(aws, stub_infra) -> None:
    """Frames belong in S3; XCom carries only keys and counts."""
    summary = pipeline.task_fetch(EXEC_DATE, run_id="test", warmup_days=120)

    assert set(summary) == {"execution_date", "sources_by_ticker", "rows", "succeeded", "failed"}
    assert all(not isinstance(v, pd.DataFrame) for v in summary.values())


def test_validate_passes_on_well_formed_raw(aws, stub_infra) -> None:
    summary = pipeline.task_fetch(EXEC_DATE, run_id="test", warmup_days=120)
    result = pipeline.task_validate(summary)
    assert result["rows"] > 0


def test_transform_stages_prices_and_features(aws, stub_infra, settings) -> None:
    summary = pipeline.task_fetch(EXEC_DATE, run_id="test", warmup_days=120)
    staged = pipeline.task_transform(summary, warmup_days=120)

    assert staged["feature_rows"] > 0
    for key in (staged["price_key"], staged["feature_key"]):
        assert aws.head_object(Bucket=settings.s3_artifacts_bucket, Key=key)["ContentLength"] > 0


def test_load_drops_warmup_rows_without_a_return(aws, stub_infra) -> None:
    """Rows that exist only as NaN warm-up should never reach the feature store."""
    summary = pipeline.task_fetch(EXEC_DATE, run_id="test", warmup_days=120)
    staged = pipeline.task_transform(summary, warmup_days=120)
    pipeline.task_load(staged)

    features = stub_infra["features_daily"]
    assert features["log_return"].notna().all()


def test_full_run_is_idempotent(aws, stub_infra) -> None:
    """Running every stage twice for one date must produce identical output."""

    def run() -> pd.DataFrame:
        summary = pipeline.task_fetch(EXEC_DATE, run_id="test", warmup_days=120)
        pipeline.task_validate(summary)
        staged = pipeline.task_transform(summary, warmup_days=120)
        pipeline.task_load(staged)
        return stub_infra["features_daily"].copy()

    first, second = run(), run()
    pd.testing.assert_frame_equal(first, second)


def test_fetch_raises_when_every_ticker_fails(aws, stub_infra, monkeypatch) -> None:
    """Total failure is not a partial failure — the run must go red."""
    from quantfolio.ingestion.base import FetchResult

    monkeypatch.setattr(
        pipeline,
        "fetch_universe",
        lambda tickers, start, end: [
            FetchResult(ticker=t, source="fake", frame=None, error="boom") for t in tickers
        ],
    )

    with pytest.raises(RuntimeError, match="no data for any ticker"):
        pipeline.task_fetch(EXEC_DATE, run_id="test", warmup_days=120)


def test_partial_failure_still_produces_a_run(aws, stub_infra, monkeypatch) -> None:
    """One dead ticker must not cost the others their data."""
    from quantfolio.ingestion.base import FetchResult

    def half_broken(tickers, start, end):
        results = []
        for ticker in tickers:
            if ticker == "BBB":
                results.append(FetchResult(ticker, "fake", None, "delisted"))
            else:
                results.append(FakeSource().fetch_safe(ticker, start, end))
        return results

    monkeypatch.setattr(pipeline, "fetch_universe", half_broken)
    summary = pipeline.task_fetch(EXEC_DATE, run_id="test", warmup_days=120)

    assert summary["succeeded"] == ["AAA"]
    assert summary["failed"] == ["BBB"]
