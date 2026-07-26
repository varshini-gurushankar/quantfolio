"""Raw storage is immutable — the property that makes replay trustworthy."""

from __future__ import annotations

from datetime import date

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from quantfolio.config import Settings
from quantfolio.storage.s3 import object_exists, raw_key, read_raw, write_raw

RUN_DATE = date(2023, 1, 3)


@pytest.fixture
def settings() -> Settings:
    # Empty endpoint so boto3 talks to moto's in-memory AWS rather than LocalStack.
    return Settings(
        s3_endpoint_url="",
        s3_raw_bucket="test-raw",
        s3_artifacts_bucket="test-artifacts",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        aws_default_region="us-east-1",
    )


@pytest.fixture
def s3(settings: Settings):
    with mock_aws():
        client = boto3.client("s3", region_name=settings.aws_default_region)
        client.create_bucket(Bucket=settings.s3_raw_bucket)
        client.create_bucket(Bucket=settings.s3_artifacts_bucket)
        yield client


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": "TEST",
            "date": pd.bdate_range("2023-01-03", periods=5),
            "adj_close": [100.0, 101.0, 102.0, 103.0, 104.0],
        }
    )


def test_key_layout_is_partitioned_by_source_ticker_and_date() -> None:
    assert raw_key("yfinance", "AAPL", RUN_DATE) == "raw/yfinance/AAPL/2023-01-03.parquet"


def test_write_then_read_round_trips(s3, settings, frame) -> None:
    write_raw(frame, "yfinance", "TEST", RUN_DATE, settings=settings, client=s3)
    restored = read_raw("yfinance", "TEST", RUN_DATE, settings=settings, client=s3)

    pd.testing.assert_frame_equal(restored, frame)


def test_rerun_does_not_overwrite_raw(s3, settings, frame) -> None:
    """A retry must leave the original bytes alone.

    Raw is the record of what the source actually said at fetch time. If a retry
    silently replaced it, a later replay would reproduce the retry's data, not
    the original run's, and the audit trail would be a fiction.
    """
    write_raw(frame, "yfinance", "TEST", RUN_DATE, settings=settings, client=s3)

    tampered = frame.copy()
    tampered["adj_close"] = 999.0
    write_raw(tampered, "yfinance", "TEST", RUN_DATE, settings=settings, client=s3)

    restored = read_raw("yfinance", "TEST", RUN_DATE, settings=settings, client=s3)
    assert restored["adj_close"].tolist() == [100.0, 101.0, 102.0, 103.0, 104.0]


def test_overwrite_is_possible_when_asked_for_explicitly(s3, settings, frame) -> None:
    """Immutability is the default, not a wall — a deliberate correction is allowed."""
    write_raw(frame, "yfinance", "TEST", RUN_DATE, settings=settings, client=s3)

    corrected = frame.copy()
    corrected["adj_close"] = 111.0
    write_raw(corrected, "yfinance", "TEST", RUN_DATE, settings=settings, client=s3, overwrite=True)

    restored = read_raw("yfinance", "TEST", RUN_DATE, settings=settings, client=s3)
    assert restored["adj_close"].eq(111.0).all()


def test_object_exists_reports_absence_without_raising(s3, settings) -> None:
    assert not object_exists(settings.s3_raw_bucket, "raw/nope/NOPE/2023-01-03.parquet", s3)


def test_different_sources_do_not_collide(s3, settings, frame) -> None:
    """The same ticker and date from two vendors are two separate records."""
    yf = frame.copy()
    av = frame.copy()
    av["adj_close"] = 200.0

    write_raw(yf, "yfinance", "TEST", RUN_DATE, settings=settings, client=s3)
    write_raw(av, "alpha_vantage", "TEST", RUN_DATE, settings=settings, client=s3)

    assert (
        read_raw("yfinance", "TEST", RUN_DATE, settings=settings, client=s3)["adj_close"].iloc[0]
        == 100.0
    )
    assert (
        read_raw("alpha_vantage", "TEST", RUN_DATE, settings=settings, client=s3)["adj_close"].iloc[
            0
        ]
        == 200.0
    )
