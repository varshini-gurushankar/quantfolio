"""Immutable raw storage.

Raw responses are written once and never edited. Everything downstream is
derived, so any transform bug can be fixed and replayed from these files without
re-hitting a rate-limited API. That is the whole reason S3 exists alongside
Postgres: immutable replayable raw vs. structured queryable features.
"""

from __future__ import annotations

import io
import logging
from datetime import date

import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError

from quantfolio.config import Settings, get_settings

logger = logging.getLogger(__name__)


def s3_client(settings: Settings | None = None):
    settings = settings or get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url or None,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_default_region,
        config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )


def raw_key(source: str, ticker: str, run_date: date) -> str:
    """``raw/{source}/{ticker}/{date}.parquet`` — partitioned for cheap replay."""
    return f"raw/{source}/{ticker}/{run_date.isoformat()}.parquet"


def object_exists(bucket: str, key: str, client=None) -> bool:
    client = client or s3_client()
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def write_raw(
    frame: pd.DataFrame,
    source: str,
    ticker: str,
    run_date: date,
    settings: Settings | None = None,
    client=None,
    overwrite: bool = False,
) -> str:
    """Write one raw partition. Refuses to overwrite unless explicitly told to.

    A re-run for the same execution date is expected (retries, backfills), and
    the correct behaviour is to leave the original bytes alone: raw is the
    record of what the source actually said at fetch time.
    """
    settings = settings or get_settings()
    client = client or s3_client(settings)
    bucket = settings.s3_raw_bucket
    key = raw_key(source, ticker, run_date)

    if not overwrite and object_exists(bucket, key, client):
        logger.info("raw partition already exists, leaving untouched: s3://%s/%s", bucket, key)
        return key

    buf = io.BytesIO()
    frame.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
    buf.seek(0)
    client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    logger.info("wrote %d rows to s3://%s/%s", len(frame), bucket, key)
    return key


def read_raw(
    source: str,
    ticker: str,
    run_date: date,
    settings: Settings | None = None,
    client=None,
) -> pd.DataFrame:
    """Read back one raw partition — the entry point for a replay."""
    settings = settings or get_settings()
    client = client or s3_client(settings)
    obj = client.get_object(Bucket=settings.s3_raw_bucket, Key=raw_key(source, ticker, run_date))
    return pd.read_parquet(io.BytesIO(obj["Body"].read()), engine="pyarrow")


def write_artifact(
    frame: pd.DataFrame,
    key: str,
    settings: Settings | None = None,
    client=None,
) -> str:
    """Write a derived artifact (portfolio weights, backtest output) to the artifacts bucket."""
    settings = settings or get_settings()
    client = client or s3_client(settings)
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
    buf.seek(0)
    client.put_object(Bucket=settings.s3_artifacts_bucket, Key=key, Body=buf.getvalue())
    return key
