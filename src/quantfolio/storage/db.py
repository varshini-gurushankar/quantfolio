"""Postgres access. Every write in the project goes through ``upsert``.

Idempotency is structural here rather than a discipline each task has to
remember: there is one write path, and it is an ``INSERT ... ON CONFLICT DO
UPDATE`` on the table's primary key. A task that retries, or a backfill that
re-runs an old execution date, overwrites its own rows and changes nothing else.
"""

from __future__ import annotations

import functools
import logging
import math
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import Table, create_engine, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

# sqlalchemy.engine.Engine resolves on both 1.4 and 2.x; the top-level
# `from sqlalchemy import Engine` shortcut exists only in 2.x.
from sqlalchemy.engine import Engine

from quantfolio.config import get_settings
from quantfolio.storage.schema import metadata

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=4)
def get_engine(database_url: str | None = None) -> Engine:
    url = database_url or get_settings().database_url
    return create_engine(url, pool_pre_ping=True, future=True)


def create_all(engine: Engine | None = None) -> None:
    """Create any missing tables. Safe to call on every boot."""
    engine = engine or get_engine()
    metadata.create_all(engine)
    logger.info("schema ensured: %s", ", ".join(sorted(metadata.tables)))


def _records(frame: pd.DataFrame, table: Table) -> list[dict[str, Any]]:
    """Frame -> row dicts, keeping only real table columns and nulling NaNs."""
    cols = [c.name for c in table.columns if c.name in frame.columns]
    rows = frame[cols].to_dict(orient="records")
    for row in rows:
        for key, value in row.items():
            # Postgres will not take NaN/NaT in a float or date column.
            if value is None or value is pd.NaT:
                row[key] = None
            elif isinstance(value, float) and math.isnan(value):
                row[key] = None
            elif value is pd.NA:
                row[key] = None
            elif hasattr(value, "item") and not isinstance(value, (str, bytes)):
                # numpy scalar -> python scalar (psycopg2 cannot adapt np.int64)
                row[key] = value.item()
    return rows


def upsert(
    frame: pd.DataFrame,
    table: Table,
    engine: Engine | None = None,
    chunk_size: int = 5_000,
) -> int:
    """Insert rows, updating on primary-key conflict. Returns rows written.

    This is the answer to "a task fails and retries — do you get duplicates?":
    no, because the conflict target is the primary key and the action is an
    update of the non-key columns.
    """
    if frame.empty:
        return 0

    engine = engine or get_engine()
    pk_cols = [c.name for c in table.primary_key.columns]
    rows = _records(frame, table)
    if not rows:
        return 0

    update_cols = [c for c in rows[0] if c not in pk_cols]
    written = 0

    with engine.begin() as conn:
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            stmt = pg_insert(table).values(chunk)
            if update_cols:
                stmt = stmt.on_conflict_do_update(
                    index_elements=pk_cols,
                    set_={c: stmt.excluded[c] for c in update_cols},
                )
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=pk_cols)
            conn.execute(stmt)
            written += len(chunk)

    logger.info("upserted %d rows into %s", written, table.name)
    return written


def append(frame: pd.DataFrame, table: Table, engine: Engine | None = None) -> int:
    """Plain insert, for append-only tables such as ``ingestion_log``."""
    if frame.empty:
        return 0
    engine = engine or get_engine()
    rows = _records(frame, table)
    with engine.begin() as conn:
        conn.execute(table.insert(), rows)
    return len(rows)


def read_prices(
    tickers: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    engine: Engine | None = None,
) -> pd.DataFrame:
    """Load daily prices, optionally filtered. Used by transforms and the API."""
    from quantfolio.storage.schema import prices_daily

    engine = engine or get_engine()
    stmt = select(prices_daily)
    if tickers:
        stmt = stmt.where(prices_daily.c.ticker.in_(tickers))
    if start:
        stmt = stmt.where(prices_daily.c.date >= start)
    if end:
        stmt = stmt.where(prices_daily.c.date <= end)
    stmt = stmt.order_by(prices_daily.c.ticker, prices_daily.c.date)

    with engine.connect() as conn:
        return pd.read_sql(stmt, conn)


def read_features(
    tickers: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    engine: Engine | None = None,
) -> pd.DataFrame:
    from quantfolio.storage.schema import features_daily

    engine = engine or get_engine()
    stmt = select(features_daily)
    if tickers:
        stmt = stmt.where(features_daily.c.ticker.in_(tickers))
    if start:
        stmt = stmt.where(features_daily.c.date >= start)
    if end:
        stmt = stmt.where(features_daily.c.date <= end)
    stmt = stmt.order_by(features_daily.c.ticker, features_daily.c.date)

    with engine.connect() as conn:
        return pd.read_sql(stmt, conn)
