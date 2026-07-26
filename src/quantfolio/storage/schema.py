"""Feature-store schema — the single source of truth for table definitions.

``docker/init-db.sql`` only creates the databases; tables are created from this
metadata so the DDL cannot drift from what the code writes.

Every fact table is keyed on ``(ticker, date)``. That primary key is what makes
the pipeline idempotent: writes go through ``INSERT ... ON CONFLICT DO UPDATE``,
so a retried or backfilled task overwrites its own rows instead of duplicating
them.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    func,
)

metadata = MetaData()

prices_daily = Table(
    "prices_daily",
    metadata,
    Column("ticker", String(16), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("open", Float),
    Column("high", Float),
    Column("low", Float),
    Column("close", Float),
    Column("adj_close", Float),
    Column("volume", BigInteger),
    Column("source", String(32), nullable=False),
    Column("ingested_at", DateTime(timezone=True), server_default=func.now()),
)

features_daily = Table(
    "features_daily",
    metadata,
    Column("ticker", String(16), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("adj_close", Float),
    Column("simple_return", Float),
    Column("log_return", Float),
    Column("sma_20", Float),
    Column("sma_60", Float),
    Column("ema_20", Float),
    Column("macd", Float),
    Column("macd_signal", Float),
    Column("macd_hist", Float),
    Column("rsi_14", Float),
    Column("volatility_20", Float),
    Column("is_imputed", Boolean, server_default="false"),
    Column("computed_at", DateTime(timezone=True), server_default=func.now()),
)

# Audit trail: what was fetched, when, from where, and whether it worked.
# Not keyed on (ticker, date) — a run per execution date per ticker, so retries
# append a new row and the history of failures stays visible.
ingestion_log = Table(
    "ingestion_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(128), nullable=False, index=True),
    Column("ticker", String(16), nullable=False, index=True),
    Column("source", String(32), nullable=False),
    Column("execution_date", Date, nullable=False, index=True),
    Column("row_count", Integer, nullable=False, server_default="0"),
    Column("status", String(16), nullable=False),  # success | failed | empty
    Column("error", Text),
    Column("s3_key", Text),
    Column("logged_at", DateTime(timezone=True), server_default=func.now()),
)

# Phase 2 tables live here too so the schema is created in one pass.
portfolio_weights = Table(
    "portfolio_weights",
    metadata,
    Column("as_of_date", Date, primary_key=True),
    Column("ticker", String(16), primary_key=True),
    Column("weight", Float, nullable=False),
    Column("method", String(32), nullable=False, server_default="markowitz_lw"),
    Column("computed_at", DateTime(timezone=True), server_default=func.now()),
)

model_metrics = Table(
    "model_metrics",
    metadata,
    Column("run_id", String(64), primary_key=True),
    Column("window_end", Date, primary_key=True),
    Column("model_name", String(64), nullable=False),
    Column("mse", Float, nullable=False),
    Column("baseline_mse", Float),
    Column("logged_at", DateTime(timezone=True), server_default=func.now()),
)

# Out-of-sample predictions paired with what actually happened. This is what the
# drift sensor measures: how the deployed model has been scoring lately, rather
# than how it scored at training time.
predictions_daily = Table(
    "predictions_daily",
    metadata,
    Column("model_name", String(64), primary_key=True),
    Column("ticker", String(16), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("y_pred", Float, nullable=False),
    Column("y_true", Float),
    Column("run_id", String(64)),
    Column("predicted_at", DateTime(timezone=True), server_default=func.now()),
)

TABLES = {t.name: t for t in metadata.tables.values()}
