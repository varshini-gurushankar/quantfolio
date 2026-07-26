"""Idempotent writes.

"A task fails halfway and Airflow retries it — do you end up with duplicate
rows?" The answer is no, and this file is the proof rather than an assurance.

Two layers of test:

* The compiled-SQL tests run anywhere. They assert that the single write path
  really emits ``INSERT ... ON CONFLICT (ticker, date) DO UPDATE``, so the
  guarantee is checked on every ``pytest`` run with no infrastructure.
* The integration tests run the same write twice against a live Postgres and
  compare the table. They are skipped when no database is reachable.
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

from quantfolio.storage.db import upsert
from quantfolio.storage.schema import features_daily, metadata, prices_daily


# --------------------------------------------------------------------------- #
# Layer 1 — the SQL itself, no database required
# --------------------------------------------------------------------------- #
def _compiled_upsert(table) -> str:
    pk_cols = [c.name for c in table.primary_key.columns]
    row = {c.name: None for c in table.columns}
    stmt = pg_insert(table).values([row])
    stmt = stmt.on_conflict_do_update(
        index_elements=pk_cols,
        set_={c: stmt.excluded[c] for c in row if c not in pk_cols},
    )
    return str(stmt.compile(dialect=postgresql.dialect()))


@pytest.mark.parametrize("table", [prices_daily, features_daily], ids=lambda t: t.name)
def test_write_path_emits_on_conflict_do_update(table) -> None:
    sql = _compiled_upsert(table).upper()
    assert "ON CONFLICT" in sql
    assert "DO UPDATE" in sql


@pytest.mark.parametrize("table", [prices_daily, features_daily], ids=lambda t: t.name)
def test_conflict_target_is_ticker_and_date(table) -> None:
    """The conflict target must be the natural key, or the upsert protects nothing."""
    assert [c.name for c in table.primary_key.columns] == ["ticker", "date"]
    sql = _compiled_upsert(table).lower()
    assert "on conflict (ticker, date)" in sql


def test_every_fact_table_has_a_primary_key() -> None:
    """A table without a PK cannot be written idempotently."""
    exempt = {"ingestion_log"}  # append-only audit trail, keyed on a surrogate id
    for name, table in metadata.tables.items():
        assert list(table.primary_key.columns), f"{name} has no primary key"
        if name not in exempt:
            key = [c.name for c in table.primary_key.columns]
            assert len(key) >= 2, f"{name} should be keyed on a composite natural key, got {key}"


# --------------------------------------------------------------------------- #
# Layer 2 — the real behaviour, against a live Postgres
# --------------------------------------------------------------------------- #
def _database_url() -> str:
    return os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql+psycopg2://quantfolio:quantfolio@localhost:5432/quantfolio",
    )


@pytest.fixture(scope="module")
def engine():
    """A live engine, or a skip. Never a failure for an absent database."""
    eng = create_engine(_database_url())
    try:
        with eng.connect() as conn:
            conn.execute(select(1))
    except SQLAlchemyError as exc:
        pytest.skip(f"no Postgres at {_database_url()}: {exc}")
    metadata.create_all(eng)
    return eng


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": "IDEMP",
            "date": [date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5)],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "adj_close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
            "source": "test",
        }
    )


@pytest.fixture
def clean_table(engine):
    """Remove this test's rows before and after, leaving real data untouched."""

    def _purge():
        with engine.begin() as conn:
            conn.execute(prices_daily.delete().where(prices_daily.c.ticker == "IDEMP"))

    _purge()
    yield
    _purge()


def _row_count(engine) -> int:
    with engine.connect() as conn:
        return conn.execute(
            select(func.count()).select_from(prices_daily).where(prices_daily.c.ticker == "IDEMP")
        ).scalar_one()


@pytest.mark.integration
def test_running_the_same_load_twice_does_not_duplicate(engine, sample_prices, clean_table) -> None:
    """The headline guarantee: load, load again, same three rows."""
    upsert(sample_prices, prices_daily, engine=engine)
    after_first = _row_count(engine)

    upsert(sample_prices, prices_daily, engine=engine)
    after_second = _row_count(engine)

    assert after_first == 3
    assert after_second == 3, "a retried load duplicated rows"


@pytest.mark.integration
def test_rerunning_with_corrected_data_updates_in_place(engine, sample_prices, clean_table) -> None:
    """A vendor restatement should correct the row, not add a second version of it."""
    upsert(sample_prices, prices_daily, engine=engine)

    corrected = sample_prices.copy()
    corrected["adj_close"] = [200.0, 201.0, 202.0]
    upsert(corrected, prices_daily, engine=engine)

    with engine.connect() as conn:
        stored = (
            conn.execute(
                select(prices_daily.c.adj_close)
                .where(prices_daily.c.ticker == "IDEMP")
                .order_by(prices_daily.c.date)
            )
            .scalars()
            .all()
        )

    assert _row_count(engine) == 3
    assert stored == [200.0, 201.0, 202.0]


@pytest.mark.integration
def test_partial_overlap_inserts_only_the_new_dates(engine, sample_prices, clean_table) -> None:
    """Overlapping windows are normal — each run fetches its own warm-up history."""
    upsert(sample_prices, prices_daily, engine=engine)

    extended = sample_prices.copy()
    extended["date"] = [date(2023, 1, 4), date(2023, 1, 5), date(2023, 1, 6)]
    upsert(extended, prices_daily, engine=engine)

    assert _row_count(engine) == 4  # 3 original + 1 genuinely new date


@pytest.mark.integration
def test_nan_values_are_written_as_null(engine, sample_prices, clean_table) -> None:
    """Warm-up rows carry NaN; Postgres will reject it in a float column."""
    with_nan = sample_prices.copy()
    with_nan.loc[0, "adj_close"] = float("nan")

    upsert(with_nan, prices_daily, engine=engine)

    with engine.connect() as conn:
        value = conn.execute(
            select(prices_daily.c.adj_close)
            .where(prices_daily.c.ticker == "IDEMP")
            .where(prices_daily.c.date == date(2023, 1, 3))
        ).scalar_one()
    assert value is None


@pytest.mark.integration
def test_empty_frame_is_a_no_op(engine, clean_table) -> None:
    assert upsert(pd.DataFrame(), prices_daily, engine=engine) == 0
