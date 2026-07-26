"""Ingestion: retries, fallback, and failure isolation.

No test here touches the network. The point is the error-handling behaviour,
which is exactly what a live call makes impossible to test reliably.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quantfolio.ingestion.base import (
    PermanentSourceError,
    PriceSource,
    TransientSourceError,
    normalize_frame,
    with_backoff,
)
from quantfolio.ingestion.runner import fetch_ticker, fetch_universe

START = date(2023, 1, 3)
END = date(2023, 1, 10)


def _valid_frame(ticker: str = "TEST", rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.bdate_range("2023-01-03", periods=rows),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "adj_close": 100.5,
            "volume": 1_000,
        }
    )


class StubSource(PriceSource):
    """A source with scripted behaviour, to drive the fallback logic."""

    def __init__(self, name: str, behaviour: str = "ok") -> None:
        self.name = name
        self.behaviour = behaviour
        self.calls: list[str] = []

    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        self.calls.append(ticker)
        if self.behaviour == "ok":
            return _valid_frame(ticker)
        if self.behaviour == "empty":
            return _valid_frame(ticker, rows=0)
        if self.behaviour == "transient":
            raise TransientSourceError(f"{ticker}: rate limited")
        raise PermanentSourceError(f"{ticker}: unknown symbol")


# --------------------------------------------------------------------------- #
# normalization
# --------------------------------------------------------------------------- #
def test_normalize_enforces_the_canonical_column_order() -> None:
    frame = _valid_frame().drop(columns=["ticker"])
    result = normalize_frame(frame, "TEST")
    assert list(result.columns) == [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]


def test_normalize_rejects_a_frame_missing_columns() -> None:
    with pytest.raises(PermanentSourceError, match="missing columns"):
        normalize_frame(pd.DataFrame({"date": [], "close": []}), "TEST")


def test_normalize_sorts_and_deduplicates() -> None:
    frame = _valid_frame(rows=3)
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    shuffled = duplicated.sample(frac=1.0, random_state=3)

    result = normalize_frame(shuffled, "TEST")

    assert len(result) == 3
    assert result["date"].is_monotonic_increasing


# --------------------------------------------------------------------------- #
# retry / backoff
# --------------------------------------------------------------------------- #
def test_transient_errors_are_retried() -> None:
    attempts = {"n": 0}

    @with_backoff(max_attempts=3, initial_wait=0.001, max_wait=0.002)
    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TransientSourceError("rate limited")
        return "ok"

    assert flaky() == "ok"
    assert attempts["n"] == 3


def test_permanent_errors_are_not_retried() -> None:
    """Retrying an unknown symbol just burns a rate-limited quota."""
    attempts = {"n": 0}

    @with_backoff(max_attempts=5, initial_wait=0.001, max_wait=0.002)
    def broken() -> str:
        attempts["n"] += 1
        raise PermanentSourceError("unknown symbol")

    with pytest.raises(PermanentSourceError):
        broken()
    assert attempts["n"] == 1


def test_retries_eventually_give_up() -> None:
    @with_backoff(max_attempts=2, initial_wait=0.001, max_wait=0.002)
    def always_failing() -> str:
        raise TransientSourceError("still rate limited")

    with pytest.raises(TransientSourceError):
        always_failing()


# --------------------------------------------------------------------------- #
# failures as data
# --------------------------------------------------------------------------- #
def test_fetch_safe_returns_a_result_instead_of_raising() -> None:
    result = StubSource("bad", "permanent").fetch_safe("TEST", START, END)
    assert not result.ok
    assert "unknown symbol" in result.error


def test_fetch_safe_reports_success() -> None:
    result = StubSource("good").fetch_safe("TEST", START, END)
    assert result.ok
    assert result.row_count == 3


# --------------------------------------------------------------------------- #
# fallback
# --------------------------------------------------------------------------- #
def test_secondary_source_is_used_when_the_primary_fails() -> None:
    primary, secondary = StubSource("primary", "transient"), StubSource("secondary", "ok")

    result = fetch_ticker("TEST", START, END, [primary, secondary])

    assert result.ok
    assert result.source == "secondary"
    assert secondary.calls == ["TEST"]


def test_secondary_is_not_called_when_the_primary_succeeds() -> None:
    """The fallback exists for failures; calling it anyway wastes a scarce quota."""
    primary, secondary = StubSource("primary", "ok"), StubSource("secondary", "ok")

    result = fetch_ticker("TEST", START, END, [primary, secondary])

    assert result.source == "primary"
    assert secondary.calls == []


def test_an_empty_response_triggers_the_fallback() -> None:
    primary, secondary = StubSource("primary", "empty"), StubSource("secondary", "ok")
    assert fetch_ticker("TEST", START, END, [primary, secondary]).source == "secondary"


def test_all_sources_failing_yields_a_combined_error() -> None:
    sources = [StubSource("primary", "transient"), StubSource("secondary", "permanent")]

    result = fetch_ticker("TEST", START, END, sources)

    assert not result.ok
    assert "primary" in result.error and "secondary" in result.error


# --------------------------------------------------------------------------- #
# partial failure across the universe
# --------------------------------------------------------------------------- #
def test_one_bad_ticker_does_not_abort_the_universe() -> None:
    """The property the DAG depends on: a delisted symbol costs the others nothing."""

    class SelectiveSource(PriceSource):
        name = "selective"

        def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame:
            if ticker == "BROKEN":
                raise PermanentSourceError("unknown symbol")
            return _valid_frame(ticker)

    results = fetch_universe(["AAPL", "BROKEN", "MSFT"], START, END, [SelectiveSource()])

    succeeded = [r.ticker for r in results if r.ok]
    failed = [r.ticker for r in results if not r.ok]

    assert succeeded == ["AAPL", "MSFT"]
    assert failed == ["BROKEN"]
    assert len(results) == 3, "every ticker must be accounted for, successful or not"
