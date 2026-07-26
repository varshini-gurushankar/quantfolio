"""Exchange-calendar alignment, the alternative to blind forward-filling."""

from __future__ import annotations

from datetime import date

import pandas as pd

from quantfolio.transforms.calendar import align_to_sessions, missing_sessions, trading_sessions

START = date(2023, 1, 1)
END = date(2023, 3, 31)


def test_weekends_are_not_sessions() -> None:
    sessions = trading_sessions(START, END)
    assert not any(d.weekday() >= 5 for d in sessions)


def test_market_holidays_are_excluded() -> None:
    """Good Friday 2023 was a weekday the NYSE was closed."""
    sessions = trading_sessions(date(2023, 4, 1), date(2023, 4, 15))
    assert pd.Timestamp("2023-04-07") not in sessions
    assert pd.Timestamp("2023-04-06") in sessions


def _frame(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": "TEST",
            "date": pd.to_datetime(dates),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "adj_close": 100.5,
            "volume": 1_000_000,
        }
    )


def test_a_missing_session_is_filled_and_flagged() -> None:
    # 2023-01-04 omitted; it was a trading day.
    frame = _frame(["2023-01-03", "2023-01-05", "2023-01-06"])
    aligned = align_to_sessions(frame, date(2023, 1, 3), date(2023, 1, 6))

    filled = aligned[aligned["date"] == pd.Timestamp("2023-01-04")]
    assert len(filled) == 1
    assert bool(filled["is_imputed"].iloc[0]) is True
    assert filled["adj_close"].iloc[0] == 100.5


def test_imputed_sessions_report_zero_volume() -> None:
    """Carrying volume forward would fabricate shares that never traded."""
    frame = _frame(["2023-01-03", "2023-01-05"])
    aligned = align_to_sessions(frame, date(2023, 1, 3), date(2023, 1, 5))

    imputed = aligned[aligned["is_imputed"]]
    assert len(imputed) == 1
    assert imputed["volume"].iloc[0] == 0


def test_real_rows_are_not_marked_imputed() -> None:
    frame = _frame(["2023-01-03", "2023-01-04", "2023-01-05"])
    aligned = align_to_sessions(frame, date(2023, 1, 3), date(2023, 1, 5))
    assert not aligned["is_imputed"].any()


def test_rows_on_non_session_dates_are_dropped() -> None:
    frame = _frame(["2023-01-03", "2023-01-07", "2023-01-04"])  # the 7th is a Saturday
    aligned = align_to_sessions(frame, date(2023, 1, 3), date(2023, 1, 6))
    assert pd.Timestamp("2023-01-07") not in set(aligned["date"])


def test_long_gaps_are_left_missing_rather_than_stale() -> None:
    """A month-long hole is a data problem to surface, not one to paper over."""
    frame = _frame(["2023-01-03", "2023-03-01"])
    aligned = align_to_sessions(frame, date(2023, 1, 3), date(2023, 3, 1), max_gap=5)

    assert aligned["close"].isna().sum() > 0, "gaps beyond max_gap must stay NaN"


def test_missing_sessions_reports_the_gaps() -> None:
    frame = _frame(["2023-01-03", "2023-01-05"])
    gaps = missing_sessions(frame, date(2023, 1, 3), date(2023, 1, 6))
    assert set(gaps) == {pd.Timestamp("2023-01-04"), pd.Timestamp("2023-01-06")}


def test_alignment_is_idempotent() -> None:
    """Aligning an already-aligned frame must change nothing."""
    frame = _frame(["2023-01-03", "2023-01-05", "2023-01-06"])
    once = align_to_sessions(frame, date(2023, 1, 3), date(2023, 1, 6))
    twice = align_to_sessions(once.drop(columns=["is_imputed"]), date(2023, 1, 3), date(2023, 1, 6))

    pd.testing.assert_series_equal(once["adj_close"], twice["adj_close"])
