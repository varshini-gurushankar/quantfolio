"""Quality gates. A check that never fails is not a check."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantfolio.quality.checks import (
    QualityCheckFailed,
    check_date_continuity,
    check_nulls,
    check_ohlc_consistency,
    check_positive_prices,
    check_range,
    check_row_count,
    check_schema,
    validate_features,
    validate_raw_prices,
)
from quantfolio.transforms.features import compute_features


def test_schema_check_names_the_missing_columns() -> None:
    frame = pd.DataFrame({"ticker": ["A"], "date": [date(2023, 1, 3)]})
    result = check_schema(frame, ["ticker", "date", "adj_close"])
    assert not result.passed
    assert "adj_close" in result.detail


def test_row_count_check() -> None:
    frame = pd.DataFrame({"a": range(5)})
    assert check_row_count(frame, 5).passed
    assert not check_row_count(frame, 6).passed


def test_null_check_flags_the_offending_column() -> None:
    frame = pd.DataFrame({"adj_close": [1.0, None, 3.0], "ticker": ["A"] * 3})
    result = check_nulls(frame, ["adj_close", "ticker"])
    assert not result.passed
    assert "adj_close" in result.detail


def test_negative_prices_are_rejected() -> None:
    frame = pd.DataFrame(
        {"open": [1.0], "high": [2.0], "low": [-0.5], "close": [1.5], "adj_close": [1.5]}
    )
    assert not check_positive_prices(frame).passed


def test_ohlc_inconsistency_is_detected() -> None:
    frame = pd.DataFrame({"open": [10.0], "high": [9.0], "low": [8.0], "close": [8.5]})
    assert not check_ohlc_consistency(frame).passed


def test_valid_ohlc_passes() -> None:
    frame = pd.DataFrame({"open": [9.0], "high": [10.0], "low": [8.0], "close": [8.5]})
    assert check_ohlc_consistency(frame).passed


def test_rsi_out_of_range_is_caught() -> None:
    """RSI in [0, 100] is the range check named in the project spec."""
    frame = pd.DataFrame({"rsi_14": [10.0, 50.0, 140.0]})
    result = check_range(frame, "rsi_14", 0.0, 100.0)
    assert not result.passed
    assert "outside" in result.detail


def test_rsi_within_range_passes(price_frame: pd.DataFrame) -> None:
    features = compute_features(price_frame)
    assert check_range(features, "rsi_14", 0.0, 100.0).passed


def test_date_continuity_detects_a_missing_session() -> None:
    frame = pd.DataFrame({"ticker": "TEST", "date": pd.to_datetime(["2023-01-03", "2023-01-05"])})
    result = check_date_continuity(frame, date(2023, 1, 3), date(2023, 1, 5))
    assert not result.passed
    assert "TEST" in result.detail


def test_report_raises_only_on_blocking_failures() -> None:
    frame = pd.DataFrame({"volatility_20": [99.0]})  # outside the non-blocking bound

    report = validate_features(
        frame.assign(ticker="T", date=pd.to_datetime(["2023-01-03"])),
        date(2023, 1, 3),
        date(2023, 1, 3),
    )
    warnings = [r for r in report.failures if not r.blocking]
    assert warnings, "an out-of-range volatility should warn"


def test_validate_raw_rejects_a_malformed_frame() -> None:
    frame = pd.DataFrame({"ticker": ["A"], "date": [date(2023, 1, 3)]})
    report = validate_raw_prices(frame)
    assert not report.passed
    with pytest.raises(QualityCheckFailed):
        report.raise_if_failed()


def test_validate_raw_accepts_a_good_frame(price_frame: pd.DataFrame) -> None:
    report = validate_raw_prices(price_frame)
    assert report.passed, [str(r) for r in report.failures]
    report.raise_if_failed()


def test_failure_message_lists_every_blocking_failure() -> None:
    """Reporting all failures at once beats fixing them one run at a time."""
    frame = pd.DataFrame({"ticker": [], "date": []})
    report = validate_raw_prices(frame)

    with pytest.raises(QualityCheckFailed) as excinfo:
        report.raise_if_failed()
    assert len(report.blocking_failures) >= 2
    assert "blocking check(s) failed" in str(excinfo.value)


def test_nan_features_do_not_trip_the_range_check() -> None:
    """Warm-up NaNs are expected and must not be read as out of range."""
    frame = pd.DataFrame({"rsi_14": [np.nan, np.nan, 55.0]})
    assert check_range(frame, "rsi_14", 0.0, 100.0).passed
