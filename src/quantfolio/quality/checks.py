"""Data quality gates.

Checks return a report rather than raising, so a run can log every failure at
once instead of stopping at the first. The DAG task then raises if any blocking
check failed — bad data must fail the pipeline loudly rather than land in the
feature store and get silently modelled on later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from quantfolio.transforms.calendar import missing_sessions

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    blocking: bool = True

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


@dataclass
class QualityReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        logger.info("%s", result)
        self.results.append(result)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    @property
    def blocking_failures(self) -> list[CheckResult]:
        return [r for r in self.failures if r.blocking]

    @property
    def passed(self) -> bool:
        return not self.blocking_failures

    def raise_if_failed(self) -> None:
        if self.blocking_failures:
            detail = "\n".join(str(r) for r in self.blocking_failures)
            raise QualityCheckFailed(
                f"{len(self.blocking_failures)} blocking check(s) failed:\n{detail}"
            )


class QualityCheckFailed(RuntimeError):
    """Raised by the DAG's quality gate. Fails the task, blocks the load."""


REQUIRED_PRICE_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]


def check_schema(frame: pd.DataFrame, required: list[str], name: str = "schema") -> CheckResult:
    missing = [c for c in required if c not in frame.columns]
    return CheckResult(
        name=name,
        passed=not missing,
        detail="all required columns present" if not missing else f"missing columns: {missing}",
    )


def check_not_empty(frame: pd.DataFrame, name: str = "not_empty") -> CheckResult:
    return CheckResult(name=name, passed=not frame.empty, detail=f"{len(frame)} rows")


def check_nulls(
    frame: pd.DataFrame, columns: list[str], max_null_fraction: float = 0.0
) -> CheckResult:
    if frame.empty:
        return CheckResult("nulls", False, "frame is empty")
    offenders = {}
    for col in columns:
        if col not in frame.columns:
            continue
        frac = float(frame[col].isna().mean())
        if frac > max_null_fraction:
            offenders[col] = round(frac, 4)
    return CheckResult(
        name="nulls",
        passed=not offenders,
        detail="no nulls above threshold"
        if not offenders
        else f"null fractions over limit: {offenders}",
    )


def check_row_count(frame: pd.DataFrame, minimum: int) -> CheckResult:
    return CheckResult(
        name="row_count",
        passed=len(frame) >= minimum,
        detail=f"{len(frame)} rows (minimum {minimum})",
    )


def check_date_continuity(
    frame: pd.DataFrame,
    start: date,
    end: date,
    exchange: str = "NYSE",
    max_missing: int = 0,
) -> CheckResult:
    """Every session the exchange was open for should have a row."""
    if frame.empty:
        return CheckResult("date_continuity", False, "frame is empty")

    per_ticker_missing: dict[str, int] = {}
    for ticker, group in frame.groupby("ticker"):
        gaps = missing_sessions(group, start, end, exchange)
        if len(gaps) > max_missing:
            per_ticker_missing[str(ticker)] = len(gaps)

    return CheckResult(
        name="date_continuity",
        passed=not per_ticker_missing,
        detail=(
            f"no gaps beyond {max_missing} session(s)"
            if not per_ticker_missing
            else f"missing sessions: {per_ticker_missing}"
        ),
    )


def check_range(
    frame: pd.DataFrame,
    column: str,
    low: float,
    high: float,
    blocking: bool = True,
) -> CheckResult:
    """Bounds check, e.g. RSI must lie in [0, 100]."""
    if column not in frame.columns:
        return CheckResult(f"range_{column}", False, f"column {column} absent", blocking=blocking)

    values = frame[column].dropna()
    if values.empty:
        return CheckResult(
            f"range_{column}", True, "no non-null values to check", blocking=blocking
        )

    out_of_range = values[(values < low) | (values > high)]
    return CheckResult(
        name=f"range_{column}",
        passed=out_of_range.empty,
        detail=(
            f"{len(values)} values within [{low}, {high}]"
            if out_of_range.empty
            else f"{len(out_of_range)} values outside [{low}, {high}] "
            f"(min={values.min():.4f}, max={values.max():.4f})"
        ),
        blocking=blocking,
    )


def check_positive_prices(frame: pd.DataFrame) -> CheckResult:
    cols = [c for c in ("open", "high", "low", "close", "adj_close") if c in frame.columns]
    if not cols:
        return CheckResult("positive_prices", False, "no price columns present")
    bad = int((frame[cols] <= 0).sum().sum())
    return CheckResult("positive_prices", bad == 0, f"{bad} non-positive price values")


def check_ohlc_consistency(frame: pd.DataFrame) -> CheckResult:
    """High must be the high and low must be the low."""
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(frame.columns):
        return CheckResult("ohlc_consistency", False, "OHLC columns absent")

    df = frame.dropna(subset=list(needed))
    violations = int(
        (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        ).sum()
    )
    return CheckResult("ohlc_consistency", violations == 0, f"{violations} inconsistent bars")


def validate_raw_prices(
    frame: pd.DataFrame,
    min_rows: int = 1,
    max_null_fraction: float = 0.0,
) -> QualityReport:
    """Gate between fetch and transform: is this frame structurally usable?"""
    report = QualityReport()
    report.add(check_schema(frame, REQUIRED_PRICE_COLUMNS))
    report.add(check_not_empty(frame))
    report.add(check_row_count(frame, min_rows))
    if not frame.empty and set(REQUIRED_PRICE_COLUMNS).issubset(frame.columns):
        report.add(check_nulls(frame, ["ticker", "date", "adj_close"], max_null_fraction))
        report.add(check_positive_prices(frame))
        report.add(check_ohlc_consistency(frame))
    return report


def validate_features(
    frame: pd.DataFrame,
    start: date,
    end: date,
    exchange: str = "NYSE",
    min_rows: int = 1,
) -> QualityReport:
    """Gate after load: are the computed features in range and complete?"""
    report = QualityReport()
    report.add(check_not_empty(frame))
    report.add(check_row_count(frame, min_rows))
    if frame.empty:
        return report

    report.add(check_date_continuity(frame, start, end, exchange))
    report.add(check_range(frame, "rsi_14", 0.0, 100.0))
    # Volatility and returns are bounded loosely: these catch corruption, not
    # unusual markets, so they warn rather than block.
    report.add(check_range(frame, "volatility_20", 0.0, 5.0, blocking=False))
    report.add(check_range(frame, "log_return", -1.0, 1.0, blocking=False))
    return report
