#!/usr/bin/env python
"""Run the pipeline for a date range without Airflow.

Useful for seeding a fresh database, or for reproducing a pipeline bug in a
debugger. Because every task is idempotent, running this over a range that was
already processed is harmless: raw partitions are left alone and Postgres rows
are overwritten with identical values.

    python scripts/backfill.py --start 2024-01-01 --end 2024-03-01
    python scripts/backfill.py --latest          # just the most recent session
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta

from quantfolio.config import get_universe
from quantfolio.pipeline import (
    task_fetch,
    task_load,
    task_quality_check,
    task_transform,
    task_validate,
)
from quantfolio.storage.db import create_all
from quantfolio.transforms.calendar import trading_sessions

logger = logging.getLogger("backfill")


def run_one(execution_date: date, run_id: str) -> dict:
    """One full pass of the five pipeline stages for a single date."""
    logger.info("=== %s ===", execution_date)
    fetched = task_fetch(execution_date, run_id=run_id)
    task_validate(fetched)  # raises if raw data is unusable
    staged = task_transform(fetched)
    loaded = task_load(staged)
    checked = task_quality_check(loaded)
    logger.info(
        "%s: %d price rows, %d feature rows, %d checks",
        execution_date,
        loaded["price_rows"],
        loaded["feature_rows"],
        checked["checks_run"],
    )
    return checked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=lambda s: datetime.fromisoformat(s).date())
    parser.add_argument("--end", type=lambda s: datetime.fromisoformat(s).date())
    parser.add_argument(
        "--latest", action="store_true", help="Run only the most recent trading session"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going if one date fails, and report the failures at the end",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    if args.latest:
        recent = trading_sessions(
            date.today() - timedelta(days=10), date.today(), get_universe().exchange
        )
        if len(recent) == 0:
            logger.error("no trading sessions in the last 10 days")
            return 1
        dates = [recent[-1].date()]
    elif args.start and args.end:
        sessions = trading_sessions(args.start, args.end, get_universe().exchange)
        dates = [d.date() for d in sessions]
    else:
        parser.error("provide --latest, or both --start and --end")
        return 2

    if not dates:
        logger.warning("no trading sessions in the requested range")
        return 0

    create_all()
    logger.info("backfilling %d session(s): %s .. %s", len(dates), dates[0], dates[-1])

    failures: list[tuple[date, str]] = []
    run_id = f"backfill_{datetime.now().strftime('%Y%m%dT%H%M%S')}"

    for execution_date in dates:
        try:
            run_one(execution_date, run_id)
        except Exception as exc:  # noqa: BLE001 - the CLI reports failures rather than tracebacking
            logger.error("%s failed: %s", execution_date, exc)
            if not args.continue_on_error:
                return 1
            failures.append((execution_date, str(exc)))

    if failures:
        logger.error("%d/%d dates failed:", len(failures), len(dates))
        for failed_date, error in failures:
            logger.error("  %s: %s", failed_date, error)
        return 1

    logger.info("backfill complete: %d session(s)", len(dates))
    return 0


if __name__ == "__main__":
    sys.exit(main())
