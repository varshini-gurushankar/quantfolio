#!/usr/bin/env python
"""Demonstrate that the drift sensor actually fires.

A monitoring rule nobody has ever seen trigger is a rule nobody should trust.
This script writes synthetic predictions whose error is deliberately inflated
over a recent window, then runs the real detector against them.

    python scripts/inject_drift.py --dry-run    # no database, prints the verdict
    python scripts/inject_drift.py              # writes to Postgres, then checks
    python scripts/inject_drift.py --cleanup    # remove the synthetic rows

The `--dry-run` mode needs no infrastructure and is what the test suite uses.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

import numpy as np
import pandas as pd

from quantfolio.models.drift import detect_drift, store_predictions

logger = logging.getLogger("inject_drift")

SYNTHETIC_MODEL = "drift_demo_model"
REFERENCE_MSE = 1.0e-4  # a plausible daily-return MSE


def build_predictions(
    n_days: int = 90,
    drift_days: int = 25,
    error_multiple: float = 3.0,
    end: date | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Predictions that score normally, then degrade over the final window.

    Error is inflated by widening the gap between prediction and truth, which is
    what a regime change looks like: the model keeps producing plausible
    numbers, they are just increasingly wrong.
    """
    rng = np.random.default_rng(seed)
    end = end or date.today()
    dates = pd.bdate_range(end=pd.Timestamp(end), periods=n_days)
    tickers = ["AAA", "BBB", "CCC"]

    base_error = np.sqrt(REFERENCE_MSE)
    rows = []

    for i, day in enumerate(dates):
        drifting = i >= len(dates) - drift_days
        # Standard deviation scales with sqrt of the MSE multiple.
        scale = base_error * (np.sqrt(error_multiple) if drifting else 1.0)

        for ticker in tickers:
            y_true = rng.normal(0.0, 0.011)
            rows.append(
                {
                    "ticker": ticker,
                    "date": day.date(),
                    "y_true": y_true,
                    "y_pred": y_true + rng.normal(0.0, scale),
                    "is_drift_period": drifting,
                }
            )

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="skip Postgres entirely")
    parser.add_argument("--cleanup", action="store_true", help="delete synthetic rows and exit")
    parser.add_argument("--drift-days", type=int, default=25)
    parser.add_argument("--error-multiple", type=float, default=3.0)
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--threshold", type=float, default=1.5)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if args.cleanup:
        return _cleanup()

    predictions = build_predictions(drift_days=args.drift_days, error_multiple=args.error_multiple)
    logger.info(
        "built %d synthetic predictions, last %d days degraded %.1fx",
        len(predictions),
        args.drift_days,
        args.error_multiple,
    )

    if not args.dry_run:
        written = store_predictions(
            predictions[["ticker", "date", "y_pred", "y_true"]],
            model_name=SYNTHETIC_MODEL,
            run_id="drift_demo",
        )
        logger.info("wrote %d rows to predictions_daily", written)

    # The control: the same detector on the clean period only must NOT fire.
    clean = predictions[~predictions["is_drift_period"]]
    clean_report = detect_drift(
        clean, REFERENCE_MSE, SYNTHETIC_MODEL, window=args.window, threshold=args.threshold
    )

    full_report = detect_drift(
        predictions, REFERENCE_MSE, SYNTHETIC_MODEL, window=args.window, threshold=args.threshold
    )

    print()
    print("Control (clean period only)  :", clean_report.summary())
    print("Injected drift (full series) :", full_report.summary())
    print()

    if full_report.breached and not clean_report.breached:
        print("PASS: the sensor fires on injected drift and stays quiet on clean data.")
        return 0

    print("FAIL: sensor did not behave as expected.")
    return 1


def _cleanup() -> int:
    from sqlalchemy import delete

    from quantfolio.storage.db import get_engine
    from quantfolio.storage.schema import predictions_daily

    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            delete(predictions_daily).where(predictions_daily.c.model_name == SYNTHETIC_MODEL)
        )
    logger.info("removed %d synthetic rows", result.rowcount)
    return 0


if __name__ == "__main__":
    sys.exit(main())
