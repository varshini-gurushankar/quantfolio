#!/usr/bin/env python
"""Run the Keras-vs-PyTorch walk-forward comparison from the command line.

    python scripts/train.py                      # against Postgres, logging to MLflow
    python scripts/train.py --no-mlflow          # skip tracking, just print the table
    python scripts/train.py --local-mlflow       # log to ./mlruns instead of a server
    python scripts/train.py --model keras_mlp    # one model only

The comparison table it prints is the honest version of the résumé claim: every
model's out-of-sample MSE next to the zero-prediction baseline, so "X% lower
MSE" always has a stated referent.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta

from quantfolio.models.keras_mlp import KerasMLP
from quantfolio.models.torch_lstm import TorchLSTM
from quantfolio.models.train import compare, load_training_data, run_experiment, walk_forward_train

logger = logging.getLogger("train")

MODELS = {"keras_mlp": KerasMLP, "torch_lstm": TorchLSTM}

# Local tracking without a server. MLflow's filesystem backend ("./mlruns") is
# in maintenance mode and raises on recent versions, so SQLite is the default.
LOCAL_TRACKING_URI = "sqlite:///mlflow.db"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--years", type=int, default=5, help="years of history to train on")
    parser.add_argument("--splits", type=int, default=5, help="walk-forward folds")
    parser.add_argument("--test-size", type=int, default=63, help="sessions per test fold")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--no-mlflow", action="store_true", help="skip experiment tracking")
    parser.add_argument(
        "--local-mlflow",
        action="store_true",
        help="log to a local ./mlflow.db instead of the tracking server",
    )
    parser.add_argument("--register", action="store_true", help="register the winning model")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    end = date.today()
    start = end - timedelta(days=365 * args.years)

    try:
        dataset = load_training_data(start=str(start), end=str(end))
    except Exception as exc:  # noqa: BLE001 - a missing database is a user error, not a crash
        logger.error("could not load training data: %s", exc)
        logger.error("run the data pipeline first (make up && make trigger)")
        return 1

    names = list(MODELS) if args.model == "all" else [args.model]
    split_kwargs = {"n_splits": args.splits, "test_size": args.test_size}

    results = []
    for name in names:
        model = MODELS[name](epochs=args.epochs)
        logger.info("=== %s (%s) ===", model.name, model.framework)

        if args.no_mlflow:
            result = walk_forward_train(dataset, model, **split_kwargs)
        else:
            result = run_experiment(
                dataset,
                model,
                # MLflow put the plain-file store into maintenance mode; SQLite
                # is the supported local backend and needs no server.
                tracking_uri=LOCAL_TRACKING_URI if args.local_mlflow else None,
                register=args.register,
                **split_kwargs,
            )
        results.append(result)

    table = compare(results)

    print()
    print("Walk-forward out-of-sample comparison")
    print("=" * 78)
    print(table.to_string(index=False))
    print()

    for result in results:
        print(f"  {result.summary()}")
    print()

    if not any(r.overall.beats_baseline for r in results):
        print(
            "No model beat the zero-prediction baseline. That is a normal result for\n"
            "daily equity returns and is reported rather than hidden: the pipeline is\n"
            "the product, the model is the workload."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
