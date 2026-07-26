"""The walk-forward training harness.

Most tests here use a trivial deterministic model rather than a real network:
the questions being asked are about the harness (fold independence, target
scaling, prediction alignment), and a linear model makes the answers exact
instead of approximate. Two slower tests exercise the real Keras and PyTorch
models to confirm the interface holds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from helpers import RecordingModel

from quantfolio.models.datasets import build_supervised
from quantfolio.models.train import compare, rolling_oos_mse, walk_forward_train
from quantfolio.transforms.features import compute_features_by_ticker


@pytest.fixture
def dataset(multi_ticker_frame: pd.DataFrame) -> pd.DataFrame:
    return build_supervised(compute_features_by_ticker(multi_ticker_frame))


# --------------------------------------------------------------------------- #
# fold mechanics
# --------------------------------------------------------------------------- #
def test_every_fold_gets_a_fresh_model(dataset: pd.DataFrame) -> None:
    """Carrying weights across folds warm-starts later folds on their own past."""
    model = RecordingModel()
    result = walk_forward_train(dataset, model, n_splits=3, test_size=40, min_train_size=100)

    assert model.reset_calls == len(result.folds)


def test_training_sets_grow_with_an_expanding_window(dataset: pd.DataFrame) -> None:
    model = RecordingModel()
    walk_forward_train(dataset, model, n_splits=3, test_size=40, min_train_size=100)

    sizes = [call["n"] for call in model.fit_calls]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) > 1


def test_targets_are_standardized_before_fitting(dataset: pd.DataFrame) -> None:
    """Daily returns are ~1e-2; unscaled they waste the training budget."""
    model = RecordingModel()
    walk_forward_train(dataset, model, n_splits=2, test_size=40, min_train_size=100)

    for call in model.fit_calls:
        assert call["y_mean"] == pytest.approx(0.0, abs=1e-9)
        assert call["y_std"] == pytest.approx(1.0, abs=1e-9)


def test_predictions_are_returned_in_raw_return_units(dataset: pd.DataFrame) -> None:
    """Scaling is inverted before scoring, so reported MSE is in return units."""
    result = walk_forward_train(
        dataset, RecordingModel(), n_splits=2, test_size=40, min_train_size=100
    )
    predictions = result.predictions()

    assert predictions["y_pred"].abs().max() < 1.0, "predictions look unscaled"
    assert result.overall.mse < 1e-2


def test_each_fold_is_evaluated_on_its_own_window(dataset: pd.DataFrame) -> None:
    result = walk_forward_train(
        dataset, RecordingModel(), n_splits=3, test_size=40, min_train_size=100
    )

    for fold in result.folds:
        dates = pd.to_datetime(fold.predictions["date"])
        assert dates.min().date() >= fold.split.test_start
        assert dates.max().date() <= fold.split.test_end


def test_folds_do_not_predict_the_same_day_twice(dataset: pd.DataFrame) -> None:
    result = walk_forward_train(
        dataset, RecordingModel(), n_splits=3, test_size=40, min_train_size=100
    )
    predictions = result.predictions()
    assert not predictions.duplicated(subset=["ticker", "date"]).any()


def test_predictions_carry_ticker_and_date(dataset: pd.DataFrame) -> None:
    """Without metadata a prediction cannot be joined back to what happened."""
    result = walk_forward_train(
        dataset, RecordingModel(), n_splits=2, test_size=40, min_train_size=100
    )
    predictions = result.predictions()

    assert {"ticker", "date", "y_true", "y_pred", "fold"}.issubset(predictions.columns)
    assert predictions["ticker"].notna().all()


def test_empty_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty dataset"):
        walk_forward_train(pd.DataFrame(), RecordingModel())


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def test_aggregate_metrics_include_the_baseline(dataset: pd.DataFrame) -> None:
    result = walk_forward_train(
        dataset, RecordingModel(), n_splits=2, test_size=40, min_train_size=100
    )
    assert result.overall.baseline_mse > 0
    assert "baseline" in result.summary()


def test_comparison_table_ranks_by_out_of_sample_mse(dataset: pd.DataFrame) -> None:
    a = walk_forward_train(dataset, RecordingModel(), n_splits=2, test_size=40, min_train_size=100)
    b = walk_forward_train(dataset, RecordingModel(), n_splits=2, test_size=40, min_train_size=100)
    b.model_name = "other"

    table = compare([a, b])

    assert list(table["oos_mse"]) == sorted(table["oos_mse"])
    assert "vs_baseline" in table.columns
    assert "vs_best_other" in table.columns


def test_rolling_oos_mse_is_a_dated_series(dataset: pd.DataFrame) -> None:
    result = walk_forward_train(
        dataset, RecordingModel(), n_splits=2, test_size=40, min_train_size=100
    )
    rolling = rolling_oos_mse(result.predictions(), window=10)

    assert isinstance(rolling, pd.Series)
    assert rolling.dropna().ge(0).all()


def test_rolling_oos_mse_of_empty_input_is_empty() -> None:
    assert rolling_oos_mse(pd.DataFrame()).empty


# --------------------------------------------------------------------------- #
# the real models
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_keras_baseline_trains_and_predicts(dataset: pd.DataFrame) -> None:
    from quantfolio.models.keras_mlp import KerasMLP

    model = KerasMLP(hidden_units=(8,), epochs=2, patience=1)
    result = walk_forward_train(dataset, model, n_splits=2, test_size=40, min_train_size=100)

    assert len(result.folds) == 2
    assert np.isfinite(result.overall.mse)
    # A converged model should be within an order of magnitude of the baseline;
    # far outside that means the output scale is wrong, not that signal is absent.
    assert result.overall.mse < result.overall.baseline_mse * 10


@pytest.mark.slow
def test_torch_challenger_trains_and_predicts(dataset: pd.DataFrame) -> None:
    from quantfolio.models.torch_lstm import TorchLSTM

    model = TorchLSTM(lookback=5, hidden_size=8, epochs=2, patience=1)
    result = walk_forward_train(dataset, model, n_splits=2, test_size=40, min_train_size=100)

    assert len(result.folds) == 2
    assert np.isfinite(result.overall.mse)
    assert result.overall.mse < result.overall.baseline_mse * 10


@pytest.mark.slow
def test_both_frameworks_are_scored_on_the_same_folds(dataset: pd.DataFrame) -> None:
    """The comparison is only meaningful if the folds are identical."""
    from quantfolio.models.keras_mlp import KerasMLP
    from quantfolio.models.torch_lstm import TorchLSTM

    keras = walk_forward_train(
        dataset,
        KerasMLP(hidden_units=(8,), epochs=2, patience=1),
        n_splits=2,
        test_size=40,
        min_train_size=100,
    )
    torch = walk_forward_train(
        dataset,
        TorchLSTM(lookback=5, hidden_size=8, epochs=2, patience=1),
        n_splits=2,
        test_size=40,
        min_train_size=100,
    )

    keras_windows = [(f.split.test_start, f.split.test_end) for f in keras.folds]
    torch_windows = [(f.split.test_start, f.split.test_end) for f in torch.folds]
    assert keras_windows == torch_windows
