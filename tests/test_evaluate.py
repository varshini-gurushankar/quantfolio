"""Evaluation metrics, especially the zero-prediction baseline.

The baseline is the whole point: "12% lower MSE" is meaningless without a stated
referent, and these tests pin down what the referent is and how it behaves.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantfolio.models.evaluate import (
    Metrics,
    aggregate,
    directional_accuracy,
    evaluate,
    information_coefficient,
    zero_prediction_mse,
)


@pytest.fixture
def returns() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.normal(0.0, 0.012, 500)


def test_zero_prediction_mse_is_the_second_moment(returns: np.ndarray) -> None:
    assert zero_prediction_mse(returns) == pytest.approx(np.mean(returns**2))


def test_predicting_zero_exactly_matches_the_baseline(returns: np.ndarray) -> None:
    metrics = evaluate(returns, np.zeros_like(returns))
    assert metrics.mse == pytest.approx(metrics.baseline_mse)
    assert metrics.improvement_over_baseline == pytest.approx(0.0)
    assert not metrics.beats_baseline


def test_a_perfect_model_beats_the_baseline(returns: np.ndarray) -> None:
    metrics = evaluate(returns, returns)
    assert metrics.mse == pytest.approx(0.0, abs=1e-18)
    assert metrics.beats_baseline
    assert metrics.improvement_over_baseline == pytest.approx(1.0)


def test_a_noisy_model_loses_to_the_baseline(returns: np.ndarray) -> None:
    """The realistic failure mode, and it must be reported as a loss."""
    rng = np.random.default_rng(9)
    noisy = returns + rng.normal(0.0, 0.05, len(returns))

    metrics = evaluate(returns, noisy)
    assert not metrics.beats_baseline
    assert metrics.improvement_over_baseline < 0
    assert "LOSES TO" in metrics.summary()


def test_improvement_is_signed_correctly(returns: np.ndarray) -> None:
    better = evaluate(returns, returns * 0.5)
    assert better.improvement_over_baseline > 0
    assert "beats" in better.summary()


def test_directional_accuracy_on_a_perfect_sign_caller() -> None:
    y_true = np.array([0.01, -0.02, 0.03, -0.01])
    y_pred = np.array([0.005, -0.001, 0.02, -0.004])
    assert directional_accuracy(y_true, y_pred) == pytest.approx(1.0)


def test_directional_accuracy_on_a_perfectly_wrong_caller() -> None:
    y_true = np.array([0.01, -0.02, 0.03])
    y_pred = np.array([-0.01, 0.02, -0.03])
    assert directional_accuracy(y_true, y_pred) == pytest.approx(0.0)


def test_all_zero_predictions_give_undefined_direction() -> None:
    """A model predicting nothing has no directional skill to credit."""
    y_true = np.array([0.01, -0.02, 0.03])
    assert np.isnan(directional_accuracy(y_true, np.zeros(3)))


def test_information_coefficient_is_a_correlation(returns: np.ndarray) -> None:
    assert information_coefficient(returns, returns) == pytest.approx(1.0)
    assert information_coefficient(returns, -returns) == pytest.approx(-1.0)


def test_information_coefficient_is_nan_for_a_constant_prediction(returns: np.ndarray) -> None:
    assert np.isnan(information_coefficient(returns, np.full_like(returns, 0.001)))


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate(np.zeros(10), np.zeros(9))


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        evaluate(np.array([]), np.array([]))


def test_aggregate_weights_folds_by_size() -> None:
    """A short final fold must not count as much as a long one."""
    big = Metrics(1.0, 1.0, 1.0, 2.0, 0.5, 0.5, 0.1, n_samples=900)
    small = Metrics(3.0, 1.7, 1.7, 2.0, -0.5, 0.5, 0.1, n_samples=100)

    combined = aggregate([big, small])

    assert combined.n_samples == 1000
    assert combined.mse == pytest.approx(0.9 * 1.0 + 0.1 * 3.0)
    assert combined.mse < 2.0, "the large fold should dominate"


def test_aggregate_ignores_nan_metrics() -> None:
    a = Metrics(1.0, 1.0, 1.0, 2.0, 0.5, float("nan"), 0.1, n_samples=100)
    b = Metrics(1.0, 1.0, 1.0, 2.0, 0.5, 0.6, 0.2, n_samples=100)

    combined = aggregate([a, b])
    assert combined.directional_accuracy == pytest.approx(0.6)


def test_aggregate_rejects_no_folds() -> None:
    with pytest.raises(ValueError, match="no folds"):
        aggregate([])
