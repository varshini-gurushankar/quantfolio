"""Model serving.

The interesting failure here is serving skew: a model loaded with the wrong
feature order or the wrong scaler still returns a plausible-looking float. These
tests pin the bundle contract so that cannot happen quietly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from helpers import RecordingModel

from quantfolio.models.datasets import MODEL_FEATURES, build_supervised
from quantfolio.models.train import Preprocessing, _bundle_metadata, walk_forward_train
from quantfolio.serving import predictor
from quantfolio.serving.predictor import (
    LoadedModel,
    ModelNotAvailable,
    _prepare_input,
    clear_cache,
)
from quantfolio.transforms.features import compute_features_by_ticker


@pytest.fixture(autouse=True)
def _clear_model_cache():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def dataset(multi_ticker_frame: pd.DataFrame) -> pd.DataFrame:
    return build_supervised(compute_features_by_ticker(multi_ticker_frame))


@pytest.fixture
def features(multi_ticker_frame: pd.DataFrame) -> pd.DataFrame:
    panel = compute_features_by_ticker(multi_ticker_frame)
    return panel[panel["ticker"] == "TEST"].copy()


@pytest.fixture
def preprocessing() -> Preprocessing:
    return Preprocessing(
        feature_columns=list(MODEL_FEATURES),
        feature_mean=[0.0] * len(MODEL_FEATURES),
        feature_std=[1.0] * len(MODEL_FEATURES),
        target_mean=0.0,
        target_std=0.01,
    )


def _loaded(preprocessing: Preprocessing, lookback: int = 1, predict=None) -> LoadedModel:
    return LoadedModel(
        name="stub",
        framework="test",
        version="1",
        preprocessing=preprocessing,
        params={"lookback": lookback},
        predict_fn=predict or (lambda x: np.zeros(len(x))),
        metadata={"beats_baseline": False},
    )


# --------------------------------------------------------------------------- #
# the bundle contract
# --------------------------------------------------------------------------- #
def test_training_captures_the_preprocessing(dataset: pd.DataFrame) -> None:
    """Weights without a scaler are not a servable model."""
    result = walk_forward_train(
        dataset, RecordingModel(), n_splits=2, test_size=40, min_train_size=100
    )

    assert result.preprocessing is not None
    assert result.preprocessing.feature_columns == list(MODEL_FEATURES)
    assert len(result.preprocessing.feature_mean) == len(MODEL_FEATURES)
    assert result.preprocessing.target_std > 0


def test_preprocessing_round_trips_through_json(preprocessing: Preprocessing) -> None:
    restored = Preprocessing.from_dict(json.loads(json.dumps(preprocessing.as_dict())))
    assert restored == preprocessing


def test_bundle_metadata_is_complete(dataset: pd.DataFrame) -> None:
    result = walk_forward_train(
        dataset, RecordingModel(), n_splits=2, test_size=40, min_train_size=100
    )
    metadata = _bundle_metadata(result, "recording.pt")

    for key in ("model_name", "framework", "weights_file", "model_params", "preprocessing"):
        assert metadata[key] is not None, f"{key} missing from the bundle manifest"
    assert metadata["trained_through"] is not None
    # JSON-serializable, since it is written to disk as-is.
    json.dumps(metadata)


def test_preprocessing_matches_the_final_fold(dataset: pd.DataFrame) -> None:
    """The deployed model is the last fold's, so the scaler must be too."""
    from quantfolio.models.datasets import apply_split, fit_scaler, walk_forward_splits

    result = walk_forward_train(
        dataset, RecordingModel(), n_splits=3, test_size=40, min_train_size=100
    )
    splits = walk_forward_splits(dataset["date"], n_splits=3, test_size=40, min_train_size=100)
    train, _ = apply_split(dataset, splits[-1])
    expected_mean, _ = fit_scaler(train, list(MODEL_FEATURES))

    np.testing.assert_allclose(result.preprocessing.feature_mean, expected_mean, rtol=1e-9)


# --------------------------------------------------------------------------- #
# input preparation
# --------------------------------------------------------------------------- #
def test_dense_input_is_a_single_scaled_row(features, preprocessing) -> None:
    x = _prepare_input(features, _loaded(preprocessing, lookback=1))
    assert x.shape == (1, len(MODEL_FEATURES))
    assert np.isfinite(x).all()


def test_sequential_input_has_the_right_window(features, preprocessing) -> None:
    x = _prepare_input(features, _loaded(preprocessing, lookback=20))
    assert x.shape == (1, 20, len(MODEL_FEATURES))


def test_prediction_uses_the_most_recent_rows(features, preprocessing) -> None:
    """A stale window would predict from last month and never say so."""
    model = _loaded(preprocessing, lookback=1)
    full = _prepare_input(features, model)
    truncated = _prepare_input(features.iloc[:-5], model)

    assert not np.allclose(full, truncated), "input did not move when new rows arrived"


def test_feature_order_follows_the_bundle_not_the_dataframe(features, preprocessing) -> None:
    """Column order is part of the contract; a reordered frame must not shift it."""
    reordered = features[list(reversed(features.columns))]

    expected = _prepare_input(features, _loaded(preprocessing))
    actual = _prepare_input(reordered, _loaded(preprocessing))

    np.testing.assert_allclose(expected, actual)


def test_scaler_from_the_bundle_is_applied(features) -> None:
    shifted = Preprocessing(
        feature_columns=list(MODEL_FEATURES),
        feature_mean=[5.0] * len(MODEL_FEATURES),
        feature_std=[2.0] * len(MODEL_FEATURES),
        target_mean=0.0,
        target_std=0.01,
    )
    unit = Preprocessing(
        feature_columns=list(MODEL_FEATURES),
        feature_mean=[0.0] * len(MODEL_FEATURES),
        feature_std=[1.0] * len(MODEL_FEATURES),
        target_mean=0.0,
        target_std=0.01,
    )

    raw = _prepare_input(features, _loaded(unit))
    scaled = _prepare_input(features, _loaded(shifted))

    np.testing.assert_allclose(scaled, (raw - 5.0) / 2.0, rtol=1e-5)


def test_too_little_history_for_a_sequence_is_reported(features, preprocessing) -> None:
    with pytest.raises(ModelNotAvailable, match="needs 20 sessions"):
        _prepare_input(features.tail(25).head(10), _loaded(preprocessing, lookback=20))


def test_all_nan_features_are_reported(preprocessing) -> None:
    empty = pd.DataFrame(
        {
            "ticker": "X",
            "date": pd.bdate_range("2023-01-02", periods=5),
            "adj_close": [100.0] * 5,
            **{c: [np.nan] * 5 for c in MODEL_FEATURES},
        }
    )
    with pytest.raises(ModelNotAvailable, match="no complete feature rows"):
        _prepare_input(empty, _loaded(preprocessing))


# --------------------------------------------------------------------------- #
# prediction output
# --------------------------------------------------------------------------- #
def test_prediction_is_unscaled_back_into_return_units(monkeypatch, features) -> None:
    """The model emits a z-score; the caller must receive a return."""
    prep = Preprocessing(
        feature_columns=list(MODEL_FEATURES),
        feature_mean=[0.0] * len(MODEL_FEATURES),
        feature_std=[1.0] * len(MODEL_FEATURES),
        target_mean=0.001,
        target_std=0.02,
    )
    model = _loaded(prep, lookback=1, predict=lambda x: np.array([2.0]))

    monkeypatch.setattr(predictor, "load_model", lambda **kw: model)
    monkeypatch.setattr(predictor, "read_features", lambda **kw: features, raising=False)
    monkeypatch.setattr("quantfolio.storage.db.read_features", lambda **kw: features)

    result = predictor.predict_ticker("TEST")

    # 2.0 z-scores above a mean of 0.001 with sigma 0.02 -> 0.041
    assert result["predicted_log_return"] == pytest.approx(2.0 * 0.02 + 0.001)
    assert result["predicted_simple_return"] == pytest.approx(np.expm1(0.041))


def test_prediction_reports_whether_the_model_beats_the_baseline(monkeypatch, features) -> None:
    """A prediction from a model worse than predicting zero is still a prediction."""
    prep = Preprocessing(
        feature_columns=list(MODEL_FEATURES),
        feature_mean=[0.0] * len(MODEL_FEATURES),
        feature_std=[1.0] * len(MODEL_FEATURES),
        target_mean=0.0,
        target_std=0.01,
    )
    model = _loaded(prep, predict=lambda x: np.array([0.5]))
    model.metadata["beats_baseline"] = True

    monkeypatch.setattr(predictor, "load_model", lambda **kw: model)
    monkeypatch.setattr("quantfolio.storage.db.read_features", lambda **kw: features)

    assert predictor.predict_ticker("TEST")["beats_baseline"] is True


def test_missing_features_are_reported_clearly(monkeypatch, preprocessing) -> None:
    monkeypatch.setattr(predictor, "load_model", lambda **kw: _loaded(preprocessing))
    monkeypatch.setattr("quantfolio.storage.db.read_features", lambda **kw: pd.DataFrame())

    with pytest.raises(ModelNotAvailable, match="no features stored"):
        predictor.predict_ticker("NOPE")


# --------------------------------------------------------------------------- #
# registry failures
# --------------------------------------------------------------------------- #
def test_no_registered_model_is_reported_not_crashed(monkeypatch) -> None:
    def no_versions(*_a, **_k):
        raise ModelNotAvailable("no versions registered under 'quantfolio_return_predictor'")

    monkeypatch.setattr(predictor, "_download_bundle", no_versions)
    with pytest.raises(ModelNotAvailable, match="no versions registered"):
        predictor.load_model()


def test_a_bundle_without_metadata_is_rejected(monkeypatch, tmp_path: Path) -> None:
    """An artifact from an older run cannot be served, and must say so."""
    monkeypatch.setattr(predictor, "_download_bundle", lambda v=None: (tmp_path, "3"))

    with pytest.raises(ModelNotAvailable, match="no metadata.json"):
        predictor.load_model()


def test_an_unknown_framework_is_rejected(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text(
        json.dumps({"framework": "jax", "model_name": "x", "weights_file": "w"})
    )
    monkeypatch.setattr(predictor, "_download_bundle", lambda v=None: (tmp_path, "1"))

    with pytest.raises(ModelNotAvailable, match="unsupported framework"):
        predictor.load_model()


def test_model_is_cached_across_calls(monkeypatch, tmp_path: Path, preprocessing) -> None:
    """Reloading per request would dominate the latency this phase measures."""
    calls = {"n": 0}

    metadata = {
        "framework": "tensorflow",
        "model_name": "cached",
        "weights_file": "w.keras",
        "model_params": {},
        "preprocessing": preprocessing.as_dict(),
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    def counting_download(v=None):
        calls["n"] += 1
        return tmp_path, "7"

    monkeypatch.setattr(predictor, "_download_bundle", counting_download)
    monkeypatch.setattr(predictor, "_rebuild_keras", lambda b, m: lambda x: np.zeros(len(x)))

    predictor.load_model()
    predictor.load_model()
    predictor.load_model()

    assert calls["n"] == 1

    predictor.clear_cache()
    predictor.load_model()
    assert calls["n"] == 2
