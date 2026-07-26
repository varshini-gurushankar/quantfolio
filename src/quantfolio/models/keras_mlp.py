"""TensorFlow/Keras dense MLP — the named baseline.

This is the model every other result is quoted against. It is deliberately
simple: a small feed-forward network over the current day's features, with no
memory of previous days. That makes it the right control for the LSTM, whose
entire claim is that the *sequence* carries information a single snapshot does
not. If the LSTM cannot beat this, the sequence claim is not supported.

Small and regularised on purpose. Daily return prediction has a signal-to-noise
ratio low enough that a large network memorises noise, and the walk-forward
folds would show it immediately.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from quantfolio.models.base import Model
from quantfolio.models.datasets import TARGET, transform

logger = logging.getLogger(__name__)


class KerasMLP(Model):
    name = "keras_mlp"
    framework = "tensorflow"

    def __init__(
        self,
        hidden_units: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 256,
        patience: int = 8,
        seed: int = 42,
    ) -> None:
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self._model = None
        self._n_features: int | None = None

    def params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "framework": self.framework,
            "hidden_units": "-".join(str(u) for u in self.hidden_units),
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "seed": self.seed,
        }

    def prepare(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        mean: np.ndarray,
        std: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        x = transform(frame, feature_columns, mean, std).astype(np.float32)
        y = frame[TARGET].to_numpy(dtype=np.float32)
        return x, y, frame[["ticker", "date"]].reset_index(drop=True)

    def _build(self, n_features: int):
        import tensorflow as tf
        from tensorflow import keras

        tf.keras.utils.set_random_seed(self.seed)

        layers: list = [keras.layers.Input(shape=(n_features,))]
        for units in self.hidden_units:
            layers.append(keras.layers.Dense(units, activation="relu"))
            layers.append(keras.layers.Dropout(self.dropout))
        layers.append(keras.layers.Dense(1, activation="linear"))

        model = keras.Sequential(layers, name=self.name)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        return model

    def fit(self, x: np.ndarray, y: np.ndarray, validation_split: float = 0.1) -> dict[str, Any]:
        from tensorflow import keras

        self._n_features = x.shape[1]
        self._model = self._build(self._n_features)

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
        )

        history = self._model.fit(
            x,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            # The validation slice is the *tail* of the training fold, not a
            # random sample: shuffling would let the model validate against days
            # interleaved with its own training data.
            validation_split=validation_split,
            shuffle=False,
            callbacks=[early_stop],
            verbose=0,
        )

        return {
            "epochs_run": len(history.history["loss"]),
            "final_train_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history.get("val_loss", [float("nan")])[-1]),
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model has not been fitted")
        return self._model.predict(x, verbose=0).ravel()

    def save(self, path: str) -> str:
        if self._model is None:
            raise RuntimeError("model has not been fitted")
        target = path if path.endswith(".keras") else f"{path}.keras"
        self._model.save(target)
        return target

    def reset(self) -> None:
        from tensorflow import keras

        self._model = None
        keras.backend.clear_session()
