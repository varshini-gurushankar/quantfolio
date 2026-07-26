"""PyTorch LSTM — the sequential challenger.

The reason there are two frameworks in this project is not that two look better
than one on a résumé. It is that the two models answer different questions, and
each framework is the natural fit for its side of the comparison.

The MLP sees one day at a time. This model sees a trailing window of sessions
and can, in principle, learn patterns that only exist in the sequence —
momentum, volatility clustering, mean reversion after a run. PyTorch's explicit
training loop makes the sequence handling and the early-stopping logic visible
rather than hidden behind a `.fit()` call, which matters when the question is
*why* the sequence model did or did not win.

Keras is used for the baseline because a dense MLP is exactly the case its
declarative API handles with the least ceremony.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import pandas as pd

from quantfolio.models.base import Model
from quantfolio.models.datasets import make_sequences

logger = logging.getLogger(__name__)


class TorchLSTM(Model):
    name = "torch_lstm"
    framework = "pytorch"

    def __init__(
        self,
        lookback: int = 20,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 256,
        patience: int = 8,
        seed: int = 42,
    ) -> None:
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self._model = None
        self._device = None

    def params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "framework": self.framework,
            "lookback": self.lookback,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
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
        return make_sequences(frame, feature_columns, mean, std, lookback=self.lookback)

    def _build(self, n_features: int):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)

        class LSTMRegressor(nn.Module):
            def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    # PyTorch only applies dropout *between* stacked layers, so
                    # it is a no-op (and warns) when num_layers == 1.
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                # Only the final timestep matters: it is the one whose next-day
                # return is being predicted.
                return self.head(self.dropout(out[:, -1, :])).squeeze(-1)

        return LSTMRegressor(n_features, self.hidden_size, self.num_layers, self.dropout)

    def fit(self, x: np.ndarray, y: np.ndarray, validation_split: float = 0.1) -> dict[str, Any]:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        self._device = torch.device("cpu")
        self._model = self._build(x.shape[2]).to(self._device)

        # Chronological validation split — the tail of the training fold, never
        # a random sample.
        n_val = max(1, int(len(x) * validation_split))
        x_train, y_train = x[:-n_val], y[:-n_val]
        x_val, y_val = x[-n_val:], y[-n_val:]

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()),
            batch_size=self.batch_size,
            shuffle=False,
        )
        x_val_t = torch.from_numpy(x_val).float().to(self._device)
        y_val_t = torch.from_numpy(y_val).float().to(self._device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        epochs_without_improvement = 0
        epochs_run = 0
        train_loss = float("nan")

        for epoch in range(self.epochs):
            epochs_run = epoch + 1
            self._model.train()
            batch_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")

            self._model.eval()
            with torch.no_grad():
                val_loss = float(criterion(self._model(x_val_t), y_val_t).item())

            if val_loss < best_val - 1e-9:
                best_val = val_loss
                # Deep-copied so later epochs cannot mutate the saved weights.
                best_state = copy.deepcopy(self._model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    logger.info("early stop at epoch %d (best val %.6e)", epochs_run, best_val)
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        return {
            "epochs_run": epochs_run,
            "final_train_loss": train_loss,
            "final_val_loss": best_val,
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("model has not been fitted")

        self._model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(x).float().to(self._device)
            return self._model(tensor).cpu().numpy().ravel()

    def save(self, path: str) -> str:
        import torch

        if self._model is None:
            raise RuntimeError("model has not been fitted")
        target = path if path.endswith(".pt") else f"{path}.pt"
        torch.save(self._model.state_dict(), target)
        return target

    def reset(self) -> None:
        self._model = None
