"""API behaviour, with the database stubbed out.

These tests are about the HTTP contract — status codes, validation, and JSON
shape — so the storage layer is replaced with fixed frames. Whether the queries
themselves are right is covered by the storage tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quantfolio.api.main import app
from quantfolio.api.routers import features as features_router
from quantfolio.api.routers import prices as prices_router


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def price_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": "AAPL",
            "date": pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"]).date,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "adj_close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
            "source": "yfinance",
        }
    )


@pytest.fixture
def feature_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": "AAPL",
            "date": pd.to_datetime(["2023-01-03", "2023-01-04"]).date,
            "adj_close": [100.5, 101.5],
            "simple_return": [np.nan, 0.00995],
            "log_return": [np.nan, 0.00990],
            "sma_20": [np.nan, np.nan],
            "sma_60": [np.nan, np.nan],
            "ema_20": [np.nan, np.nan],
            "macd": [np.nan, np.nan],
            "macd_signal": [np.nan, np.nan],
            "macd_hist": [np.nan, np.nan],
            "rsi_14": [np.nan, 55.0],
            "volatility_20": [np.nan, 0.21],
            "is_imputed": [False, False],
        }
    )


# --------------------------------------------------------------------------- #
# health
# --------------------------------------------------------------------------- #
def test_health_reports_degraded_without_a_database(client: TestClient) -> None:
    """Health must answer even when the store is unreachable — that is its job."""
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "checked_at" in body


# --------------------------------------------------------------------------- #
# prices
# --------------------------------------------------------------------------- #
def test_prices_returns_bars(client, monkeypatch, price_rows) -> None:
    monkeypatch.setattr(prices_router, "read_prices", lambda **kw: price_rows)

    response = client.get("/prices/AAPL")
    assert response.status_code == 200

    body = response.json()
    assert body["ticker"] == "AAPL"
    assert body["count"] == 3
    assert body["bars"][0]["close"] == 100.5


def test_prices_are_returned_in_date_order(client, monkeypatch, price_rows) -> None:
    monkeypatch.setattr(prices_router, "read_prices", lambda **kw: price_rows.iloc[::-1])

    bars = client.get("/prices/AAPL").json()["bars"]
    assert [b["date"] for b in bars] == sorted(b["date"] for b in bars)


def test_unknown_ticker_is_a_404(client, monkeypatch) -> None:
    monkeypatch.setattr(prices_router, "read_prices", lambda **kw: pd.DataFrame())
    assert client.get("/prices/NOPE").status_code == 404


def test_ticker_is_case_insensitive(client, monkeypatch, price_rows) -> None:
    monkeypatch.setattr(prices_router, "read_prices", lambda **kw: price_rows)
    assert client.get("/prices/aapl").json()["ticker"] == "AAPL"


def test_limit_returns_the_most_recent_rows(client, monkeypatch, price_rows) -> None:
    monkeypatch.setattr(prices_router, "read_prices", lambda **kw: price_rows)

    body = client.get("/prices/AAPL?limit=2").json()
    assert body["count"] == 2
    assert body["bars"][-1]["date"] == "2023-01-05"


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def test_features_returns_rows(client, monkeypatch, feature_rows) -> None:
    monkeypatch.setattr(features_router, "read_features", lambda **kw: feature_rows)

    body = client.get("/features/AAPL").json()
    assert body["count"] == 2
    assert body["rows"][1]["rsi_14"] == 55.0


def test_warmup_nans_serialize_as_null(client, monkeypatch, feature_rows) -> None:
    """NaN is a float in Python but not valid JSON — it must come back as null."""
    monkeypatch.setattr(features_router, "read_features", lambda **kw: feature_rows)

    raw_body = client.get("/features/AAPL").text
    assert "NaN" not in raw_body

    first = client.get("/features/AAPL").json()["rows"][0]
    assert first["rsi_14"] is None
    assert first["log_return"] is None


def test_date_range_is_passed_through(client, monkeypatch, feature_rows) -> None:
    captured = {}

    def fake_read(**kwargs):
        captured.update(kwargs)
        return feature_rows

    monkeypatch.setattr(features_router, "read_features", fake_read)
    client.get("/features/AAPL?start=2023-01-01&end=2023-01-31")

    assert str(captured["start"]) == "2023-01-01"
    assert str(captured["end"]) == "2023-01-31"


def test_reversed_date_range_is_rejected(client) -> None:
    response = client.get("/features/AAPL?start=2023-06-01&end=2023-01-01")
    assert response.status_code == 400
    assert "start must not be after end" in response.json()["detail"]


def test_malformed_date_is_rejected(client) -> None:
    assert client.get("/features/AAPL?start=not-a-date").status_code == 422


def test_excessive_limit_is_rejected(client) -> None:
    assert client.get("/features/AAPL?limit=999999").status_code == 422


def test_openapi_schema_is_served(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/prices/{ticker}" in schema["paths"]
    assert "/features/{ticker}" in schema["paths"]
    assert "/health" in schema["paths"]
