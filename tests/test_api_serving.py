"""The Phase 3 endpoints and request instrumentation.

Two behaviours get particular attention: an unavailable model must produce 503
rather than 500 (the service is healthy, there is simply nothing to serve), and
latency labels must be low-cardinality (per route, never per ticker).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quantfolio.api.main import app
from quantfolio.api.routers import portfolio as portfolio_router
from quantfolio.api.routers import predict as predict_router
from quantfolio.serving.predictor import ModelNotAvailable


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def prediction() -> dict:
    return {
        "ticker": "AAPL",
        "as_of": date(2024, 6, 28),
        "predicted_log_return": 0.0012,
        "predicted_simple_return": 0.0012007,
        "model_name": "torch_lstm",
        "model_version": "3",
        "framework": "pytorch",
        "beats_baseline": True,
    }


@pytest.fixture
def weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_date": [date(2024, 6, 28)] * 3,
            "ticker": ["AAPL", "MSFT", "SPY"],
            "weight": [0.5, 0.3, 0.2],
            "method": ["markowitz_ledoit_wolf"] * 3,
        }
    )


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #
def test_prediction_is_served(client, monkeypatch, prediction) -> None:
    monkeypatch.setattr(predict_router, "predict_ticker", lambda *a, **k: prediction)

    body = client.get("/predict/AAPL").json()

    assert body["ticker"] == "AAPL"
    assert body["predicted_log_return"] == pytest.approx(0.0012)
    assert body["model_version"] == "3"


def test_prediction_reports_its_own_latency(client, monkeypatch, prediction) -> None:
    monkeypatch.setattr(predict_router, "predict_ticker", lambda *a, **k: prediction)

    body = client.get("/predict/AAPL").json()
    assert body["inference_seconds"] >= 0


def test_prediction_states_whether_the_model_beats_the_baseline(
    client, monkeypatch, prediction
) -> None:
    """A caller must be able to tell a useful prediction from a decorative one."""
    prediction["beats_baseline"] = False
    monkeypatch.setattr(predict_router, "predict_ticker", lambda *a, **k: prediction)

    assert client.get("/predict/AAPL").json()["beats_baseline"] is False


def test_ticker_is_normalized(client, monkeypatch, prediction) -> None:
    seen = {}

    def capture(ticker, **kwargs):
        seen["ticker"] = ticker
        return prediction

    monkeypatch.setattr(predict_router, "predict_ticker", capture)
    client.get("/predict/aapl")

    assert seen["ticker"] == "AAPL"


def test_no_model_yields_503_not_500(client, monkeypatch) -> None:
    """The service is fine; there is just nothing registered yet."""

    def unavailable(*_a, **_k):
        raise ModelNotAvailable("no versions registered")

    monkeypatch.setattr(predict_router, "predict_ticker", unavailable)

    response = client.get("/predict/AAPL")
    assert response.status_code == 503
    assert "no versions registered" in response.json()["detail"]


def test_model_status_reports_absence_without_failing(client, monkeypatch) -> None:
    def unavailable(*_a, **_k):
        raise ModelNotAvailable("nothing registered")

    monkeypatch.setattr(predict_router, "load_model", unavailable)

    body = client.get("/predict/model/status").json()
    assert body["loaded"] is False
    assert "nothing registered" in body["detail"]


def test_model_status_reports_the_loaded_model(client, monkeypatch) -> None:
    class Stub:
        name, framework, version = "torch_lstm", "pytorch", "3"
        metadata = {
            "oos_mse": 1.1e-4,
            "baseline_mse": 1.2e-4,
            "beats_baseline": True,
            "trained_through": "2024-06-28",
        }

    monkeypatch.setattr(predict_router, "load_model", lambda *a, **k: Stub())

    body = client.get("/predict/model/status").json()
    assert body["loaded"] is True
    assert body["model_name"] == "torch_lstm"
    assert body["beats_baseline"] is True


# --------------------------------------------------------------------------- #
# /portfolio
# --------------------------------------------------------------------------- #
def test_portfolio_is_served(client, monkeypatch, weights) -> None:
    monkeypatch.setattr(portfolio_router, "latest_weights", lambda: weights)

    body = client.get("/portfolio").json()

    assert body["n_holdings"] == 3
    assert body["total_weight"] == pytest.approx(1.0)
    assert body["holdings"][0]["ticker"] == "AAPL"


def test_portfolio_reports_concentration(client, monkeypatch, weights) -> None:
    """Herfindahl index: 1/N when equally weighted, 1.0 for a single position."""
    monkeypatch.setattr(portfolio_router, "latest_weights", lambda: weights)

    body = client.get("/portfolio").json()
    assert body["concentration"] == pytest.approx(0.5**2 + 0.3**2 + 0.2**2)


def test_no_weights_yields_503(client, monkeypatch) -> None:
    monkeypatch.setattr(portfolio_router, "latest_weights", lambda: pd.DataFrame())

    response = client.get("/portfolio")
    assert response.status_code == 503
    assert "training pipeline" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# instrumentation
# --------------------------------------------------------------------------- #
def test_metrics_endpoint_exposes_the_histogram(client) -> None:
    client.get("/health")
    text = client.get("/metrics").text

    assert "quantfolio_request_duration_seconds_bucket" in text
    assert "quantfolio_requests_total" in text


def test_latency_is_a_histogram_not_an_average(client) -> None:
    """Percentiles need buckets; an average would hide the tail entirely."""
    client.get("/health")
    text = client.get("/metrics").text

    assert "quantfolio_request_duration_seconds_bucket{" in text
    assert "le=" in text


def test_metrics_are_labelled_by_route_not_by_url(client, monkeypatch, prediction) -> None:
    """Labelling by raw path would create one time series per ticker."""
    monkeypatch.setattr(predict_router, "predict_ticker", lambda *a, **k: prediction)

    client.get("/predict/AAPL")
    client.get("/predict/MSFT")
    text = client.get("/metrics").text

    assert 'endpoint="/predict/{ticker}"' in text
    assert 'endpoint="/predict/AAPL"' not in text


def test_failed_requests_are_recorded_with_their_status(client, monkeypatch) -> None:
    monkeypatch.setattr(portfolio_router, "latest_weights", lambda: pd.DataFrame())

    client.get("/portfolio")
    text = client.get("/metrics").text

    assert 'status="503"' in text


def test_metrics_endpoint_does_not_measure_itself(client) -> None:
    """Scraped every 15s; counting it would swamp the real traffic."""
    client.get("/metrics")
    text = client.get("/metrics").text

    assert 'endpoint="/metrics"' not in text


def test_new_endpoints_are_in_the_schema(client) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert "/predict/{ticker}" in paths
    assert "/portfolio" in paths
    assert "/predict/model/status" in paths


# --------------------------------------------------------------------------- #
# dependency failures
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("path", "module", "attribute"),
    [
        ("/portfolio", "portfolio", "latest_weights"),
        ("/predict/AAPL", "predict", "predict_ticker"),
    ],
)
def test_database_outage_is_503_not_500(client, monkeypatch, path, module, attribute) -> None:
    """A dependency being down is not a bug in this service.

    Returning 500 sends whoever is on call to read application logs looking for
    a crash that never happened; 503 names the real problem.
    """
    from sqlalchemy.exc import OperationalError

    import quantfolio.api.routers as routers

    target = getattr(routers, module)

    def refused(*_a, **_k):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    monkeypatch.setattr(target, attribute, refused)

    response = client.get(path)
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]


@pytest.mark.parametrize("path", ["/prices/AAPL", "/features/AAPL"])
def test_read_endpoints_also_degrade_to_503(client, monkeypatch, path) -> None:
    from sqlalchemy.exc import OperationalError

    from quantfolio.api.routers import features as features_router
    from quantfolio.api.routers import prices as prices_router

    def refused(**_k):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    monkeypatch.setattr(prices_router, "read_prices", refused)
    monkeypatch.setattr(features_router, "read_features", refused)

    assert client.get(path).status_code == 503
