"""Request instrumentation.

Unlike the Airflow tasks, the API is long-lived, so Prometheus can scrape it
directly — no Pushgateway involved. That difference is the whole reason the
pipeline pushes and the service exposes.

Latency is a **histogram**, not a gauge or a counter of averages. An average
latency hides exactly the thing worth knowing: a p99 of 2s and a p50 of 20ms
average out to something that looks fine and serves a bad experience to one
request in a hundred. Buckets are set for a service whose slow path is a model
forward pass, so they cluster where the interesting answers live.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)

REQUEST_LATENCY = Histogram(
    "quantfolio_request_duration_seconds",
    "Request latency by endpoint",
    labelnames=("method", "endpoint", "status"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

REQUEST_COUNT = Counter(
    "quantfolio_requests_total",
    "Total requests by endpoint and outcome",
    labelnames=("method", "endpoint", "status"),
)

PREDICTION_LATENCY = Histogram(
    "quantfolio_prediction_duration_seconds",
    "Model inference latency, excluding HTTP overhead",
    labelnames=("model",),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

MODEL_INFO = Gauge(
    "quantfolio_model_loaded",
    "1 when a model is loaded and servable",
    labelnames=("model", "version", "framework"),
)


def _route_template(request: Request) -> str:
    """Group by route pattern, not by URL.

    ``/prices/AAPL`` and ``/prices/MSFT`` are the same endpoint; labelling by
    raw path would create one time series per ticker and blow up cardinality.
    """
    route = request.scope.get("route")
    return getattr(route, "path", None) or request.url.path


def install(app: FastAPI) -> None:
    """Attach the latency middleware and the /metrics endpoint."""

    @app.middleware("http")
    async def track(request: Request, call_next: Callable[[Request], Awaitable[Response]]):
        # /metrics measuring itself is noise, and it is scraped every 15s.
        if request.url.path == "/metrics":
            return await call_next(request)

        start = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            elapsed = time.perf_counter() - start
            labels = (request.method, _route_template(request), status)
            REQUEST_LATENCY.labels(*labels).observe(elapsed)
            REQUEST_COUNT.labels(*labels).inc()

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
