"""Load test for the API.

The point is to replace a guessed latency figure with a measured one. Any
"sub-Xms" claim should come from this file's output, at a stated concurrency,
against a stated endpoint mix — a number without those three things attached is
not a measurement.

Headless run, writing a CSV summary:

    uv run locust -f scripts/locustfile.py --host http://localhost:8000 \
        --headless -u 50 -r 10 -t 60s --csv results/load

Interactive, with the web UI at http://localhost:8089:

    uv run locust -f scripts/locustfile.py --host http://localhost:8000

Or via make:

    make loadtest
"""

from __future__ import annotations

import random

from locust import HttpUser, between, events, task

# The real universe, so cache behaviour resembles production rather than
# hammering one hot key.
TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "JPM",
    "JNJ",
    "V",
    "PG",
    "XOM",
    "UNH",
    "SPY",
]


class ReadOnlyUser(HttpUser):
    """Weighted to reflect a realistic mix: cheap reads dominate, inference is rare."""

    wait_time = between(0.1, 0.5)

    @task(10)
    def health(self) -> None:
        self.client.get("/health", name="/health")

    @task(30)
    def features(self) -> None:
        ticker = random.choice(TICKERS)
        self.client.get(
            f"/features/{ticker}?limit=100",
            name="/features/{ticker}",
        )

    @task(20)
    def prices(self) -> None:
        ticker = random.choice(TICKERS)
        self.client.get(f"/prices/{ticker}?limit=100", name="/prices/{ticker}")

    @task(10)
    def portfolio(self) -> None:
        # 503 is a valid answer before the optimizer has ever run; counting it
        # as a failure would make the latency numbers unreadable.
        with self.client.get("/portfolio", name="/portfolio", catch_response=True) as response:
            if response.status_code in (200, 503):
                response.success()

    @task(5)
    def predict(self) -> None:
        ticker = random.choice(TICKERS)
        with self.client.get(
            f"/predict/{ticker}", name="/predict/{ticker}", catch_response=True
        ) as response:
            if response.status_code in (200, 503):
                response.success()


@events.test_stop.add_listener
def report(environment, **_kwargs) -> None:
    """Print the percentiles worth quoting, per endpoint and overall."""
    stats = environment.stats

    print()
    print("=" * 78)
    print("Measured latency (ms)")
    print("=" * 78)
    print(f"{'endpoint':<28}{'reqs':>8}{'fail':>7}{'p50':>9}{'p95':>9}{'p99':>9}")

    for entry in sorted(stats.entries.values(), key=lambda e: e.name):
        # entry.name is the route template the task registered, which is what
        # makes these rows readable; the dict key ordering is not.
        print(
            f"{entry.name:<28}{entry.num_requests:>8}{entry.num_failures:>7}"
            f"{entry.get_response_time_percentile(0.50) or 0:>9.0f}"
            f"{entry.get_response_time_percentile(0.95) or 0:>9.0f}"
            f"{entry.get_response_time_percentile(0.99) or 0:>9.0f}"
        )

    total = stats.total
    print("-" * 78)
    print(
        f"{'TOTAL':<28}{total.num_requests:>8}{total.num_failures:>7}"
        f"{total.get_response_time_percentile(0.50) or 0:>9.0f}"
        f"{total.get_response_time_percentile(0.95) or 0:>9.0f}"
        f"{total.get_response_time_percentile(0.99) or 0:>9.0f}"
    )
    # user_count is already zero by the time this fires, so report the target
    # the run was configured with — the number that belongs beside a percentile.
    options = getattr(environment, "parsed_options", None)
    users = getattr(options, "num_users", None) or "n/a"

    print()
    print(f"RPS: {total.total_rps:.1f} | target users: {users}")
    print("Quote these numbers with the concurrency and endpoint mix that produced them.")
    print()
