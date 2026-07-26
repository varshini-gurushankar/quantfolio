"""Fetch orchestration: primary source, secondary fallback, per-ticker isolation."""

from __future__ import annotations

import logging
from datetime import date

from quantfolio.config import Settings, get_settings
from quantfolio.ingestion.alpha_vantage_client import AlphaVantageSource
from quantfolio.ingestion.base import FetchResult, PriceSource
from quantfolio.ingestion.yfinance_client import YFinanceSource

logger = logging.getLogger(__name__)


def build_sources(settings: Settings | None = None) -> list[PriceSource]:
    """Sources in priority order. Alpha Vantage is skipped when no key is set."""
    settings = settings or get_settings()
    sources: list[PriceSource] = [YFinanceSource()]
    if settings.alpha_vantage_api_key:
        sources.append(AlphaVantageSource(settings.alpha_vantage_api_key))
    else:
        logger.info("no ALPHA_VANTAGE_API_KEY set; running with yfinance only")
    return sources


def fetch_ticker(
    ticker: str,
    start: date,
    end: date,
    sources: list[PriceSource] | None = None,
) -> FetchResult:
    """Try each source in order; return the first non-empty result.

    If every source fails, the returned ``FetchResult`` carries the concatenated
    errors — the caller records it and moves on rather than failing the run.
    """
    sources = sources or build_sources()
    errors: list[str] = []

    for source in sources:
        result = source.fetch_safe(ticker, start, end)
        if result.ok:
            if errors:
                logger.warning(
                    "%s: fell back to %s after %s", ticker, source.name, "; ".join(errors)
                )
            return result
        errors.append(f"{source.name}: {result.error or 'no rows'}")

    return FetchResult(ticker=ticker, source="none", frame=None, error="; ".join(errors))


def fetch_universe(
    tickers: list[str],
    start: date,
    end: date,
    sources: list[PriceSource] | None = None,
) -> list[FetchResult]:
    """Fetch every ticker, isolating failures so one bad symbol cannot abort the run."""
    sources = sources or build_sources()
    results = [fetch_ticker(t, start, end, sources) for t in tickers]

    failed = [r.ticker for r in results if not r.ok]
    if failed:
        logger.warning("%d/%d tickers failed: %s", len(failed), len(results), ", ".join(failed))
    return results
