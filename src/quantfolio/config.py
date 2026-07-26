"""Runtime configuration: environment variables plus the universe YAML.

Two sources, deliberately separated:

* ``Settings`` — infrastructure (credentials, endpoints, URLs). Environment only,
  so the same image runs against LocalStack or real AWS with no code change.
* ``Universe`` — research parameters (tickers, feature windows). File only, so a
  change to the universe or a feature window is a reviewable config diff rather
  than a code edit, and so tests and the DAG read identical values.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNIVERSE_PATH = PROJECT_ROOT / "config" / "universe.yml"


class Settings(BaseSettings):
    """Infrastructure settings, read from the environment."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_user: str = "quantfolio"
    postgres_password: str = "quantfolio"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "quantfolio"

    aws_access_key_id: str = "test"
    aws_secret_access_key: str = "test"
    aws_default_region: str = "us-east-1"
    # Empty string means "use real AWS"; LocalStack sets this to its edge port.
    s3_endpoint_url: str = "http://localhost:4566"
    s3_raw_bucket: str = "quantfolio-raw"
    s3_artifacts_bucket: str = "quantfolio-artifacts"

    pushgateway_url: str = "http://localhost:9091"
    mlflow_tracking_uri: str = "http://localhost:5000"

    alpha_vantage_api_key: str = ""

    universe_path: Path = DEFAULT_UNIVERSE_PATH

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@dataclass(frozen=True)
class MacdParams:
    fast: int = 12
    slow: int = 26
    signal: int = 9


@dataclass(frozen=True)
class FeatureParams:
    sma_windows: tuple[int, ...] = (20, 60)
    ema_span: int = 20
    macd: MacdParams = field(default_factory=MacdParams)
    rsi_window: int = 14
    volatility_window: int = 20


@dataclass(frozen=True)
class CleaningParams:
    mad_window: int = 63
    mad_scale: float = 8.0


@dataclass(frozen=True)
class Universe:
    tickers: tuple[str, ...]
    benchmark: str
    exchange: str
    backfill_start: str
    features: FeatureParams
    cleaning: CleaningParams

    @classmethod
    def from_yaml(cls, path: Path | str) -> Universe:
        raw = yaml.safe_load(Path(path).read_text())
        feats = raw.get("features", {})
        clean = raw.get("cleaning", {})
        return cls(
            tickers=tuple(raw["tickers"]),
            benchmark=raw.get("benchmark", "SPY"),
            exchange=raw.get("exchange", "NYSE"),
            backfill_start=str(raw.get("backfill_start", "2015-01-01")),
            features=FeatureParams(
                sma_windows=tuple(feats.get("sma_windows", (20, 60))),
                ema_span=int(feats.get("ema_span", 20)),
                macd=MacdParams(**feats.get("macd", {})),
                rsi_window=int(feats.get("rsi_window", 14)),
                volatility_window=int(feats.get("volatility_window", 20)),
            ),
            cleaning=CleaningParams(
                mad_window=int(clean.get("mad_window", 63)),
                mad_scale=float(clean.get("mad_scale", 8.0)),
            ),
        )


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@functools.lru_cache(maxsize=1)
def get_universe() -> Universe:
    return Universe.from_yaml(get_settings().universe_path)
