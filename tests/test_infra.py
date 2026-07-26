"""Infrastructure-as-code and dashboard configuration.

These files are deployed artifacts, not documentation, so they are checked like
code. A dashboard referencing a metric nothing emits is broken in exactly the
way nobody notices until an incident.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
TERRAFORM = ROOT / "terraform"
DASHBOARDS = ROOT / "monitoring/grafana/provisioning/dashboards"

# Either binary reads the same HCL; whichever is installed is fine.
TF_BINARY = shutil.which("terraform") or shutil.which("tofu")


# --------------------------------------------------------------------------- #
# compose
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text())


def test_every_service_the_readme_promises_exists(compose) -> None:
    expected = {
        "postgres",
        "localstack",
        "airflow-webserver",
        "airflow-scheduler",
        "api",
        "mlflow",
        "prometheus",
        "pushgateway",
        "grafana",
    }
    assert expected <= set(compose["services"])


def test_api_can_reach_the_model_registry(compose) -> None:
    """/predict pulls from MLflow, whose artifacts live in S3."""
    env = compose["services"]["api"]["environment"]
    assert "MLFLOW_TRACKING_URI" in env
    assert "MLFLOW_S3_ENDPOINT_URL" in env


def test_prometheus_scrapes_the_api_directly() -> None:
    """The API is long-lived, so it is scraped rather than pushing."""
    config = yaml.safe_load((ROOT / "monitoring/prometheus.yml").read_text())
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}

    assert "quantfolio_api" in jobs
    assert jobs["quantfolio_api"]["metrics_path"] == "/metrics"


def test_pushgateway_keeps_the_pushed_labels() -> None:
    """Without honor_labels Prometheus overwrites the job label the batch set."""
    config = yaml.safe_load((ROOT / "monitoring/prometheus.yml").read_text())
    gateway = next(j for j in config["scrape_configs"] if j["job_name"] == "pushgateway")

    assert gateway["honor_labels"] is True


# --------------------------------------------------------------------------- #
# dashboards
# --------------------------------------------------------------------------- #
def _dashboards() -> list[Path]:
    return sorted(DASHBOARDS.glob("*.json"))


def test_dashboards_are_valid_json() -> None:
    assert _dashboards(), "no dashboards found"
    for path in _dashboards():
        json.loads(path.read_text())


def test_dashboard_uids_are_unique() -> None:
    uids = [json.loads(p.read_text())["uid"] for p in _dashboards()]
    assert len(uids) == len(set(uids))


def test_dashboard_panels_reference_the_provisioned_datasource() -> None:
    """A dashboard pointing at a datasource uid that does not exist renders empty."""
    datasource = yaml.safe_load(
        (ROOT / "monitoring/grafana/provisioning/datasources/prometheus.yml").read_text()
    )
    known = {ds["uid"] for ds in datasource["datasources"]}

    for path in _dashboards():
        for panel in json.loads(path.read_text())["panels"]:
            if panel.get("type") == "row":
                continue
            uid = panel.get("datasource", {}).get("uid")
            assert uid in known, f"{path.name}/{panel['title']} points at unknown datasource {uid}"


def test_every_queried_metric_is_actually_emitted() -> None:
    """The check that catches a renamed metric before an incident does."""
    emitted = set()
    for module in ("metrics/push.py", "api/metrics.py"):
        source = (ROOT / "src/quantfolio" / module).read_text()
        for line in source.splitlines():
            if '"quantfolio_' in line:
                emitted.add(line.split('"quantfolio_')[1].split('"')[0])

    emitted = {f"quantfolio_{name}" for name in emitted}
    assert emitted, "no metric names found in the source"

    for path in _dashboards():
        content = path.read_text()
        for target in _iter_targets(json.loads(content)):
            for token in _metric_tokens(target):
                base = token.removesuffix("_bucket").removesuffix("_count").removesuffix("_sum")
                assert base in emitted, f"{path.name} queries {token}, which nothing emits"


def _iter_targets(dashboard: dict):
    for panel in dashboard["panels"]:
        for target in panel.get("targets", []):
            if "expr" in target:
                yield target["expr"]


def _metric_tokens(expr: str) -> set[str]:
    import re

    return set(re.findall(r"quantfolio_[a-z0-9_]+", expr))


# --------------------------------------------------------------------------- #
# terraform
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(TF_BINARY is None, reason="neither terraform nor tofu installed")
def test_terraform_is_formatted() -> None:
    result = subprocess.run(
        [TF_BINARY, "fmt", "-check", "-recursive"],
        cwd=TERRAFORM,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"unformatted files:\n{result.stdout}"


@pytest.mark.skipif(TF_BINARY is None, reason="neither terraform nor tofu installed")
@pytest.mark.slow
def test_terraform_configuration_is_valid() -> None:
    """Runs init offline, then validate — no cloud credentials involved."""
    init = subprocess.run(
        [TF_BINARY, "init", "-backend=false", "-input=false"],
        cwd=TERRAFORM,
        capture_output=True,
        text=True,
    )
    assert init.returncode == 0, init.stderr

    validate = subprocess.run(
        [TF_BINARY, "validate"], cwd=TERRAFORM, capture_output=True, text=True
    )
    assert validate.returncode == 0, validate.stdout + validate.stderr


def test_raw_bucket_is_versioned_in_the_config() -> None:
    """The replay guarantee is enforced by the bucket, not only by application code."""
    main = (TERRAFORM / "main.tf").read_text()
    assert 'resource "aws_s3_bucket_versioning" "raw"' in main
    assert 'status = "Enabled"' in main


def test_buckets_block_public_access() -> None:
    main = (TERRAFORM / "main.tf").read_text()
    assert main.count('resource "aws_s3_bucket_public_access_block"') == 2


def test_compute_resources_default_to_off() -> None:
    """LocalStack cannot really create these, so apply must not claim it did."""
    variables = (TERRAFORM / "variables.tf").read_text()
    assert 'variable "enable_compute"' in variables
    assert "default     = false" in variables
