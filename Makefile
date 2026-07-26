.DEFAULT_GOAL := help
COMPOSE := docker compose
DAG := market_data_pipeline

.PHONY: help up down restart logs ps build init test test-fast test-slow lint fmt \
        trigger backfill train train-local drift psql s3-ls weights clean \
        loadtest predict tf-init tf-plan tf-apply tf-destroy tf-verify

# terraform and opentofu read the same HCL; use whichever is installed.
TF := $(shell command -v terraform 2>/dev/null || command -v tofu 2>/dev/null)

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

up: ## Build and start the whole stack
	@test -f .env || (echo "no .env found — copying .env.example" && cp .env.example .env)
	$(COMPOSE) up -d --build
	@echo ""
	@echo "  Airflow    http://localhost:8080  (airflow / airflow)"
	@echo "  API        http://localhost:8000/docs"
	@echo "  MLflow     http://localhost:5000"
	@echo "  Grafana    http://localhost:3000  (admin / admin)"
	@echo "  Prometheus http://localhost:9090"

down: ## Stop the stack (keeps volumes)
	$(COMPOSE) down

clean: ## Stop the stack and delete all data volumes
	$(COMPOSE) down -v

restart: down up ## Restart the stack

build: ## Rebuild images without starting
	$(COMPOSE) build

ps: ## Show container status
	$(COMPOSE) ps

logs: ## Tail logs (make logs SERVICE=airflow-scheduler)
	$(COMPOSE) logs -f $(SERVICE)

init: ## Create the project tables in Postgres
	$(COMPOSE) run --rm airflow-scheduler \
		python -c "from quantfolio.storage.db import create_all; create_all()"

trigger: ## Unpause and trigger one pipeline run
	$(COMPOSE) exec airflow-scheduler airflow dags unpause $(DAG)
	$(COMPOSE) exec airflow-scheduler airflow dags trigger $(DAG)

backfill: ## Backfill a date range (make backfill START=2024-01-01 END=2024-03-01)
	$(COMPOSE) exec airflow-scheduler \
		airflow dags backfill --start-date $(START) --end-date $(END) $(DAG)

train: ## Train both frameworks in the stack, logging to the MLflow server
	$(COMPOSE) exec airflow-scheduler python /opt/quantfolio/scripts/train.py

train-local: ## Train outside Docker, logging to ./mlflow.db
	uv run --extra ml --extra dev python scripts/train.py --local-mlflow

drift: ## Demonstrate that the drift sensor fires (no infrastructure needed)
	uv run --extra ml --extra dev python scripts/inject_drift.py --dry-run

psql: ## Open a psql shell on the feature store
	$(COMPOSE) exec postgres psql -U quantfolio -d quantfolio

weights: ## Show the most recent portfolio weights
	$(COMPOSE) exec postgres psql -U quantfolio -d quantfolio -c \
		"SELECT ticker, ROUND(weight::numeric, 4) AS weight, method FROM portfolio_weights \
		 WHERE as_of_date = (SELECT MAX(as_of_date) FROM portfolio_weights) ORDER BY weight DESC;"

s3-ls: ## List the raw bucket in LocalStack
	$(COMPOSE) exec localstack awslocal s3 ls s3://quantfolio-raw --recursive

predict: ## Check what the API is currently able to serve
	@curl -s http://localhost:8000/predict/model/status | python3 -m json.tool
	@curl -s http://localhost:8000/predict/AAPL | python3 -m json.tool

loadtest: ## Measure p50/p95/p99 (make loadtest USERS=50 TIME=60s)
	@mkdir -p results
	uv run --extra dev locust -f scripts/locustfile.py \
		--host http://localhost:8000 --headless \
		-u $(or $(USERS),50) -r 10 -t $(or $(TIME),60s) \
		--csv results/load
	@echo "CSV written to results/load_stats.csv — quote these with the concurrency used"

# --------------------------------------------------------------------------- #
# terraform (works with terraform or opentofu)
# --------------------------------------------------------------------------- #
tf-init: ## Initialize the Terraform working directory
	@test -n "$(TF)" || (echo "install terraform or opentofu first" && exit 1)
	cd terraform && $(TF) init

tf-plan: ## Show what would be created against LocalStack
	cd terraform && $(TF) plan

tf-apply: ## Really create the buckets in LocalStack
	cd terraform && $(TF) apply -auto-approve

tf-verify: ## Prove Terraform created the buckets
	$(COMPOSE) exec localstack awslocal s3 ls

tf-destroy: ## Tear the Terraform-managed resources back down
	cd terraform && $(TF) destroy -auto-approve

test: ## Run the full test suite
	uv run --extra dev --extra api pytest

test-fast: ## Run tests, skipping live infrastructure and network training
	uv run --extra dev --extra api pytest -m "not integration and not slow"

test-slow: ## Run only the tests that train real networks
	uv run --extra dev --extra api --extra ml pytest -m slow

lint: ## Check formatting and lint rules
	uv run --extra dev --extra api ruff check src tests dags
	uv run --extra dev --extra api ruff format --check src tests dags

fmt: ## Auto-format
	uv run --extra dev --extra api ruff format src tests dags
	uv run --extra dev --extra api ruff check --fix src tests dags
