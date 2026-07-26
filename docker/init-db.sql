-- Runs once, on first Postgres boot (docker-entrypoint-initdb.d).
--
-- This file creates databases only. Tables come from the SQLAlchemy metadata in
-- src/quantfolio/storage/schema.py via create_all(), so there is exactly one
-- definition of each table and the DDL cannot drift from what the code writes.

-- Airflow's own metadata database, kept separate from project data so a
-- `docker compose down -v` on one does not imply losing the other, and so
-- Airflow's schema migrations never touch the feature store.
CREATE DATABASE airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow TO quantfolio;

-- MLflow's backend store (Phase 2).
CREATE DATABASE mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO quantfolio;

-- The feature store itself (POSTGRES_DB) is created by the official image's
-- entrypoint before this script runs.
