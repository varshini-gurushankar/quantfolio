"""Drift sensor: retrain when the deployed model's error deteriorates.

check_drift -> (short circuit) -> trigger training_pipeline

Runs daily. The sensor only decides; ``ShortCircuitOperator`` stops the run when
there is nothing to do, so a quiet day costs one cheap query and the downstream
trigger is skipped rather than failed.

The threshold logic (persistent breach, judged against the model's own
training-time error) lives in ``quantfolio.models.drift`` and is unit-tested
against a synthetic drift injection — see ``scripts/inject_drift.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pendulum
from airflow.decorators import dag, task
from airflow.operators.python import ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from quantfolio.research_pipeline import DEFAULT_MONITORED_MODEL, task_check_drift

DEFAULT_ARGS = {
    "owner": "quantfolio",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def _decide(**context) -> bool:
    """Return True to let the trigger run, False to short-circuit the DAG."""
    report = context["ti"].xcom_pull(task_ids="check_drift")
    breached = bool(report and report.get("breached"))
    print(f"drift decision: {'RETRAIN' if breached else 'no action'} — {report}")
    return breached


@dag(
    dag_id="drift_sensor",
    description="Monitors rolling out-of-sample MSE and triggers retraining on drift",
    default_args=DEFAULT_ARGS,
    schedule="0 6 * * 1-5",  # weekday mornings, after the overnight data run
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["quantfolio", "ml", "monitoring", "phase2"],
    doc_md=__doc__,
)
def drift_sensor():
    @task(task_id="check_drift")
    def check(logical_date: datetime | None = None, **_) -> dict:
        """Compare recent rolling OOS MSE against the model's training-time MSE."""
        exec_date = (logical_date or datetime.now(UTC)).date()
        return task_check_drift(exec_date, model_name=DEFAULT_MONITORED_MODEL)

    report = check()

    gate = ShortCircuitOperator(
        task_id="drift_detected",
        python_callable=_decide,
        # A skipped downstream trigger is the normal, healthy outcome; it should
        # not paint the DAG run as failed.
        ignore_downstream_trigger_rules=True,
    )

    retrain = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="training_pipeline",
        # Do not block a daily sensor for the hours training takes.
        wait_for_completion=False,
        reset_dag_run=True,
    )

    report >> gate >> retrain


drift_sensor()
