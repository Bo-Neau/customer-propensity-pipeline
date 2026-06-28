"""
Propensity pipeline DAG.

ingest -> validate -> transform -> quality_gate -> train -> evaluate -> deploy

Each task writes its output to disk (or returns a small string like an
MLflow run_id) and passes only that through XCom. Never pass a DataFrame
through XCom, it is backed by the metadata database and is not built for
bulk data.

The `from airflow.sdk import dag, task` import is the canonical TaskFlow
import for Airflow 3.x (confirmed against the current TaskFlow docs for
3.2.2).
"""
import sys

import pendulum
from airflow.exceptions import AirflowException
from airflow.sdk import dag, task

sys.path.insert(0, "/opt/airflow/src")

import ingest
import validate_schema
import transform
import quality_gate
import train
import evaluate
import deploy

RAW_CSV_PATH = "/opt/airflow/data/raw/2019-Oct.csv"
STAGED_PARQUET_PATH = "/opt/airflow/data/staged/events.parquet"
FEATURES_PARQUET_PATH = "/opt/airflow/data/processed/features.parquet"


@dag(
    dag_id="propensity_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["propensity", "pyspark", "portfolio"],
)
def propensity_pipeline():

    @task
    def ingest_task():
        return ingest.run(RAW_CSV_PATH, STAGED_PARQUET_PATH)

    @task
    def validate_task(staged_path: str):
        try:
            return validate_schema.run(staged_path)
        except validate_schema.SchemaValidationError as exc:
            raise AirflowException(str(exc)) from exc

    @task
    def transform_task(validated_path: str):
        return transform.run(validated_path, FEATURES_PARQUET_PATH)

    @task
    def quality_gate_task(features_path: str):
        try:
            return quality_gate.run(features_path)
        except quality_gate.QualityGateError as exc:
            raise AirflowException(str(exc)) from exc

    @task
    def train_task(gated_features_path: str):
        return train.run(gated_features_path)

    @task
    def evaluate_task(train_run_id: str):
        # NOTE: do not name this parameter `run_id`. Airflow injects `run_id`
        # (and ds, ti, params, ...) as reserved task-context keys, and a
        # TaskFlow arg with the same name raises "The key 'run_id' in args is a
        # part of kwargs and therefore reserved." at execution time.
        try:
            return evaluate.run(train_run_id)
        except evaluate.ModelQualityError as exc:
            raise AirflowException(str(exc)) from exc

    @task
    def deploy_task(evaluated_run_id: str):
        deploy.run(evaluated_run_id)

    staged = ingest_task()
    validated = validate_task(staged)
    features = transform_task(validated)
    gated = quality_gate_task(features)
    train_run_id = train_task(gated)
    evaluated = evaluate_task(train_run_id)
    deploy_task(evaluated)


propensity_pipeline()
