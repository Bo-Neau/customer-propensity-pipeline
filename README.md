# Customer Propensity Pipeline: PySpark + Airflow Rebuild

Migrated a single machine pandas pipeline to a distributed, orchestrated
architecture (PySpark plus Apache Airflow 3), with an automated data quality
gate that blocks bad batches before they reach the model.

The original pipeline ran on 607,056 records with pandas, scikit-learn, and
XGBoost. This rebuild applies the same engagement, intent, and conversion
scoring framework to the REES46 multi-category e-commerce behavior dataset
(40 to 67 million events per month), roughly 70 to 100 times the original
scale. It has been run end to end on the full October 2019 file:
**42,448,764 events aggregated to 9.24 million session level features**, with
a model that reaches **0.935 validation AUC** and is registered in MLflow.

## Architecture

```
ingest -> validate -> transform -> quality_gate -> train -> evaluate -> deploy
```

- **ingest**: raw CSV staged to Parquet (PySpark)
- **validate**: structural schema checks, fails fast on a malformed batch (PySpark)
- **transform**: feature engineering, aggregates raw events to one row per session (PySpark)
- **quality_gate**: semantic and statistical checks on the feature table (Great Expectations Core)
- **train**: XGBoost classifier, logged to MLflow
- **evaluate**: AUC threshold gate against the logged run
- **deploy**: promotes the validated model version by pointing the MLflow `@production` alias at it

Every task writes its output to disk and passes only a small string (a file
path or an MLflow run id) through Airflow XCom. Bulk data never travels through
the metadata database.

## Results

| Metric | Value |
| --- | --- |
| DAG tasks green | 7 of 7 |
| Events ingested | 42,448,764 |
| Session features | 9,244,421 (unique, non null sessions) |
| Class balance | 6.81 percent buyers, 93.19 percent non buyers |
| Validation AUC | 0.935 |
| Registered model | `propensity-model` v1, alias `@production` |

The quality gate earns its place: on the real data it caught 2 null and 350
duplicate session keys, which were resolved at the feature engineering layer so
the gate passes on a clean, correctly keyed table.

Top features by gain: `cart_count`, `session_duration_seconds`, and
`view_count`. Buyers add to cart roughly 16 times more often than non buyers
and, notably, spend less time per session, consistent with decisive,
intent driven visits.

A full engineering report, including the eight defects found and fixed while
bringing the scaffold to its first green run, is in
[Propensity_Pipeline_Report.pdf](Propensity_Pipeline_Report.pdf).

## Stack

PySpark 4.1.2, Apache Airflow 3.2.2 (Docker), Great Expectations Core 1.18.1,
XGBoost 3.3.0, MLflow 3.14.0, PostgreSQL 16, OpenJDK 17.

## Dataset

eCommerce behavior data from multi-category store, courtesy of the REES46
Marketing Platform via Kaggle. Free to use with attribution.
https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store

## Setup

1. Download one month of REES46 data into `data/raw/`:

   ```
   kaggle datasets download \
     -d mkechinov/ecommerce-behavior-data-from-multi-category-store \
     -f 2019-Oct.csv -p data/raw/ --unzip
   ```

2. `cp .env.example .env` and set `AIRFLOW_UID` (run `id -u` on Mac and Linux,
   leave the default on Windows).

3. Build and start the stack:

   ```
   docker compose up --build
   ```

4. Open the Airflow UI at `http://localhost:8080`. Airflow 3 uses the
   SimpleAuthManager, which generates the admin password on each start. The
   username is `admin`; retrieve the current password with:

   ```
   docker compose logs airflow-api-server | grep "Password for user"
   ```

5. Trigger the `propensity_pipeline` DAG, from the UI or the CLI:

   ```
   docker compose exec airflow-scheduler airflow dags trigger propensity_pipeline
   ```

## Notes on the build

This project runs PySpark in `local[*]` mode inside the Airflow worker rather
than standing up a separate Spark cluster, a deliberate scope decision for a
single node portfolio build. The pandas bridge in `train.py` is intentional:
the per session feature table fits comfortably in memory. See `CLAUDE.md` for
the version sensitive details that were verified during commissioning.

## Original pipeline

The pandas, scikit-learn version this project is based on:
https://bo-neau.github.io/machine-learning-project.github.io
