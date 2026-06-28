"""
Quality gate task.

Semantic and statistical checks on the engineered feature table using
GX Core (the free, open-source library, not GX Cloud, which has been
shut down). A failed expectation raises, which fails the Airflow task
and stops a bad batch from reaching training.

NOTE: GX Core's Python API has changed across versions (legacy
DataContext API vs the newer Fluent/Core API used below). Confirm the
exact method names against the current docs at
https://docs.greatexpectations.io before relying on this file, see
CLAUDE.md item 3.
"""
import great_expectations as gx
import pandas as pd


class QualityGateError(Exception):
    pass


EXPECTATIONS = [
    ("user_session", "not_null"),
    ("user_session", "unique"),
    ("engagement_score", "not_null"),
]


def run(features_parquet_path: str) -> str:
    pdf = pd.read_parquet(features_parquet_path)

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("propensity_features")
    asset = data_source.add_dataframe_asset(name="features")
    batch_definition = asset.add_batch_definition_whole_dataframe("batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": pdf})

    checks = [
        gx.expectations.ExpectColumnValuesToNotBeNull(column="user_session"),
        gx.expectations.ExpectColumnValuesToBeUnique(column="user_session"),
        gx.expectations.ExpectColumnValuesToBeBetween(column="ordered", min_value=0, max_value=1),
        gx.expectations.ExpectColumnValuesToBeBetween(column="view_count", min_value=0, max_value=None),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="engagement_score"),
    ]

    results = [batch.validate(expectation) for expectation in checks]
    failed = [r for r in results if not r.success]

    if failed:
        raise QualityGateError(f"{len(failed)} expectation(s) failed quality gate, check GX results")

    print(f"Quality gate passed: {len(results)} expectations checked on {len(pdf)} rows")
    return features_parquet_path
