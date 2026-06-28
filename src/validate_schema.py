"""
Validate task.

Structural checks on the staged data: required columns present, non-empty,
event_type values within the expected set. This runs before any feature
engineering work is spent on a malformed batch.

Semantic/statistical checks (value ranges, uniqueness, nulls on derived
features) belong in quality_gate.py instead, this file only checks shape.
"""
from pyspark.sql import SparkSession

REQUIRED_COLUMNS = {
    "event_time", "event_type", "product_id", "category_id",
    "category_code", "brand", "price", "user_id", "user_session",
}

ALLOWED_EVENT_TYPES = {"view", "cart", "purchase", "remove_from_cart"}


class SchemaValidationError(Exception):
    pass


def run(staged_parquet_path: str) -> str:
    spark = (
        SparkSession.builder
        .appName("propensity-validate")
        .master("local[*]")
        .getOrCreate()
    )
    try:
        df = spark.read.parquet(staged_parquet_path)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise SchemaValidationError(f"Missing required columns: {missing}")

        row_count = df.count()
        if row_count == 0:
            raise SchemaValidationError("Staged dataset is empty")

        observed_event_types = {
            row["event_type"]
            for row in df.select("event_type").distinct().collect()
        }
        unexpected = observed_event_types - ALLOWED_EVENT_TYPES
        if unexpected:
            raise SchemaValidationError(f"Unexpected event_type values: {unexpected}")

        print(f"Schema validation passed: {row_count} rows, columns OK")
        return staged_parquet_path
    finally:
        spark.stop()
