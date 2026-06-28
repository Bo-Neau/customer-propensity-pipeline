"""
Minimal smoke test for the transform logic.

Run with: pytest tests/

Uses a tiny synthetic dataset, not the full REES46 file, so it runs in
seconds and catches obvious breakage before you ever wire this into Airflow.
"""
import sys

sys.path.insert(0, "src")

from pyspark.sql import SparkSession

import transform


def test_transform_produces_expected_columns(tmp_path):
    spark = SparkSession.builder.appName("test").master("local[1]").getOrCreate()

    rows = [
        ("s1", "u1", "view", "p1", "electronics.smartphone", "apple", 999.0, "2019-10-01T00:00:00Z"),
        ("s1", "u1", "cart", "p1", "electronics.smartphone", "apple", 999.0, "2019-10-01T00:01:00Z"),
        ("s1", "u1", "purchase", "p1", "electronics.smartphone", "apple", 999.0, "2019-10-01T00:02:00Z"),
        ("s2", "u2", "view", "p2", "electronics.tv", "samsung", 499.0, "2019-10-01T01:00:00Z"),
    ]
    columns = ["user_session", "user_id", "event_type", "product_id", "category_code", "brand", "price", "event_time"]
    df = spark.createDataFrame(rows, columns)

    input_path = str(tmp_path / "input.parquet")
    output_path = str(tmp_path / "output.parquet")
    df.write.parquet(input_path)

    transform.run(input_path, output_path)

    result = spark.read.parquet(output_path).toPandas()
    spark.stop()

    assert "engagement_score" in result.columns
    assert "intent_score" in result.columns
    assert "ordered" in result.columns
    assert result.loc[result["user_session"] == "s1", "ordered"].iloc[0] == 1
    assert result.loc[result["user_session"] == "s2", "ordered"].iloc[0] == 0
