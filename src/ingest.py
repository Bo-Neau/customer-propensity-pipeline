"""
Ingest task.

Reads the raw REES46 e-commerce behavior CSV and stages it as Parquet.
Runs as a PySpark job since the raw monthly CSV (40 to 67 million rows) is
too large to comfortably load with pandas.
"""
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

RAW_SCHEMA = StructType([
    StructField("event_time", StringType(), True),
    StructField("event_type", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("category_id", StringType(), True),
    StructField("category_code", StringType(), True),
    StructField("brand", StringType(), True),
    StructField("price", DoubleType(), True),
    StructField("user_id", StringType(), True),
    StructField("user_session", StringType(), True),
])


def run(raw_csv_path: str, staged_parquet_path: str) -> str:
    spark = (
        SparkSession.builder
        .appName("propensity-ingest")
        .master("local[*]")
        .getOrCreate()
    )
    try:
        df = spark.read.csv(raw_csv_path, header=True, schema=RAW_SCHEMA)
        df.write.mode("overwrite").parquet(staged_parquet_path)
        row_count = df.count()
        print(f"Ingested {row_count} rows from {raw_csv_path} to {staged_parquet_path}")
        return staged_parquet_path
    finally:
        spark.stop()


if __name__ == "__main__":
    run("/opt/airflow/data/raw/2019-Oct.csv", "/opt/airflow/data/staged/events.parquet")
