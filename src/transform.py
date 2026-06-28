"""
Transform task.

Aggregates raw events to one row per user_session and builds the
engagement / intent / conversion features, the same framework the original
pandas pipeline used, applied here at REES46 scale.

view events    -> engagement
cart events    -> intent
purchase event -> the "ordered" target (1 if the session contains a purchase)
"""
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def run(validated_parquet_path: str, features_output_path: str) -> str:
    # In local[*] mode the driver JVM is also the executor, and the default
    # 1g heap is far too small to shuffle ~42M rows (the groupBy + countDistinct
    # below). Without this the JVM OOMs mid-shuffle and py4j reports "Error
    # while sending or receiving". 4g fits comfortably inside an 8GB Docker VM
    # alongside the Airflow processes; override with SPARK_DRIVER_MEMORY.
    driver_memory = os.environ.get("SPARK_DRIVER_MEMORY", "4g")
    spark = (
        SparkSession.builder
        .appName("propensity-transform")
        .master("local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )
    try:
        df = spark.read.parquet(validated_parquet_path)
        # REES46 timestamps look like "2019-10-01 00:00:00 UTC"; the test
        # fixtures use ISO "2019-10-01T00:00:00Z". Parse both explicitly so a
        # format mismatch cannot silently null out every session timestamp.
        df = df.withColumn(
            "event_time",
            F.coalesce(
                F.to_timestamp("event_time", "yyyy-MM-dd HH:mm:ss 'UTC'"),
                F.to_timestamp("event_time", "yyyy-MM-dd'T'HH:mm:ss'Z'"),
                F.to_timestamp("event_time"),
            ),
        )

        # The feature grain is one row per user_session (see module docstring),
        # so user_session must be a clean unique key for the downstream quality
        # gate. Drop the handful of events with a null session id, and group by
        # user_session alone. Grouping by (user_session, user_id) would emit
        # duplicate user_session rows for the ~350 REES46 sessions that carry
        # more than one user_id; keep a single representative user_id instead.
        df = df.filter(F.col("user_session").isNotNull())

        session_agg = (
            df.groupBy("user_session")
            .agg(
                F.first("user_id", ignorenulls=True).alias("user_id"),
                F.sum(F.when(F.col("event_type") == "view", 1).otherwise(0)).alias("view_count"),
                F.sum(F.when(F.col("event_type") == "cart", 1).otherwise(0)).alias("cart_count"),
                F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("purchase_count"),
                F.countDistinct("category_code").alias("distinct_categories_viewed"),
                F.countDistinct("brand").alias("distinct_brands_viewed"),
                F.avg("price").alias("avg_price_viewed"),
                F.max("price").alias("max_price_viewed"),
                F.min("event_time").alias("session_start"),
                F.max("event_time").alias("session_end"),
            )
        )

        session_agg = (
            session_agg
            .withColumn(
                "session_duration_seconds",
                F.col("session_end").cast("long") - F.col("session_start").cast("long"),
            )
            .withColumn("engagement_score", F.col("view_count") + F.col("distinct_categories_viewed"))
            .withColumn("intent_score", F.col("cart_count") * 2 + F.col("distinct_brands_viewed"))
            .withColumn("ordered", F.when(F.col("purchase_count") > 0, 1).otherwise(0))
        )

        session_agg.write.mode("overwrite").parquet(features_output_path)
        print(f"Wrote feature table to {features_output_path}")
        return features_output_path
    finally:
        spark.stop()
