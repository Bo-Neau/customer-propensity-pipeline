"""
Train task.

Bridges the Spark-built feature table to pandas (it comfortably fits in
memory at the per-session granularity produced by transform.py) and trains
an XGBoost classifier, keeping the same model family and hyperparameter
approach as the original pandas pipeline. Logs the run to MLflow.
"""
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

FEATURE_COLUMNS = [
    "view_count", "cart_count", "distinct_categories_viewed",
    "distinct_brands_viewed", "avg_price_viewed", "max_price_viewed",
    "session_duration_seconds", "engagement_score", "intent_score",
]
TARGET_COLUMN = "ordered"


def run(features_parquet_path: str) -> str:
    pdf = pd.read_parquet(features_parquet_path)
    pdf = pdf.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    X = pdf[FEATURE_COLUMNS]
    y = pdf[TARGET_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    mlflow.set_experiment("propensity-pipeline")
    with mlflow.start_run() as run_obj:
        positive = max((y_train == 1).sum(), 1)
        negative = (y_train == 0).sum()

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=negative / positive,
            eval_metric="auc",
        )
        model.fit(X_train, y_train)

        val_preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_preds)

        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 6)
        mlflow.log_metric("auc", auc)
        # MLflow 3 renamed the positional artifact_path arg to name; the model
        # is still addressable as runs:/<run_id>/model in deploy.py.
        mlflow.xgboost.log_model(model, name="model")

        print(f"Trained model, validation AUC: {auc:.4f}")
        return run_obj.info.run_id
