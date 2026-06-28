"""
Evaluate task.

Pulls the logged AUC back from MLflow and gates the pipeline on it. This is
what makes the pipeline an actual model-quality gate rather than a script
that always reports success regardless of how the model performed.
"""
import mlflow

AUC_THRESHOLD = 0.90


class ModelQualityError(Exception):
    pass


def run(run_id: str) -> str:
    client = mlflow.tracking.MlflowClient()
    run_obj = client.get_run(run_id)
    auc = run_obj.data.metrics.get("auc")

    if auc is None:
        raise ModelQualityError(f"No AUC metric found for run {run_id}")

    if auc < AUC_THRESHOLD:
        raise ModelQualityError(
            f"Model AUC {auc:.4f} is below the deployment threshold of {AUC_THRESHOLD}"
        )

    print(f"Evaluation passed: AUC {auc:.4f} >= threshold {AUC_THRESHOLD}")
    return run_id
