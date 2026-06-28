"""
Deploy task.

Promotes the validated MLflow model version by pointing the "production"
registry alias at it. A real serving layer (Streamlit app, SageMaker
endpoint, etc) would load `models:/propensity-model@production` rather than
a specific run.

NOTE: MLflow 3 removed model stages (the old
`transition_model_version_stage` / "Production" stage API). The current
equivalent is named aliases, set with `set_registered_model_alias`. An
alias points at exactly one version, so reassigning it here is what
"promote to production" means now.
"""
import mlflow

MODEL_NAME = "propensity-model"
PRODUCTION_ALIAS = "production"


def run(run_id: str) -> None:
    client = mlflow.tracking.MlflowClient()

    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(model_uri, MODEL_NAME)

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=PRODUCTION_ALIAS,
        version=registered.version,
    )

    print(
        f"Deployed {MODEL_NAME} version {registered.version}, "
        f"alias @{PRODUCTION_ALIAS} now points to it"
    )
