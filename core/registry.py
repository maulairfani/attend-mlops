"""MLflow Model Registry wrapper.

Handles model registration and retrieval from MLflow Model Registry.
Artifacts (ONNX files) are stored in MinIO via MLflow artifact store.
"""

import mlflow
from mlflow.tracking import MlflowClient


def register_model(model_path: str, metrics: dict, model_name: str, thresholds: dict) -> bool:
    """
    Register model artifact to MLflow Model Registry if metrics pass thresholds.

    Args:
        model_path: local path to ONNX model file
        metrics: evaluation metrics dict (tar_at_far_0.1, tar_at_far_1.0, auc)
        model_name: name to register under in MLflow Model Registry
        thresholds: minimum metric values to pass (from configs/thresholds.yaml)

    Returns:
        True if registered, False if metrics did not pass thresholds.
    """
    for metric, min_value in thresholds.items():
        if metric in metrics and metrics[metric] < min_value:
            print(f"FAIL: {metric} = {metrics[metric]:.4f} < threshold {min_value}")
            return False

    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(model_path, artifact_path="model")

        run_id = mlflow.active_run().info.run_id

    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)

    client.set_registered_model_tag(model_name, "metrics", str(metrics))
    print(f"Registered {model_name} version {mv.version}")
    return True


def get_production_model(model_name: str, download_dir: str = "/tmp/mlflow_models") -> str:
    """
    Fetch the current production model artifact from MLflow Model Registry.

    The production model is the latest version tagged as 'production' alias.

    Args:
        model_name: registered model name in MLflow
        download_dir: local directory to download artifact to

    Returns:
        Local path to downloaded ONNX model file.
    """
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(model_name, "production")
    model_uri = f"models:/{model_name}@production"
    local_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=download_dir)
    print(f"Downloaded {model_name} v{model_version.version} to {local_path}")
    return local_path


def promote_to_production(model_name: str, version: int) -> None:
    """
    Promote a specific model version to production alias.

    Args:
        model_name: registered model name
        version: version number to promote
    """
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "production", str(version))
    print(f"Promoted {model_name} v{version} to production")
