"""ZenML step: log a benchmark run to MLflow.

Experiment name: "attend-face-recognition-benchmark"
Run name: model_name
Tags: dataset=<dataset_name>
"""

import mlflow
from zenml import step


@step
def log_mlflow(
    metrics: dict,
    model_name: str,
    dataset_name: str,
) -> None:
    mlflow.set_experiment("attend-face-recognition-benchmark")
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tags({"dataset": dataset_name})
        mlflow.log_params({"model": model_name, "dataset": dataset_name})
        mlflow.log_metrics({
            "tar_at_far_0.1": metrics["tar_at_far_0.1"],
            "tar_at_far_1.0": metrics["tar_at_far_1.0"],
            "auc": metrics["auc"],
            "n_pairs": float(metrics["n_pairs"]),
            "n_skipped": float(metrics["n_skipped"]),
        })
    print(f"[log_mlflow:{model_name}] logged to MLflow")
