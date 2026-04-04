from zenml import step


@step
def register(model_path: str, metrics: dict) -> None:
    """
    Register model to W&B Model Registry if metrics pass thresholds.

    Thresholds defined in configs/thresholds.yaml.
    On success, notifies Attend.AI backend via webhook.
    """
    raise NotImplementedError
