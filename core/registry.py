"""W&B Model Registry wrapper."""


def register_model(model_path: str, metrics: dict, model_name: str, thresholds: dict) -> bool:
    """
    Register model artifact to W&B Model Registry.

    Returns True if registered, False if metrics did not pass thresholds.
    """
    raise NotImplementedError


def get_production_model(model_name: str) -> str:
    """
    Fetch the current production model artifact path from W&B registry.

    Returns:
        Local path to downloaded model artifact.
    """
    raise NotImplementedError
