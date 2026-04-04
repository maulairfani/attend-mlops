from zenml import step


@step
def evaluate(model_path: str, test_data_path: str) -> dict:
    """
    Evaluate face recognition model.

    Primary metric: TAR@FAR (True Accept Rate at False Accept Rate).

    Returns:
        dict of metrics, e.g.:
        {
            "tar_at_far_0.1": 0.95,
            "tar_at_far_0.01": 0.91,
            "rank1": 0.97,
        }
    """
    raise NotImplementedError
