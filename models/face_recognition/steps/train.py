from zenml import step


@step
def train(preprocessed_data_path: str, config_path: str) -> str:
    """
    Fine-tune face recognition model.

    Supports:
        - Facenet512
        - ArcFace
        - AdaFace

    Logs all runs to W&B.

    Returns:
        Path to saved model artifact.
    """
    raise NotImplementedError
