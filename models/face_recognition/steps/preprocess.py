from zenml import step


@step
def preprocess(data_path: str) -> str:
    """
    Preprocess raw face images.

    Operations:
        - Face detection & alignment
        - Resize to model input size
        - Normalize pixel values
        - Data augmentation (flip, brightness, etc.)

    Returns:
        Path to preprocessed dataset.
    """
    raise NotImplementedError
