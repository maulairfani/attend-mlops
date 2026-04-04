from zenml import step


@step
def validate_dataset(data_path: str) -> bool:
    """
    Validate dataset before training using Great Expectations.

    Checks:
        - Schema (required columns/folder structure)
        - Null / corrupt images
        - Minimum images per identity
        - Class distribution
    """
    raise NotImplementedError
