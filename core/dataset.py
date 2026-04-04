"""DVC dataset management utilities."""


def pull_dataset(dataset_name: str, version: str = None) -> str:
    """
    Pull a versioned dataset via DVC.

    Returns:
        Local path to dataset.
    """
    raise NotImplementedError


def push_dataset(data_path: str, dataset_name: str) -> str:
    """
    Push dataset to DVC remote and return version hash.
    """
    raise NotImplementedError
