"""ZenML step: resolve a benchmark dataset to (image_root, unique_images, pairs).

Dispatches to the appropriate adapter in `datasets/` by `dataset_name`.
Adding a new verification dataset = add an adapter + register here.
"""

from typing import Tuple

from typing_extensions import Annotated
from zenml import step

from models.face_recognition.datasets.lfw import load_lfw


@step
def load_dataset(
    dataset_name: str,
    data_path: str,
    max_pairs: int | None = None,
) -> Tuple[
    Annotated[str, "image_root"],
    Annotated[list[str], "unique_images"],
    Annotated[list[tuple[str, str, bool]], "pairs"],
]:
    if dataset_name == "lfw":
        image_root, unique_images, pairs = load_lfw(data_path)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. Supported: lfw"
        )

    if max_pairs is not None and max_pairs < len(pairs):
        pairs = pairs[:max_pairs]
        # Recompute unique_images from the sliced pair set so extract_embeddings
        # only processes images actually needed.
        unique_images = sorted({p for a, b, _ in pairs for p in (a, b)})

    print(f"[load_dataset] {dataset_name}: {len(unique_images)} unique images, {len(pairs)} pairs")
    return image_root, unique_images, pairs
