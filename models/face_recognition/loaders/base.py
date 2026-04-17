"""ModelLoader protocol for face recognition benchmark.

Each loader owns its full per-model pipeline: detect → align → resize → normalize
→ forward pass → L2-normalized embedding. This keeps comparisons apple-to-apple
against each model's training distribution.

Color contract: every loader's `embed()` receives a **BGR uint8** image
(native cv2.imread output). Each loader performs its own internal BGR↔RGB
conversion per the model's training spec. Never pass RGB arrays here.
"""

from typing import Protocol

import numpy as np


class ModelLoader(Protocol):
    name: str

    def load(self) -> None:
        """Download weights / initialize runtime. Called once before first embed."""
        ...

    def embed(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """Detect, align, and embed a single face.

        Args:
            image_bgr: uint8 HxWx3 BGR array (cv2.imread output).

        Returns:
            1-D embedding (model-specific dim), or None if detection failed.
            Embeddings are NOT L2-normalized here — cosine sim handles that.
        """
        ...
