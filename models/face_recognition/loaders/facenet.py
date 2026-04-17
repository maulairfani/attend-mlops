"""FaceNet512 loader via DeepFace.

Training spec (VGGFace2 via Inception-ResNet-v1 @ 512-d):
- Input: **160x160** (NOT 112x112 — don't force-feed a 112 crop)
- Alignment: MTCNN 5-point
- Normalization: per-image prewhitening  ((x - mean) / max(std, 1/sqrt(N)))
- Color at API: BGR (DeepFace handles BGR→RGB internally before model)

All preprocessing is owned by DeepFace.represent(); we pass the raw BGR image
and let it detect → align (MTCNN) → resize (160) → prewhiten → forward.
"""

import numpy as np
from deepface import DeepFace


class FaceNet512Loader:
    name = "facenet512"

    def load(self) -> None:
        # DeepFace lazy-loads models on first represent() call; pre-warm by
        # triggering a one-time build via the DeepFace ModelBuilder pattern.
        # A no-op here keeps the interface symmetric; actual download happens
        # on first embed().
        return None

    def embed(self, image_bgr: np.ndarray) -> np.ndarray | None:
        try:
            results = DeepFace.represent(
                img_path=image_bgr,
                model_name="Facenet512",
                detector_backend="mtcnn",
                enforce_detection=True,
                align=True,
            )
        except (ValueError, Exception):
            # enforce_detection=True raises ValueError when no face found
            return None

        if not results:
            return None

        def area(entry: dict) -> float:
            fa = entry.get("facial_area", {})
            return float(fa.get("w", 0)) * float(fa.get("h", 0))

        best = max(results, key=area)
        return np.asarray(best["embedding"], dtype=np.float32)


def facenet512() -> FaceNet512Loader:
    return FaceNet512Loader()
