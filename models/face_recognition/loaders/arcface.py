"""ArcFace loaders (InsightFace FaceAnalysis).

Covers:
- arcface_buffalo_l   — SCRFD-10GF + iResNet50 @ WebFace600K (w600k_r50.onnx)
- arcface_antelopev2  — SCRFD-10GF + iResNet100 @ Glint360K (glintr100.onnx)

Training spec (both):
- Input: 112x112 BGR (uint8 at API; swapped to RGB internally via swapRB=True
  inside cv2.dnn.blobFromImages → model sees RGB)
- Normalization: (pixel - 127.5) / 127.5 → [-1, 1]
- Alignment: SCRFD detection + 5-point landmark affine warp (handled by FaceAnalysis)

FaceAnalysis.get() returns list[Face]; we pick the largest bbox and use
`.embedding` (raw recognizer output, NOT L2-normalized) to match the base contract.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

import numpy as np  # noqa: E402
from insightface.app import FaceAnalysis  # noqa: E402


class ArcFaceLoader:
    def __init__(self, pack_name: str, display_name: str):
        if pack_name not in {"buffalo_l", "antelopev2"}:
            raise ValueError(
                f"Invalid InsightFace pack: {pack_name!r}. "
                "Valid: buffalo_l, antelopev2"
            )
        self.pack_name = pack_name
        self.name = display_name
        self._app: FaceAnalysis | None = None

    def load(self) -> None:
        app = FaceAnalysis(
            name=self.pack_name,
            allowed_modules=["detection", "recognition"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        self._app = app

    def embed(self, image_bgr: np.ndarray) -> np.ndarray | None:
        if self._app is None:
            raise RuntimeError("load() must be called before embed()")

        faces = self._app.get(image_bgr)
        if not faces:
            return None

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return np.asarray(face.embedding, dtype=np.float32)


def buffalo_l() -> ArcFaceLoader:
    return ArcFaceLoader(pack_name="buffalo_l", display_name="arcface_buffalo_l")


def antelopev2() -> ArcFaceLoader:
    return ArcFaceLoader(pack_name="antelopev2", display_name="arcface_antelopev2")
