"""AdaFace IR-50 loader (trained on MS1MV3).

Training spec (verified against mk-minchul/AdaFace repo):
- Input: 112x112 BGR (explicit in AdaFace docs — differs from InsightFace which
  internally uses RGB)
- Normalization: (pixel/255 - 0.5) / 0.5 → [-1, 1], equivalent to (pixel - 127.5)/127.5
- Alignment: 5-point landmark affine to ArcFace reference template
  (MS1MV3 was aligned with RetinaFace; we use SCRFD-500MF via buffalo_sc
  as a practical substitute — the 5-pt template is identical, so the post-warp
  112x112 crops are nearly identical across detector choice)

Checkpoint: minchul/cvlface_adaface_ir50_ms1mv3 (full pickled nn.Module).
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

import numpy as np  # noqa: E402
from insightface.app import FaceAnalysis  # noqa: E402
from insightface.utils import face_align  # noqa: E402


class AdaFaceIr50Loader:
    name = "adaface_ir50"

    def __init__(self):
        self._detector: FaceAnalysis | None = None
        self._model = None
        self._torch = None

    def load(self) -> None:
        import torch
        from huggingface_hub import hf_hub_download

        detector = FaceAnalysis(
            name="buffalo_sc",
            allowed_modules=["detection"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        detector.prepare(ctx_id=0, det_size=(640, 640))

        model_path = hf_hub_download(
            repo_id="minchul/cvlface_adaface_ir50_ms1mv3",
            filename="model.pt",
        )
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        model.eval()

        self._detector = detector
        self._model = model
        self._torch = torch

    def embed(self, image_bgr: np.ndarray) -> np.ndarray | None:
        if self._detector is None or self._model is None:
            raise RuntimeError("load() must be called before embed()")

        faces = self._detector.get(image_bgr)
        if not faces:
            return None

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        aligned_bgr = face_align.norm_crop(image_bgr, landmark=face.kps, image_size=112)

        # BGR uint8 HWC → BGR float32 CHW normalized to [-1, 1]
        tensor = (aligned_bgr.astype(np.float32) - 127.5) / 127.5
        tensor = tensor.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 112, 112)

        torch = self._torch
        with torch.no_grad():
            out = self._model(torch.from_numpy(tensor))

        # AdaFace model may return (feature, norm) tuple or just feature
        emb = out[0] if isinstance(out, (tuple, list)) else out
        return emb.squeeze(0).cpu().numpy().astype(np.float32)


def ir50() -> AdaFaceIr50Loader:
    return AdaFaceIr50Loader()
