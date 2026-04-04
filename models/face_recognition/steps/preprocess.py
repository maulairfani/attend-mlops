"""Preprocess LFW images for face recognition benchmarking.

Operations:
- Face detection & alignment via InsightFace
- Resize to 112x112 (ArcFace standard input size)
- Normalize: (pixel - 127.5) / 128.0
- Save as .npy arrays keyed by image path

Images that fail detection are skipped and logged.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from zenml import step


def _load_detector() -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_sc", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def _align_and_normalize(app: FaceAnalysis, image_path: Path) -> np.ndarray | None:
    """Detect, align, and normalize a single face image. Returns None if no face found."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    faces = app.get(img)
    if not faces:
        return None

    # Take the largest face if multiple detected
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    aligned = face.normed_embedding  # InsightFace already returns L2-normalized embedding

    # For preprocessing we want the aligned face image, not the embedding
    # Use the aligned crop from kps (5 keypoints)
    from insightface.utils import face_align
    aligned_img = face_align.norm_crop(img, landmark=face.kps, image_size=112)

    # Normalize: (pixel - 127.5) / 128.0
    normalized = (aligned_img.astype(np.float32) - 127.5) / 128.0
    return normalized


@step
def preprocess(data_path: str) -> str:
    """
    Detect, align, and normalize all LFW face images.

    Input:
        data_path: path to LFW dataset directory (contains lfw_funneled/ and pairs.txt)

    Output:
        Path to preprocessed directory containing:
        - aligned/<person>/<image>.npy  — normalized face arrays (112x112x3 float32)
        - skipped.json                  — list of images where no face was detected
    """
    data_dir = Path(data_path)
    images_dir = data_dir / "lfw_funneled"
    out_dir = data_dir / "aligned"
    out_dir.mkdir(parents=True, exist_ok=True)

    app = _load_detector()

    image_paths = list(images_dir.rglob("*.jpg"))
    print(f"Processing {len(image_paths)} images...")

    skipped = []
    processed = 0

    for img_path in image_paths:
        rel = img_path.relative_to(images_dir)
        out_path = out_dir / rel.with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            processed += 1
            continue

        result = _align_and_normalize(app, img_path)
        if result is None:
            skipped.append(str(rel))
            continue

        np.save(out_path, result)
        processed += 1

    skipped_path = data_dir / "skipped.json"
    with open(skipped_path, "w") as f:
        json.dump(skipped, f, indent=2)

    print(f"Done. Processed: {processed}, Skipped: {len(skipped)}")
    print(f"Skipped list saved to {skipped_path}")

    return str(out_dir.resolve())
