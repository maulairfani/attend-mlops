"""ZenML step: run a model's full per-image pipeline, persist embeddings to disk.

Cache layout:
    <cache_root>/<model_name>/<rel_path_without_ext>.npy

Each entry is the raw (unnormalized) embedding from the model loader.
Detection failures are NOT persisted as cache hits — they retry on every run
(acceptable because LFW miss rate is low; simplicity > micro-optimization).

Return value: path to the per-model cache directory. `compute_similarities`
reads from it.
"""

from pathlib import Path

import cv2
import numpy as np
from zenml import step

from models.face_recognition.loaders import get_loader


def _cache_path(cache_root: Path, model_name: str, rel_image_path: str) -> Path:
    rel = Path(rel_image_path).with_suffix(".npy")
    return cache_root / model_name / rel


@step
def extract_embeddings(
    image_root: str,
    unique_images: list[str],
    model_name: str,
    cache_root: str = "data/cache/embeddings",
) -> str:
    image_root_path = Path(image_root)
    cache_root_path = Path(cache_root)
    model_cache_dir = cache_root_path / model_name

    # Split into hits (skip) vs misses (need computing).
    misses: list[str] = []
    hits = 0
    for rel in unique_images:
        if _cache_path(cache_root_path, model_name, rel).exists():
            hits += 1
        else:
            misses.append(rel)

    print(f"[extract_embeddings:{model_name}] cache: {hits} hits, {len(misses)} misses")

    if not misses:
        return str(model_cache_dir.resolve())

    loader = get_loader(model_name)
    loader.load()

    n_failed = 0
    for i, rel in enumerate(misses, 1):
        img_path = image_root_path / rel
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  cv2.imread failed: {rel}")
            n_failed += 1
            continue

        emb = loader.embed(image_bgr)
        if emb is None:
            n_failed += 1
            continue

        out_path = _cache_path(cache_root_path, model_name, rel)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, emb)

        if i % 500 == 0:
            print(f"  [{i}/{len(misses)}] processed")

    print(
        f"[extract_embeddings:{model_name}] done. "
        f"written={len(misses) - n_failed}, failed_detect={n_failed}"
    )
    return str(model_cache_dir.resolve())
