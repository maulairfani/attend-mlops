"""ZenML step: load cached embeddings for each pair, compute cosine similarity.

Pairs without a cached embedding for both images are skipped (counted in
`n_skipped` downstream). This keeps the pair loop cheap — all heavy work
(detect/align/embed) already happened in `extract_embeddings`.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from typing_extensions import Annotated
from zenml import step


def _load(cache_dir: Path, rel_image_path: str) -> np.ndarray | None:
    p = cache_dir / Path(rel_image_path).with_suffix(".npy")
    if not p.exists():
        return None
    return np.load(p)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


@step
def compute_similarities(
    pairs: list[tuple[str, str, bool]],
    cache_path: str,
) -> Tuple[
    Annotated[list[float], "similarities"],
    Annotated[list[int], "labels"],
    Annotated[int, "n_skipped"],
]:
    cache_dir = Path(cache_path)

    sims: list[float] = []
    labels: list[int] = []
    n_skipped = 0

    for rel_a, rel_b, is_same in pairs:
        emb_a = _load(cache_dir, rel_a)
        emb_b = _load(cache_dir, rel_b)
        if emb_a is None or emb_b is None:
            n_skipped += 1
            continue
        sims.append(_cosine(emb_a, emb_b))
        labels.append(int(is_same))

    print(f"[compute_similarities] evaluated={len(sims)}, skipped={n_skipped}")
    return sims, labels, n_skipped
