"""Evaluate a face recognition model on LFW pairs.

Metrics:
- TAR@FAR=0.1%  (True Accept Rate at False Accept Rate 0.1%)
- TAR@FAR=1.0%  (True Accept Rate at False Accept Rate 1.0%)
- AUC           (Area Under ROC curve)

Model candidates:
- "arcface_buffalo_l"  — InsightFace buffalo_l (ArcFace R50, current Attend.AI model)
- "arcface_r100"       — InsightFace buffalo (ArcFace R100)
- "adaface_ir50"       — AdaFace IR-50 (via HuggingFace)
- "facenet512"         — FaceNet 512-d (via ONNX)
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

from pathlib import Path  # noqa: E402

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
from insightface.app import FaceAnalysis  # noqa: E402
from sklearn.metrics import auc, roc_curve  # noqa: E402
from zenml import step  # noqa: E402


# --- Model loaders ---

def _load_insightface(model_pack: str):
    """Load an InsightFace model pack. Returns a FaceAnalysis app."""
    app = FaceAnalysis(name=model_pack, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def _get_embedding_insightface(app, face_array: np.ndarray) -> np.ndarray | None:
    """Extract embedding from a preprocessed (112x112 float32) face array."""
    # Convert back to uint8 for InsightFace input
    img = np.clip((face_array * 128.0 + 127.5), 0, 255).astype(np.uint8)
    import cv2
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    faces = app.get(img_bgr)
    if not faces:
        return None
    return faces[0].embedding


def _load_model(model_name: str):
    """Load model by name. Returns (model_object, get_embedding_fn)."""
    if model_name == "arcface_buffalo_l":
        app = _load_insightface("buffalo_l")
        return app, lambda arr: _get_embedding_insightface(app, arr)

    elif model_name == "arcface_r100":
        app = _load_insightface("buffalo")
        return app, lambda arr: _get_embedding_insightface(app, arr)

    elif model_name == "adaface_ir50":
        import torch
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id="minchul/cvlface_adaface_ir50_ms1mv3",
            filename="model.pt",
        )
        model = torch.load(model_path, map_location="cpu")
        model.eval()

        def get_embedding(face_array: np.ndarray) -> np.ndarray | None:
            import torch
            tensor = torch.from_numpy(face_array.transpose(2, 0, 1)).unsqueeze(0).float()
            with torch.no_grad():
                emb = model(tensor)
            return emb.squeeze().numpy()

        return model, get_embedding

    elif model_name == "facenet512":
        # FaceNet512 via deepface — outputs 512-d embedding
        # deepface handles its own model download
        from deepface import DeepFace

        def get_embedding(face_array: np.ndarray) -> np.ndarray | None:
            # Convert back to uint8 RGB for deepface
            img = np.clip((face_array * 128.0 + 127.5), 0, 255).astype(np.uint8)
            result = DeepFace.represent(img, model_name="Facenet512", enforce_detection=False)
            return np.array(result[0]["embedding"])

        return None, get_embedding

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: arcface_buffalo_l, arcface_r100, adaface_ir50, facenet512")


# --- Pairs loader ---

def _load_pairs(pairs_path: Path, aligned_dir: Path):
    """
    Parse LFW pairs.txt and return list of (emb_path_1, emb_path_2, is_same) tuples.

    pairs.txt format:
        Line 1: <num_folds>\t<pairs_per_fold>
        Same pairs: <name>\t<img_idx_1>\t<img_idx_2>
        Diff pairs: <name_1>\t<img_idx_1>\t<name_2>\t<img_idx_2>
    """
    pairs = []
    with open(pairs_path) as f:
        lines = f.read().strip().splitlines()

    header = lines[0].split()
    # header: <num_folds> <pairs_per_fold>
    _ = header

    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) == 3:
            # Same person pair
            name, idx1, idx2 = parts
            p1 = aligned_dir / name / f"{name}_{int(idx1):04d}.npy"
            p2 = aligned_dir / name / f"{name}_{int(idx2):04d}.npy"
            pairs.append((p1, p2, True))
        elif len(parts) == 4:
            # Different person pair
            name1, idx1, name2, idx2 = parts
            p1 = aligned_dir / name1 / f"{name1}_{int(idx1):04d}.npy"
            p2 = aligned_dir / name2 / f"{name2}_{int(idx2):04d}.npy"
            pairs.append((p1, p2, False))

    return pairs


# --- TAR@FAR computation ---

def _compute_tar_at_far(similarities: list[float], labels: list[bool], target_far: float) -> float:
    """Return TAR at a specific FAR threshold."""
    fpr, tpr, _ = roc_curve(labels, similarities)
    # Find TPR where FPR is closest to target_far
    idx = np.searchsorted(fpr, target_far)
    if idx >= len(tpr):
        return float(tpr[-1])
    return float(tpr[idx])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


# --- ZenML step ---

@step
def evaluate(model_name: str, preprocessed_data_path: str, pairs_path: str, max_pairs: int | None = None) -> dict:
    """
    Evaluate a face recognition model on LFW pairs.

    Args:
        model_name: one of arcface_buffalo_l, arcface_r100, adaface_ir50, facenet512
        preprocessed_data_path: path to aligned/ directory from preprocess step
        pairs_path: path to LFW pairs.txt

    Returns:
        {
            "model": model_name,
            "tar_at_far_0.1": float,
            "tar_at_far_1.0": float,
            "auc": float,
            "n_pairs": int,
            "n_skipped": int,
        }
    """
    aligned_dir = Path(preprocessed_data_path)
    pairs = _load_pairs(Path(pairs_path), aligned_dir)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    _, get_embedding = _load_model(model_name)

    similarities = []
    labels = []
    skipped = 0

    print(f"Evaluating {model_name} on {len(pairs)} pairs...")
    for p1, p2, is_same in pairs:
        if not p1.exists() or not p2.exists():
            skipped += 1
            continue

        arr1 = np.load(p1)
        arr2 = np.load(p2)

        emb1 = get_embedding(arr1)
        emb2 = get_embedding(arr2)

        if emb1 is None or emb2 is None:
            skipped += 1
            continue

        similarities.append(_cosine_similarity(emb1, emb2))
        labels.append(int(is_same))

    fpr, tpr, _ = roc_curve(labels, similarities)
    roc_auc = float(auc(fpr, tpr))

    metrics = {
        "model": model_name,
        "tar_at_far_0.1": _compute_tar_at_far(similarities, labels, target_far=0.001),
        "tar_at_far_1.0": _compute_tar_at_far(similarities, labels, target_far=0.01),
        "auc": roc_auc,
        "n_pairs": len(similarities),
        "n_skipped": skipped,
    }

    print(f"Results for {model_name}:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Log to MLflow
    mlflow.set_experiment("attend-face-recognition-benchmark")
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({"model": model_name})
        mlflow.log_metrics({
            "tar_at_far_0.1": metrics["tar_at_far_0.1"],
            "tar_at_far_1.0": metrics["tar_at_far_1.0"],
            "auc": metrics["auc"],
            "n_pairs": float(metrics["n_pairs"]),
            "n_skipped": float(metrics["n_skipped"]),
        })

    return metrics
