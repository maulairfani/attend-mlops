"""Benchmark pipeline for face recognition model selection.

Sprint 1 objective: evaluate pre-trained model candidates on LFW pairs
to select the best baseline model for Attend.AI production.

Candidates (each evaluated under its own training-time preprocessing):
- arcface_buffalo_l   (InsightFace SCRFD + iResNet50 @ WebFace600K, current Attend.AI model)
- arcface_antelopev2  (InsightFace SCRFD + iResNet100 @ Glint360K)
- adaface_ir50        (AdaFace IR-50 @ MS1MV3)
- facenet512          (FaceNet 512-d @ VGGFace2)

Pipeline (5 steps, per-model fan-out after load_dataset):

    load_dataset ──► extract_embeddings ──► compute_similarities ──► compute_metrics ──► log_mlflow

Usage:
    python -m models.face_recognition.pipelines.benchmark_pipeline --model arcface_buffalo_l
    python -m models.face_recognition.pipelines.benchmark_pipeline --all
"""

import argparse
import os

import mlflow
from dotenv import load_dotenv
from zenml import pipeline

load_dotenv()

from models.face_recognition.steps.compute_metrics import compute_metrics
from models.face_recognition.steps.compute_similarities import compute_similarities
from models.face_recognition.steps.extract_embeddings import extract_embeddings
from models.face_recognition.steps.load_dataset import load_dataset
from models.face_recognition.steps.log_mlflow import log_mlflow

ALL_CANDIDATES = [
    "arcface_buffalo_l",
    "arcface_antelopev2",
    "adaface_ir50",
    "facenet512",
]

DEFAULT_DATASET = "lfw"
DEFAULT_DATA_PATH = "data/raw/lfw"
DEFAULT_CACHE_ROOT = "data/cache/embeddings"


@pipeline
def benchmark_pipeline(
    model_name: str,
    dataset_name: str = DEFAULT_DATASET,
    data_path: str = DEFAULT_DATA_PATH,
    cache_root: str = DEFAULT_CACHE_ROOT,
    max_pairs: int | None = None,
):
    image_root, unique_images, pairs = load_dataset(
        dataset_name=dataset_name,
        data_path=data_path,
        max_pairs=max_pairs,
    )
    cache_path = extract_embeddings(
        image_root=image_root,
        unique_images=unique_images,
        model_name=model_name,
        cache_root=cache_root,
    )
    similarities, labels, n_skipped = compute_similarities(
        pairs=pairs,
        cache_path=cache_path,
    )
    metrics = compute_metrics(
        similarities=similarities,
        labels=labels,
        n_skipped=n_skipped,
        model_name=model_name,
    )
    log_mlflow(
        metrics=metrics,
        model_name=model_name,
        dataset_name=dataset_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark face recognition models on a verification dataset")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=ALL_CANDIDATES, help="Single model to benchmark")
    group.add_argument("--all", action="store_true", help="Benchmark all candidates")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset name (default: lfw)")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Dataset root path")
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT, help="Embedding cache root")
    parser.add_argument("--max-pairs", type=int, default=None, help="Limit number of pairs (for fast smoke tests)")
    args = parser.parse_args()

    candidates = ALL_CANDIDATES if args.all else [args.model]

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    for model_name in candidates:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name}  (dataset={args.dataset})")
        print(f"{'='*60}")
        benchmark_pipeline(
            model_name=model_name,
            dataset_name=args.dataset,
            data_path=args.data_path,
            cache_root=args.cache_root,
            max_pairs=args.max_pairs,
        )
