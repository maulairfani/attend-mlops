"""Benchmark pipeline for face recognition model selection.

Sprint 1 objective: evaluate pre-trained model candidates on LFW pairs
to select the best baseline model for Attend.AI production.

Candidates:
- arcface_buffalo_l  (InsightFace ArcFace R50, current Attend.AI model)
- arcface_r100       (InsightFace ArcFace R100)
- adaface_ir50       (AdaFace IR-50)
- facenet512         (FaceNet 512-d)

Usage:
    # Benchmark a single model
    python -m models.face_recognition.pipelines.benchmark_pipeline --model arcface_buffalo_l

    # Benchmark all candidates
    python -m models.face_recognition.pipelines.benchmark_pipeline --all
"""

import argparse
import os

import mlflow
from dotenv import load_dotenv
from zenml import pipeline

load_dotenv()

from models.face_recognition.steps.evaluate import evaluate
from models.face_recognition.steps.preprocess import preprocess

ALL_CANDIDATES = [
    "arcface_buffalo_l",
    "arcface_r100",
    "adaface_ir50",
    "facenet512",
]

LFW_DATA_PATH = "data/raw/lfw"


@pipeline
def benchmark_pipeline(model_name: str, max_images: int | None = None):
    """
    Preprocess LFW images and evaluate one model candidate.

    Steps:
        1. preprocess — align and normalize all LFW images (cached after first run)
        2. evaluate   — compute TAR@FAR and AUC, log to MLflow
    """
    aligned_path = preprocess(data_path=LFW_DATA_PATH, max_images=max_images)
    evaluate(
        model_name=model_name,
        preprocessed_data_path=aligned_path,
        pairs_path=f"{LFW_DATA_PATH}/pairs.txt",
        max_pairs=max_images,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark face recognition models on LFW")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=ALL_CANDIDATES, help="Single model to benchmark")
    group.add_argument("--all", action="store_true", help="Benchmark all candidates")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images to preprocess (default: all)")
    args = parser.parse_args()

    candidates = ALL_CANDIDATES if args.all else [args.model]

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    for model_name in candidates:
        print(f"\n{'='*50}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*50}")
        benchmark_pipeline(model_name=model_name, max_images=args.max_images)
