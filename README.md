# attend-mlops

MLOps pipeline for [Attend.AI](https://github.com/naobyprawira/attend) — handles model lifecycle management and training pipelines for all Attend.AI models.

## Overview

This repo manages the full ML lifecycle: data versioning, experiment tracking, training, evaluation, model registration, and deployment to the Attend.AI backend.

## Stack

| Component | Tool |
|---|---|
| Pipeline orchestration | ZenML |
| Experiment tracking | MLflow (self-hosted) |
| Model registry | MLflow Model Registry (self-hosted) |
| Artifact storage | MinIO (self-hosted, S3-compatible) |
| Data versioning | DVC |
| Data validation | Great Expectations |
| Drift detection | Evidently AI |

## Project Structure

```
attend-mlops/
├── models/
│   └── face_recognition/
│       ├── pipelines/         # ZenML pipelines
│       ├── steps/             # Atomic pipeline steps
│       ├── configs/           # Hyperparameters & thresholds
│       ├── evaluation/        # Evaluation scripts
│       └── serving/           # Export & deployment logic
├── data/
│   ├── sources/               # Dataset download & export scripts
│   ├── processors/            # Preprocessing & augmentation
│   └── validation/            # Great Expectations suites
├── core/
│   ├── registry.py            # MLflow Model Registry wrapper
│   ├── monitoring.py          # Evidently AI drift detection
│   └── dataset.py             # DVC dataset management
├── notebooks/                 # Experimentation only, never production
├── configs/                   # Global ZenML stack config
├── tests/
├── dvc.yaml
└── pyproject.toml
```

## Training Pipeline

```
Export enrollment data (Attend.AI)
→ Data validation (Great Expectations)
→ Merge with public datasets (LFW, etc.)
→ Preprocessing & augmentation
→ Training / fine-tuning
→ Evaluation (TAR@FAR)
→ Compare with current production model
→ Pass: register to MLflow → notify Attend.AI (canary deploy)
→ Fail: log as failed experiment in MLflow
```

## Setup

```bash
# Install dependencies
pip install -e ".[dev]"

# Copy and fill in credentials
cp .env.example .env

# Initialize ZenML
zenml init
zenml stack register attend-mlops-local -o default -a default
zenml stack set attend-mlops-local

# Initialize DVC
dvc init
```

> MLflow and MinIO must be running and reachable before executing any pipeline.
> Set `MLFLOW_TRACKING_URI` and MinIO credentials in `.env`.

## Running the Benchmark Pipeline (Sprint 1)

Benchmark pre-trained model candidates on LFW pairs to select the best baseline:

```bash
# Download LFW dataset first
python data/sources/download_lfw.py

# Benchmark a single model
python -m models.face_recognition.pipelines.benchmark_pipeline --model arcface_buffalo_l

# Benchmark all candidates
python -m models.face_recognition.pipelines.benchmark_pipeline --all
```

Results are logged to MLflow under experiment `attend-face-recognition-benchmark`.

## Running the Training Pipeline

```bash
python -m models.face_recognition.pipelines.training_pipeline
```

## Data

Data is versioned with DVC. To pull datasets:

```bash
dvc pull
```

To export fresh enrollment data from Attend.AI:

```bash
python data/sources/export_attend.py
```

## Deployment

Models are deployed to Attend.AI via a pull-based mechanism:
1. Model promoted in MLflow Model Registry (`production` alias)
2. Attend.AI backend notified via webhook
3. Backend downloads new artifact from MLflow (stored in MinIO) and hot-reloads without restart
4. Canary rollout: gradual traffic shift from old model to new

## Adding a New Model

1. Create `models/<model_name>/` with the same structure as `face_recognition/`
2. Implement steps in `models/<model_name>/steps/`
3. Define pipeline in `models/<model_name>/pipelines/`
4. Add configs in `models/<model_name>/configs/`
5. Register shared utilities in `core/` if needed
