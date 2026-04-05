# attend-mlops

MLOps pipeline for [Attend.AI](https://github.com/naobyprawira/attend) — handles model lifecycle management and training pipelines for all Attend.AI models.

## Overview

This repo manages the full ML lifecycle: data versioning, experiment tracking, training, evaluation, model registration, and deployment to the Attend.AI backend.

## Stack

| Component | Tool |
| --- | --- |
| Pipeline orchestration | ZenML |
| Experiment tracking | MLflow (self-hosted) |
| Model registry | MLflow Model Registry (self-hosted) |
| Artifact storage | MinIO (self-hosted, S3-compatible) |
| Data versioning | DVC |
| Data validation | Great Expectations |
| Drift detection | Evidently AI |

## Project Structure

```text
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

```text
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

---

## Setup

Two scenarios depending on your team's setup.

### Scenario A — Services + Training on the same machine (solo / local)

Best for individual development or early experimentation.

#### 1. Start the services via Docker

```bash
cp .env.example .env
# Optional: change passwords in .env

docker compose up -d
```

Services running on `localhost`:

| Service | Port | Access |
| --- | --- | --- |
| MinIO S3 API | 9000 | used by DVC & MLflow |
| MinIO Console | 9001 | [localhost:9001](http://localhost:9001) |
| MLflow | 5000 | [localhost:5000](http://localhost:5000) |
| ZenML | 8237 | [localhost:8237](http://localhost:8237) |

#### 2. Install Python dependencies

```bash
pip install -e ".[dev]"
```

#### 3. Configure `.env`

`.env` points to `localhost` by default — no changes needed if the services run on the same machine.

#### 4. Connect ZenML to the local server

```bash
zenml init
zenml connect --url http://localhost:8237
zenml stack register attend-mlops -o default -a default --set
```

#### 5. Set up DVC remote *(once, then commit)*

```bash
dvc remote add -d minio s3://dvc-store
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin
git add .dvc/config && git commit -m "chore: configure DVC remote"
```

#### 6. Pull datasets

```bash
dvc pull
```

---

### Scenario B — Services on VPS, Training on local machines (team setup)

The services (MLflow, MinIO, ZenML) run on a shared VPS. Each developer trains on their own machine.

#### B.1 — Deploy the services to the VPS *(done once by the admin)*

```bash
# On the VPS
git clone <repo-url> attend-mlops
cd attend-mlops

cp .env.example .env
# REQUIRED: change all default passwords in .env before running

docker compose up -d
```

Open the following ports in the VPS firewall:

| Port | Service |
| --- | --- |
| 5000 | MLflow |
| 9000 | MinIO S3 API |
| 9001 | MinIO Console |
| 8237 | ZenML |

#### B.2 — Set up DVC remote *(done once by the admin, then commit)*

```bash
dvc remote add -d minio s3://dvc-store
dvc remote modify minio endpointurl http://<vps-ip>:9000
dvc remote modify minio access_key_id <MINIO_ROOT_USER>
dvc remote modify minio secret_access_key <MINIO_ROOT_PASSWORD>
git add .dvc/config && git commit -m "chore: configure DVC remote"
```

#### B.3 — Per-developer setup (local)

##### Install Python dependencies

```bash
pip install -e ".[dev]"
```

##### Configure `.env` with VPS credentials

```bash
cp .env.example .env
```

Edit `.env` with the VPS IP and credentials from the admin:

```env
MLFLOW_TRACKING_URI=http://<vps-ip>:5000
MLFLOW_S3_ENDPOINT_URL=http://<vps-ip>:9000
AWS_ACCESS_KEY_ID=<from-admin>
AWS_SECRET_ACCESS_KEY=<from-admin>
```

##### Connect ZenML to the VPS server

```bash
zenml init
zenml connect --url http://<vps-ip>:8237
zenml stack register attend-mlops -o default -a default --set
```

##### Pull datasets

```bash
dvc pull
```

---

## Running Pipelines

### Benchmark Pipeline

Benchmark candidate models (ArcFace, AdaFace, Facenet512) on LFW pairs to select the best baseline:

```bash
# Download the LFW dataset
python data/sources/download_lfw.py

# Benchmark a single model
python -m models.face_recognition.pipelines.benchmark_pipeline --model arcface_buffalo_l

# Benchmark all candidates
python -m models.face_recognition.pipelines.benchmark_pipeline --all
```

Results are logged to MLflow under the experiment `attend-face-recognition-benchmark`.

### Run Training Pipeline

```bash
python -m models.face_recognition.pipelines.training_pipeline
```

### Export Enrollment Data from Attend.AI

```bash
python data/sources/export_attend.py
```

Make sure `ATTEND_DB_PATH` and `ATTEND_PHOTOS_DIR` in `.env` point to the correct Attend.AI instance.

---

## Deployment to Attend.AI

Models are deployed via a pull-based mechanism:

1. Model is promoted in MLflow Model Registry with the `production` alias
2. Attend.AI backend is notified via webhook
3. Backend downloads the new ONNX artifact from MLflow (stored in MinIO) and hot-reloads without restart
4. Canary rollout: traffic is gradually shifted from the old model to the new one

---

## Adding a New Model

1. Create `models/<model_name>/` with the same structure as `face_recognition/`
2. Implement steps in `models/<model_name>/steps/`
3. Define the pipeline in `models/<model_name>/pipelines/`
4. Add configs in `models/<model_name>/configs/`
5. Register shared utilities in `core/` if needed
