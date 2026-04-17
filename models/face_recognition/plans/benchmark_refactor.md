# Benchmark Pipeline Refactor Plan

Status: not started. Plan agreed 2026-04-17, implementation pending.

## Motivation

Current benchmark pipeline is not apple-to-apple across models. Three root causes:

1. **Shared ArcFace-style preprocessing forced on all models.** `preprocess.py` outputs 112×112 `(x−127.5)/128` BGR arrays for every image. This is wrong for FaceNet512 (trained on 160×160 with MTCNN + per-image prewhitening on VGGFace2) and approximately wrong for AdaFace (different norm constant, color order concerns).
2. **Double-preprocessing for InsightFace.** `evaluate.py` denormalizes the preprocessed float32 array back to uint8 and re-runs `FaceAnalysis.get()`, which re-detects + re-aligns. The preprocess step's work is thrown away for ArcFace candidates.
3. **Bugs:**
   - `"buffalo"` is not a valid InsightFace pack name (`evaluate.py:55`). Valid packs: `buffalo_l`, `buffalo_m`, `buffalo_s`, `buffalo_sc`, `antelopev2`. For ArcFace R100 the correct pack is `antelopev2` (ResNet100 @ Glint360K).
   - Docstring in `evaluate.py:9` and `benchmark_pipeline.py:7` says `buffalo_l` uses iResNet100. Wrong — it uses iResNet50 (`w600k_r50.onnx` @ WebFace600K). Commit `939c52a`-adjacent docstring edit flipped correct → incorrect; needs revert.

## Design principle

**Each model uses the preprocessing it was trained with.** Fair comparison measures each model at its training distribution optimum, not on an artificially shared pipeline. Input size, detector, landmark scheme, resize interpolation, color order, and normalization are all part of the model — benchmarking with mismatched preprocessing measures "model + wrong preprocessing," not the model.

| Model | Input | Detector | Normalization | API color (what we feed) | Internal color (model sees) |
|---|---|---|---|---|---|
| `arcface_buffalo_l` (iResNet50 @ WebFace600K) | 112×112 | SCRFD-10GF | (x−127.5)/127.5 | BGR | RGB (swapRB in blobFromImages) |
| `arcface_antelopev2` (iResNet100 @ Glint360K) | 112×112 | SCRFD-10GF | (x−127.5)/127.5 | BGR | RGB (swapRB in blobFromImages) |
| `adaface_ir50` (MS1MV3) | 112×112 | RetinaFace | (x/255−0.5)/0.5 | BGR | BGR |
| `facenet512` (VGGFace2) | **160×160** | MTCNN | per-image prewhitening | BGR | RGB (internal `[:, :, ::-1]`) |

**Color convention: every loader's `embed(image_path)` receives a BGR uint8 numpy array** (output of `cv2.imread`). Each library handles its own internal conversion. Never pass RGB directly to any loader — silent embedding distortion.

Dataset input: raw `lfw_funneled` JPGs (250×250, congealing-aligned with background). NOT pre-cropped — detect+align inside each model loader.

## Target architecture

Dataset-agnostic, 5-step pipeline. Adapter pattern lets us add CFP-FP / AgeDB-30 / CPLFW / CALFW / internal Attend.AI verification sets by adding an adapter, without touching the pipeline.

```
load_dataset (once)
    │
    ▼
(image_root, unique_images, pairs)
    │
    ▼  [fan-out per model]
extract_embeddings  →  compute_similarities  →  compute_metrics  →  log_mlflow
```

### Step contracts

| Step | Input | Output | Nature |
|---|---|---|---|
| `load_dataset` | `dataset_name: str` | `(image_root: str, unique_images: list[str], pairs: list[(str, str, bool)])` | IO + adapter dispatch |
| `extract_embeddings` | `(image_root, unique_images, model_name, cache_dir)` | `cache_path: str` | Model forward, cache-aware (disk) |
| `compute_similarities` | `(pairs, cache_path)` | `(sims: list[float], labels: list[int])` | Disk read + cosine |
| `compute_metrics` | `(sims, labels, model_name)` | `metrics: dict` | Pure |
| `log_mlflow` | `(metrics, model_name, dataset_name)` | `None` | Side effect |

### Module structure

```
models/face_recognition/
├── datasets/
│   ├── __init__.py
│   └── lfw.py                     # folder+pairs.txt → (image_root, unique_images, pairs)
├── loaders/                        # plain modules, not ZenML steps
│   ├── __init__.py
│   ├── base.py                    # ModelLoader protocol: load(), embed(image_path) → np.ndarray
│   ├── arcface.py                 # buffalo_l, antelopev2 via FaceAnalysis
│   ├── adaface.py                 # RetinaFace align + torch forward
│   └── facenet.py                 # DeepFace native MTCNN + 160×160
├── steps/
│   ├── load_dataset.py
│   ├── extract_embeddings.py
│   ├── compute_similarities.py
│   ├── compute_metrics.py
│   └── log_mlflow.py
└── pipelines/
    └── benchmark_pipeline.py      # wires steps; fan-out per model in ALL_CANDIDATES
```

### Persistent embedding cache

- Layout: `data/cache/embeddings/<model_name>/<rel_image_path>.npy`
- Key: `(model_name, relative_image_path)`. LFW paths are stable → relpath sufficient.
- On `extract_embeddings`: enumerate unique images → for each, check cache → miss runs full model pipeline → write npy.
- Reusable across runs. Add `--clear-cache` flag on pipeline if force recompute needed.

### Model loader interface (`loaders/base.py`)

```python
class ModelLoader(Protocol):
    name: str
    def load(self) -> None: ...
    def embed(self, image_path: str) -> np.ndarray | None: ...  # None if detect fails
```

Each loader owns its full detect→align→resize→normalize→forward chain. No shared preprocessing.

## Files to change

**Delete:**
- `models/face_recognition/steps/preprocess.py`

**Create:**
- `models/face_recognition/datasets/__init__.py`, `lfw.py`
- `models/face_recognition/loaders/__init__.py`, `base.py`, `arcface.py`, `adaface.py`, `facenet.py`
- `models/face_recognition/steps/load_dataset.py`
- `models/face_recognition/steps/extract_embeddings.py`
- `models/face_recognition/steps/compute_similarities.py`
- `models/face_recognition/steps/compute_metrics.py`
- `models/face_recognition/steps/log_mlflow.py`

**Rewrite:**
- `models/face_recognition/steps/evaluate.py` → remove (logic redistributed across new steps)
- `models/face_recognition/pipelines/benchmark_pipeline.py` → wire new step graph

**Edit:**
- `ALL_CANDIDATES` in `benchmark_pipeline.py`: rename `"arcface_r100"` → `"arcface_antelopev2"` (be honest about what the pack actually is).
- Revert docstring in `benchmark_pipeline.py:7` and the corresponding line in the rewritten evaluate: `buffalo_l` is iResNet50, not iResNet100.
- `dvc.yaml`: check whether `preprocess` has a stage; remove if so.

## Decisions locked in

- Drop shared preprocess entirely (no utility, no GE validation).
- Persist embedding cache to disk; reuse across runs.
- Dataset-agnostic via adapters; LFW is the first adapter. Verification-pair structure only (template-based like IJB-B/C is out of scope).
- Pipeline granularity: 5 steps (not 2), for observability + testability + swappable metrics/logging.

## Verified specs

- **AdaFace IR-50 (MS1MV3)** confirmed via AdaFace repo: 112×112, BGR, CHW float32, normalized `(x/255 − 0.5)/0.5` → `[-1, 1]`. Training data aligned with RetinaFace 5-pt → use RetinaFace for alignment (not MTCNN as in author's inference example — stricter match to training distribution).

## Open items when resuming

- Decide whether `extract_embeddings` ZenML step caching (by input hash) layers cleanly on top of our disk cache, or if we disable ZenML step caching here to avoid double-caching confusion.
- Add a `--clear-cache` flag: CLI arg on pipeline entrypoint, or pipeline config parameter.
