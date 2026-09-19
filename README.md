# Knowledge Distillation MLOps Pipeline (CIFAR-10)

Reproducible, tracked, deployable pipeline for distilling a large CNN teacher
(`DeepNN`) into a small CNN student (`LightNN`) on CIFAR-10 — rebuilt from an
exploratory notebook into DVC-orchestrated stages with MLflow experiment
tracking, Docker packaging, CI, and a FastAPI serving endpoint.

## Project layout

```
src/kd_pipeline/   reusable, unit-tested logic (data, models, training, distillation, I/O)
scripts/           thin CLI entrypoints, one per DVC stage
api/                FastAPI serving app for the distilled student
frontend/           static HTML/CSS/JS UI, served by the API app itself
tests/              pytest unit tests (no real training)
notebooks/          exploratory notebook that calls into src/kd_pipeline
params.yaml         single source of truth for hyperparameters
dvc.yaml            pipeline stage definitions
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
```

## Run the pipeline

```powershell
dvc init                      # first time only
dvc remote add -d localcache <path-to-a-local-folder-outside-the-repo>
dvc repro                     # runs download_data -> train_teacher -> train_student_baseline -> distill_student -> evaluate
```

`dvc repro` re-runs only the stages whose code/params/deps changed (tracked via `dvc.lock`).

## Experiment tracking

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Each pipeline stage logs its params/metrics/artifacts to a local SQLite-backed
MLflow store; the distilled student is registered in the local Model Registry.
DVC owns pipeline reproducibility and the canonical checkpoint files under
`models/`; MLflow owns run comparison and the model registry — both reference
the same `.pt` files.

## Tests

```powershell
pytest tests/ -v
```

## Docker

```powershell
docker build --target train -t kd-train .
docker build --target serve -t kd-serve .
docker compose up train      # runs dvc repro in a container
docker compose up api        # serves the FastAPI app on :8000
```

## Serving API + frontend

```powershell
uvicorn api.main:app --reload
```

- `GET /health`
- `POST /predict` — multipart image upload, returns predicted class, confidence, and the full
  ranked probability distribution over all 10 classes.
- `GET /` — serves `frontend/index.html`, a static, dependency-free UI: drag-and-drop an image,
  see the predicted class with a confidence ring, a bar chart of all 10 class probabilities, and
  a plain-language explanation of *why* (confidence level + margin over the runner-up class,
  grounded in the actual returned probabilities — not a fabricated explanation). Same-origin, no
  build step, no CORS needed since the API serves it directly.

## Notes on the original notebook

`KD.ipynb` was the original, Colab-only, notebook implementation. It has been
superseded by this pipeline; `notebooks/KD_exploration.ipynb` is a trimmed
version that imports from `src/kd_pipeline` for interactive exploration
instead of duplicating training/distillation logic.
