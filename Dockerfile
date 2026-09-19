# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/
COPY params.yaml dvc.yaml ./
RUN pip install --no-cache-dir -e .

# ---- train: runs the full DVC pipeline ----
FROM base AS train
ENTRYPOINT ["dvc", "repro"]

# ---- serve: runs the FastAPI inference service ----
FROM base AS serve
COPY api/ api/
EXPOSE 8000
ENTRYPOINT ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
