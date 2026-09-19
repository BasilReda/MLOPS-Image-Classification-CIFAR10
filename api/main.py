"""FastAPI serving app for the distilled student model."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from api.inference import Predictor
from api.schemas import HealthResponse, PredictionResponse

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["predictor"] = Predictor()
    yield
    _state.clear()


app = FastAPI(title="KD CIFAR-10 Student API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")

    image_bytes = await file.read()
    predictor: Predictor = _state["predictor"]
    result = predictor.predict(image_bytes)
    return PredictionResponse(**result)
