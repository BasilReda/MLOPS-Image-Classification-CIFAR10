from pydantic import BaseModel


class ClassProbability(BaseModel):
    class_name: str
    probability: float


class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    probabilities: list[ClassProbability]  # all classes, sorted descending by probability


class HealthResponse(BaseModel):
    status: str
