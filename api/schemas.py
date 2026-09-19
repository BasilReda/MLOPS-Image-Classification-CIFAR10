from pydantic import BaseModel


class ClassProbability(BaseModel):
    class_name: str
    probability: float


class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    top3: list[ClassProbability]


class HealthResponse(BaseModel):
    status: str
