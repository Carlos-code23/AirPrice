"""Esquemas de respuesta de la API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Información reproducible del modelo servido."""

    name: str
    version: str
    trained_at: str
    classes: list[str]
    n_features: int
    metrics: dict[str, float] = Field(default_factory=dict)
    key_fields_benchmark: dict | None = None


class Prediction(BaseModel):
    """Predicción individual."""

    label: str = Field(..., description="ECONOMICO, MEDIO, ALTO o PREMIUM.")
    index: int = Field(..., ge=0, le=3, description="Índice ordinal de la clase.")
    confidence: float = Field(..., ge=0, le=1)
    probabilities: dict[str, float] = Field(
        ..., description="Probabilidad por clase; suma 1."
    )


class PredictionResponse(BaseModel):
    """Respuesta de ``POST /api/v1/predict``."""

    request_id: str
    model: ModelInfo
    prediction: Prediction
    warnings: list[str] = Field(default_factory=list)


class BatchPredictionResponse(BaseModel):
    """Respuesta de ``POST /api/v1/predict/batch``."""

    request_id: str
    model: ModelInfo
    predictions: list[Prediction]
    warnings: list[list[str]] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Error uniforme de la API."""

    detail: str
    request_id: str | None = None
