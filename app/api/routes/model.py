"""Información del modelo servido."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_model_service
from app.schemas.prediction import ModelInfo
from app.services.model_service import ModelService

router = APIRouter(prefix="/api/v1", tags=["Model"])


@router.get(
    "/model/info",
    response_model=ModelInfo,
    summary="Información y métricas del modelo",
    description=(
        "Devuelve el modelo servido, sus métricas de test y el benchmark "
        "con solo los campos clave de la API."
    ),
)
def model_info(service: ModelService = Depends(get_model_service)) -> dict:
    return service.model_info()
