"""Esquema de campos y valores válidos para el frontend."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_model_service
from app.services.model_service import ModelService

router = APIRouter(prefix="/api/v1", tags=["Schema"])


@router.get(
    "/schema",
    summary="Campos, defaults y valores válidos",
    description=(
        "Devuelve los campos clave que acepta `/predict`, sus defaults, "
        "las categorías válidas y el vocabulario de amenidades. El frontend "
        "de React puede usar esta respuesta para poblar sus formularios."
    ),
)
def feature_schema(service: ModelService = Depends(get_model_service)) -> dict:
    return service.schema
