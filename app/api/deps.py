"""Dependencias compartidas de FastAPI."""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from app.services.model_service import ModelService


def get_model_service(request: Request) -> ModelService:
    """Devuelve el servicio cargado o responde 503."""
    service = getattr(request.app.state, "model_service", None)
    if service is None or not service.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible. Revisa /health.",
        )
    return service
