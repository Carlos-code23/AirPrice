"""Endpoint de salud."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Estado del servicio",
    description="Indica si el modelo está cargado y listo para predecir.",
)
def health(request: Request) -> dict:
    service = getattr(request.app.state, "model_service", None)
    cargado = bool(service and service.ready)
    return {
        "status": "ok" if cargado else "degraded",
        "model_loaded": cargado,
        "version": getattr(request.app.state, "api_version", "1.0.0"),
        "loaded_at": getattr(service, "loaded_at", None) if service else None,
    }
