"""Punto de entrada de la API AirPrice."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.routes import health, model, predict, schema
from app.core.config import settings
from app.core.logging import RequestIdMiddleware, configurar_logging
from app.core.rate_limit import limiter
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo una sola vez al arrancar."""
    configurar_logging(settings.log_level)
    servicio = ModelService(
        bundle_path=settings.model_bundle_path,
        schema_path=settings.feature_schema_path,
        card_path=settings.model_card_path,
    )
    servicio.load()
    app.state.model_service = servicio
    app.state.api_version = settings.api_version
    yield
    app.state.model_service = None


def create_app() -> FastAPI:
    """Factory de la aplicación FastAPI."""
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.api_version,
        lifespan=lifespan,
        openapi_url="/api/v1/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=[
            {"name": "Health", "description": "Estado del servicio."},
            {"name": "Model", "description": "Modelo y métricas."},
            {"name": "Schema", "description": "Campos y valores válidos."},
            {"name": "Predictions", "description": "Predicción de rangos de precio."},
        ],
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(model.router)
    app.include_router(schema.router)
    app.include_router(predict.router)

    @app.get("/", tags=["Health"], summary="Información del servicio")
    def root() -> dict:
        return {
            "service": settings.app_name,
            "version": settings.api_version,
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/api/v1/openapi.json",
            "health": "/health",
            "predict": "/api/v1/predict",
        }

    @app.exception_handler(Exception)
    async def error_no_controlado(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        logger.exception("Error no controlado [%s]", request_id)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Error interno del servidor.",
                "request_id": request_id,
            },
        )

    return app


app = create_app()
