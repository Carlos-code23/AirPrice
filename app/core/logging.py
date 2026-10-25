"""Utilidades de logging y request-id."""

from __future__ import annotations

import logging
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


def configurar_logging(nivel: str = "INFO") -> None:
    """Configura un logging simple y consistente para la API."""
    logging.basicConfig(
        level=getattr(logging, nivel.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Añade un ``request_id`` a cada petición y a la respuesta."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
