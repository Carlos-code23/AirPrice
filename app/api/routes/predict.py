"""Endpoints de predicción."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.deps import get_model_service
from app.core.config import settings
from app.core.rate_limit import limiter
from app.schemas.listing import BatchListingInput, ListingInput
from app.schemas.prediction import BatchPredictionResponse, PredictionResponse
from app.services.feature_service import InvalidInputError
from app.services.model_service import ModelService

router = APIRouter(prefix="/api/v1", tags=["Predictions"])


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid.uuid4()))


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predecir el rango de precio de un alojamiento",
    description=(
        "Recibe los campos clave del alojamiento, completa los campos "
        "faltantes con valores de entrenamiento y devuelve la clase ordinal "
        "y las probabilidades."
    ),
)
@limiter.limit(settings.rate_limit_single)
def predict(
    request: Request,
    payload: ListingInput,
    service: ModelService = Depends(get_model_service),
) -> PredictionResponse:
    try:
        prediccion, warnings = service.predict(payload.model_dump())
    except InvalidInputError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    return PredictionResponse(
        request_id=_request_id(request),
        model=service.model_info(),
        prediction=prediccion,
        warnings=warnings,
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predecir un lote de alojamientos",
    description=(
        f"Procesa hasta {settings.max_batch_size} alojamientos por petición. "
        "Para lotes mayores, divide la solicitud."
    ),
)
@limiter.limit(settings.rate_limit_batch)
def predict_batch(
    request: Request,
    payload: BatchListingInput,
    service: ModelService = Depends(get_model_service),
) -> BatchPredictionResponse:
    if len(payload.listings) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"El lote supera el máximo de {settings.max_batch_size} "
                "alojamientos."
            ),
        )

    try:
        predicciones, warnings = service.predict_batch(
            [item.model_dump() for item in payload.listings]
        )
    except InvalidInputError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    return BatchPredictionResponse(
        request_id=_request_id(request),
        model=service.model_info(),
        predictions=predicciones,
        warnings=warnings,
    )
