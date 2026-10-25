"""Servicio de carga y predicción del modelo LightGBM extendido."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from src.utils import CLASES

from app.services.feature_service import FeatureBuilder

logger = logging.getLogger(__name__)


class ModelNotReadyError(RuntimeError):
    """El bundle no está cargado o es inválido."""


class ModelService:
    """Carga el bundle una sola vez y expone predicciones."""

    def __init__(self, bundle_path: Path, schema_path: Path, card_path: Path):
        self.bundle_path = Path(bundle_path)
        self.schema_path = Path(schema_path)
        self.card_path = Path(card_path)
        self.model = None
        self.builder: FeatureBuilder | None = None
        self.bundle: dict | None = None
        self.schema: dict = {}
        self.card: dict = {}
        self.loaded_at: str | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.builder is not None

    def load(self) -> None:
        """Carga el bundle y valida el contrato de features."""
        if not self.bundle_path.exists():
            raise ModelNotReadyError(
                f"No existe el bundle de inferencia: {self.bundle_path}. "
                "Ejecuta: python -m app.scripts.export_bundle"
            )

        bundle = joblib.load(self.bundle_path)
        for clave in ("model", "feature_names", "template", "kmeans", "density", "pois"):
            if clave not in bundle:
                raise ModelNotReadyError(f"El bundle no contiene la clave '{clave}'.")

        self.bundle = bundle
        self.model = bundle["model"]
        self.builder = FeatureBuilder(bundle)
        self.schema = (
            json.loads(self.schema_path.read_text(encoding="utf-8"))
            if self.schema_path.exists()
            else bundle.get("feature_schema", {})
        )
        self.card = (
            json.loads(self.card_path.read_text(encoding="utf-8"))
            if self.card_path.exists()
            else bundle.get("model_card", {})
        )
        self.loaded_at = datetime.now(timezone.utc).isoformat()
        logger.info(
            "Modelo cargado: %s features, versión %s",
            len(bundle["feature_names"]),
            self.card.get("version", "desconocida"),
        )

    def model_info(self) -> dict:
        if not self.ready:
            raise ModelNotReadyError("Modelo no disponible.")
        card = self.card or {}
        return {
            "name": card.get("name", "lightgbm_ext"),
            "version": card.get("version", "1.0.0"),
            "trained_at": card.get("trained_at", "desconocida"),
            "classes": list(CLASES),
            "n_features": len(self.builder.feature_names),
            "metrics": card.get("metrics", {}),
            "key_fields_benchmark": card.get("key_fields_benchmark"),
        }

    def predict(self, payload: dict) -> tuple[dict, list[str]]:
        if not self.ready:
            raise ModelNotReadyError("Modelo no disponible.")
        X, warnings = self.builder.build(payload)
        probabilidades = np.asarray(self.model.predict_proba(X))[0]

        clases_modelo = [str(clase) for clase in self.model.named_steps["modelo"].classes_]
        mapa = {clase: float(prob) for clase, prob in zip(clases_modelo, probabilidades)}
        ordenadas = {clase: float(mapa.get(clase, 0.0)) for clase in CLASES}
        etiqueta = max(ordenadas, key=ordenadas.get)
        confianza = float(ordenadas[etiqueta])

        if confianza < 0.50:
            warnings.append(
                "Confianza menor a 0.50: interpreta la predicción con cautela."
            )

        prediccion = {
            "label": etiqueta,
            "index": CLASES.index(etiqueta),
            "confidence": confianza,
            "probabilities": ordenadas,
        }
        return prediccion, warnings

    def predict_batch(self, payloads: list[dict]) -> tuple[list[dict], list[list[str]]]:
        predicciones = []
        advertencias = []
        for payload in payloads:
            prediccion, warnings = self.predict(payload)
            predicciones.append(prediccion)
            advertencias.append(warnings)
        return predicciones, advertencias
