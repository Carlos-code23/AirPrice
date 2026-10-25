"""Configuración de la API mediante variables de entorno."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

RAIZ = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Configuración con prefijo ``AIRPRICE_``.

    Ejemplo de ``.env``::

        AIRPRICE_ALLOWED_ORIGINS=http://localhost:5173,https://midominio.com
        AIRPRICE_MODEL_BUNDLE_PATH=models/airprice_inference_bundle.joblib
    """

    model_config = SettingsConfigDict(
        env_prefix="AIRPRICE_",
        env_file=RAIZ / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AirPrice API"
    app_description: str = (
        "API pública para clasificar alojamientos de Airbnb en Londres "
        "en los rangos ECONOMICO, MEDIO, ALTO y PREMIUM."
    )
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"

    model_bundle_path: Path = RAIZ / "models" / "airprice_inference_bundle.joblib"
    feature_schema_path: Path = RAIZ / "app" / "artifacts" / "feature_schema.json"
    model_card_path: Path = RAIZ / "app" / "artifacts" / "model_card.json"

    allowed_origins: str = "http://localhost:5173,http://localhost:3000"
    max_batch_size: int = 50
    rate_limit_single: str = "30/minute"
    rate_limit_batch: str = "10/minute"
    log_level: str = "INFO"

    @property
    def cors_origins(self) -> list[str]:
        """Convierte la lista separada por comas en una lista de orígenes."""
        return [origen.strip() for origen in self.allowed_origins.split(",") if origen.strip()]


settings = Settings()
