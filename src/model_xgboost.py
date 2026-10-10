"""Modelo XGBoost para la clasificación de rangos de precio en AirPrice."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .preprocessing import construir_preprocesador
from .utils import SEED


def construir_modelo_xgboost(
    X: pd.DataFrame,
    learning_rate: float = 0.1,
) -> Pipeline:
    """Construye el pipeline de preprocesamiento y clasificación XGBoost.

    El preprocesamiento se mantiene dentro del Pipeline para garantizar
    que el encoder se ajuste únicamente con los datos de entrenamiento.

    Args:
        X: matriz de características utilizada para identificar las
            columnas numéricas y categóricas.
        learning_rate: tasa de aprendizaje utilizada por XGBoost.

    Returns:
        Pipeline compuesto por el preprocesador y el clasificador XGBoost.
    """
    preprocesador = construir_preprocesador(
        X,
        escalar=False,
    )

    clasificador = XGBClassifier(
        objective="multi:softprob",
        random_state=SEED,
        learning_rate=learning_rate,
    )

    return Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("modelo", clasificador),
        ]
    )