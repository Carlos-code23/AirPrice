"""Modelo Random Forest para la clasificación de rangos de precio en AirPrice."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .preprocessing import construir_preprocesador
from .utils import SEED


def construir_modelo_random_forest(
    X: pd.DataFrame,
    n_estimators: int = 300,
    max_depth: int | None = None,
    max_features: str | float | None = "sqrt",
    min_samples_leaf: int = 1,
    class_weight: str | dict | None = None,
) -> Pipeline:
    """Construye el pipeline de preprocesamiento y clasificación Random Forest.

    El preprocesamiento se mantiene dentro del Pipeline para garantizar que el
    encoder se ajuste únicamente con los datos de entrenamiento.

    Args:
        X: matriz de características para identificar columnas numéricas y categóricas.
        n_estimators: número de árboles.
        max_depth: profundidad máxima de los árboles.
        max_features: número de variables consideradas en cada división.
        min_samples_leaf: mínimo de muestras por hoja.
        class_weight: pesos de clase (``None`` si el dataset está balanceado).

    Returns:
        Pipeline compuesto por el preprocesador y el clasificador Random Forest.
    """
    preprocesador = construir_preprocesador(X, escalar=False)

    clasificador = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=SEED,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("modelo", clasificador),
        ]
    )
