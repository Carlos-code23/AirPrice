"""Modelo LightGBM para la clasificación de rangos de precio en AirPrice."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from .preprocessing import construir_preprocesador
from .utils import SEED


def construir_modelo_lightgbm(
    X: pd.DataFrame,
    num_leaves: int = 31,
) -> Pipeline:
    """Construye el pipeline de preprocesamiento y clasificación LightGBM.

    El preprocesamiento se mantiene dentro del Pipeline para garantizar
    que el encoder se ajuste únicamente con los datos de entrenamiento.

    Args:
        X: matriz de características.
        num_leaves: número máximo de hojas por árbol.

    Returns:
        Pipeline compuesto por el preprocesador y el clasificador LightGBM.
    """
    preprocesador = construir_preprocesador(
        X,
        escalar=False,
    )

    clasificador = LGBMClassifier(
        random_state=SEED,
        num_leaves=num_leaves,
    )

    return Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("modelo", clasificador),
        ]
    )