from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import CLEAN_PATH

TARGET = "price_range"
COLUMNA_PRECIO = "price_clean"

COLUMNAS_DESCARTADAS = [
    "host_response_rate_num",
    "host_acceptance_rate_num",
    "instant_bookable",
    "reviews_per_month_missing",
    "review_scores_rating_missing",
    "host_response_rate_num_missing",
    "host_acceptance_rate_num_missing",
]

CATEGORICAS = [
    "room_type",
    "property_type",
    "neighbourhood_cleansed",
    "host_is_superhost",
    "host_identity_verified",
]


def cargar_datos(ruta: str | Path = CLEAN_PATH) -> pd.DataFrame:
    """Carga el dataset limpio desde disco."""
    return pd.read_csv(ruta, low_memory=False)


def descartar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas sin información o redundantes detectadas en el EDA."""
    columnas = [c for c in COLUMNAS_DESCARTADAS if c in df.columns]
    return df.drop(columns=columnas)


def construir_X_y(df: pd.DataFrame, target: str = TARGET):
    """Separa features y objetivo, evitando fuga de datos con ``price_clean``.

    Returns:
        Tupla ``(X, y)``.
    """
    df = descartar_columnas(df)
    excluir = [target, COLUMNA_PRECIO, "id"]
    X = df.drop(columns=[c for c in excluir if c in df.columns])
    y = df[target]
    return X, y


def columnas_categoricas(X: pd.DataFrame) -> list[str]:
    """Devuelve las columnas categóricas presentes en ``X``."""
    return [c for c in CATEGORICAS if c in X.columns]


def construir_preprocesador(
    X: pd.DataFrame, escalar: bool = False
) -> ColumnTransformer:
    """Construye el ``ColumnTransformer`` de codificación y escalado.

    Args:
        X: matriz de features.
        escalar: si es ``True`` aplica ``StandardScaler`` a las numéricas
            (necesario para modelos lineales/basados en distancia).
    """
    categoricas = columnas_categoricas(X)
    numericas = [c for c in X.columns if c not in categoricas]

    transformador_numerico = StandardScaler() if escalar else "passthrough"

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categoricas,
            ),
            ("num", transformador_numerico, numericas),
        ],
        remainder="drop",
    )


def pesos_clase(y) -> dict:
    """Calcula pesos balanceados por clase para usar en el entrenamiento."""
    from sklearn.utils.class_weight import compute_class_weight

    clases = np.unique(y)
    pesos = compute_class_weight(class_weight="balanced", classes=clases, y=y)
    return dict(zip(clases, pesos))
