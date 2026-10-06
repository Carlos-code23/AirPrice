"""Utilidades compartidas de AirPrice: semilla, rutas, split y métricas."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

SEED = 42

RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR = RAIZ / "data"
CLEAN_PATH = DATA_DIR / "processed" / "airprice_clean.csv"
MODELS_DIR = RAIZ / "models"
REPORTS_DIR = RAIZ / "reports"

CLASES = ["ECONOMICO", "MEDIO", "ALTO", "PREMIUM"]


def fijar_semilla(seed: int = SEED) -> None:
    """Fija la semilla global para que los experimentos sean reproducibles."""
    random.seed(seed)
    np.random.seed(seed)


def cargar_dataset(ruta: str | Path = CLEAN_PATH) -> pd.DataFrame:
    """Carga el dataset limpio desde disco."""
    return pd.read_csv(ruta, low_memory=False)


def split_estratificado(
    df: pd.DataFrame,
    target: str = "price_range",
    excluir: list[str] | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = SEED,
):
    """Divide el dataset en train/validación/test de forma estratificada.

    Args:
        df: dataset con la columna objetivo.
        target: columna objetivo ordinal.
        excluir: columnas extra a quitar de las features (p. ej. ``price_clean``
            para evitar fuga de datos).
        test_size: proporción del conjunto de prueba.
        val_size: proporción del conjunto de validación.
        seed: semilla de aleatoriedad.

    Returns:
        Tupla ``(X_train, X_val, X_test, y_train, y_val, y_test)``.
    """
    from sklearn.model_selection import train_test_split

    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no existe en el dataset.")

    columnas_excluir = {target, *(excluir or [])}
    X = df.drop(columns=[c for c in columnas_excluir if c in df.columns])
    y = df[target]

    fraccion_temp = test_size + val_size

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=fraccion_temp,
        stratify=y,
        random_state=seed,
    )

    proporcion_val = val_size / fraccion_temp

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - proporcion_val,
        stratify=y_temp,
        random_state=seed,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluar_modelo(y_true, y_pred, etiquetas: list[str] | None = None) -> dict:
    """Calcula las métricas robustas al desbalance exigidas por el proyecto.

    Returns:
        Diccionario con ``accuracy``, ``balanced_accuracy``, ``f1_macro``,
        ``f1_weighted`` y ``matriz_confusion``.
    """
    etiquetas = etiquetas or CLASES
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "matriz_confusion": confusion_matrix(y_true, y_pred, labels=etiquetas),
    }


def guardar_modelo(modelo, nombre: str, directorio: str | Path = MODELS_DIR) -> Path:
    """Guarda un modelo entrenado en disco con ``joblib``."""
    import joblib

    destino = Path(directorio)
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / f"{nombre}.joblib"
    joblib.dump(modelo, ruta)
    return ruta
