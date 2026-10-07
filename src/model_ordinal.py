"""Regresión Logística Ordinal (mord.LogisticAT) para AirPrice."""

from __future__ import annotations

import mord
import numpy as np
from sklearn.pipeline import Pipeline

from .preprocessing import construir_preprocesador
from .utils import CLASES

A_NUMERO = {clase: i for i, clase in enumerate(CLASES)}  # ECONOMICO→0 ... PREMIUM→3


def codificar_y(y):
    """Convierte price_range a enteros ordenados."""
    return np.asarray([A_NUMERO[v] for v in y])


def decodificar_y(y_num):
    """Convierte las predicciones enteras de vuelta a nombres de clase."""
    return np.asarray([CLASES[int(i)] for i in y_num])


def construir_modelo_ordinal(X, alpha: float = 1.0) -> Pipeline:
    """Pipeline con escalado obligatorio + LogisticAT."""
    return Pipeline([
        ("prep", construir_preprocesador(X, escalar=True)),
        ("modelo", mord.LogisticAT(alpha=alpha)),
    ])


def entrenar(X_train, y_train, alpha: float = 1.0) -> Pipeline:
    modelo = construir_modelo_ordinal(X_train, alpha)
    modelo.fit(X_train, codificar_y(y_train))  # el scaler se ajusta solo con train
    return modelo


def predecir(modelo, X):
    return decodificar_y(modelo.predict(X))
