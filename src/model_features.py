"""Pipelines, hiperparámetros y métricas para los escenarios con features.

Se mantiene separado de ``src/model_*.py`` para no alterar los modelos
baseline. Todos los pipelines comparten el preprocesador de
``src/preprocessing_features.py`` y la misma semilla.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import mord
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .preprocessing_features import construir_preprocesador_features
from .utils import CLASES, SEED, evaluar_modelo

#: Mejores hiperparámetros encontrados en la baseline (validación).
PARAMS_BASE = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
    },
    "xgboost": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
    },
    "lightgbm": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "num_leaves": 63,
    },
    "ordinal_logistic": {"alpha": 0.1},
}

#: Configuración razonable para el escenario extendido. El tuning posterior
#: puede reemplazarla; estos valores regularizan un poco más que la baseline.
PARAMS_EXT = {
    "random_forest": {
        "n_estimators": 300,
        "max_depth": None,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
    },
    "xgboost": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_lambda": 1.0,
    },
    "lightgbm": {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    "ordinal_logistic": {"alpha": 0.1},
}

#: Distribución para ``RandomizedSearchCV`` sobre LightGBM extendido.
GRID_LGBM_EXT = {
    "modelo__n_estimators": [200, 300, 500, 700],
    "modelo__learning_rate": [0.03, 0.05, 0.08],
    "modelo__num_leaves": [31, 63, 127, 255],
    "modelo__min_child_samples": [10, 20, 40, 80],
    "modelo__subsample": [0.7, 0.8, 0.9, 1.0],
    "modelo__colsample_bytree": [0.6, 0.8, 1.0],
    "modelo__reg_lambda": [0.0, 2.0, 5.0, 10.0],
    "modelo__reg_alpha": [0.0, 1.0],
}


def construir_pipeline_features(
    estimador,
    X: pd.DataFrame,
    escalar: bool = False,
    imputar: bool = False,
    target_encoding: list[str] | None = None,
    min_frequency: float | None = None,
) -> Pipeline:
    """Envuelve un estimador con el preprocesador de features."""
    return Pipeline(
        steps=[
            (
                "preprocesador",
                construir_preprocesador_features(
                    X,
                    escalar=escalar,
                    imputar=imputar,
                    target_encoding=target_encoding,
                    min_frequency=min_frequency,
                ),
            ),
            ("modelo", estimador),
        ]
    )


def construir_rf_features(
    X: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int | None = None,
    max_features: str | float | None = "sqrt",
    min_samples_leaf: int = 1,
    min_samples_split: int = 2,
    class_weight: str | dict | None = None,
    imputar: bool = False,
    target_encoding: list[str] | None = None,
    min_frequency: float | None = None,
) -> Pipeline:
    """Random Forest con el preprocesador de features."""
    estimador = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=SEED,
        n_jobs=-1,
    )
    return construir_pipeline_features(
        estimador,
        X,
        escalar=False,
        imputar=imputar,
        target_encoding=target_encoding,
        min_frequency=min_frequency,
    )


def construir_xgb_features(
    X: pd.DataFrame,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    min_child_weight: float = 1.0,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    imputar: bool = False,
    target_encoding: list[str] | None = None,
    min_frequency: float | None = None,
) -> Pipeline:
    """XGBoost con el preprocesador de features (etiquetas enteras)."""
    estimador = XGBClassifier(
        objective="multi:softprob",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=-1,
    )
    return construir_pipeline_features(
        estimador,
        X,
        escalar=False,
        imputar=imputar,
        target_encoding=target_encoding,
        min_frequency=min_frequency,
    )


def construir_lgbm_features(
    X: pd.DataFrame,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    num_leaves: int = 63,
    min_child_samples: int = 20,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    reg_lambda: float = 0.0,
    reg_alpha: float = 0.0,
    class_weight: str | dict | None = None,
    imputar: bool = False,
    target_encoding: list[str] | None = None,
    min_frequency: float | None = None,
) -> Pipeline:
    """LightGBM con el preprocesador de features."""
    estimador = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        class_weight=class_weight,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    return construir_pipeline_features(
        estimador,
        X,
        escalar=False,
        imputar=imputar,
        target_encoding=target_encoding,
        min_frequency=min_frequency,
    )


def construir_ordinal_features(
    X: pd.DataFrame,
    alpha: float = 0.1,
    imputar: bool = False,
    target_encoding: list[str] | None = None,
    min_frequency: float | None = None,
) -> Pipeline:
    """Regresión Logística Ordinal (``mord.LogisticAT``) con escalado."""
    return construir_pipeline_features(
        mord.LogisticAT(alpha=alpha),
        X,
        escalar=True,
        imputar=imputar,
        target_encoding=target_encoding,
        min_frequency=min_frequency,
    )


def construir_modelo_features(
    nombre: str,
    X: pd.DataFrame,
    imputar: bool = False,
    target_encoding: list[str] | None = None,
    min_frequency: float | None = None,
    **params,
) -> Pipeline:
    """Dispatcher por nombre de modelo."""
    constructores = {
        "random_forest": construir_rf_features,
        "xgboost": construir_xgb_features,
        "lightgbm": construir_lgbm_features,
        "ordinal_logistic": construir_ordinal_features,
    }
    if nombre not in constructores:
        raise ValueError(
            f"Modelo desconocido: {nombre}. Opciones: {sorted(constructores)}"
        )
    return constructores[nombre](
        X,
        imputar=imputar,
        target_encoding=target_encoding,
        min_frequency=min_frequency,
        **params,
    )


def evaluar_modelo_ordinal(y_true, y_pred) -> dict:
    """Métricas de ``src.utils.evaluar_modelo`` más métricas ordinales.

    ``accuracy_adjacent`` mide el acierto exacto o en la clase vecina. Es la
    métrica adecuada para un target ordinal: un error ECONOMICO↔MEDIO es mucho
    menos grave que ECONOMICO↔PREMIUM. ``qwk`` es el kappa cuadrático ponderado.
    """
    metricas = evaluar_modelo(y_true, y_pred, etiquetas=CLASES)
    indice = {clase: i for i, clase in enumerate(CLASES)}
    yt = np.asarray([indice[v] for v in np.asarray(y_true)])
    yp = np.asarray([indice[v] for v in np.asarray(y_pred)])
    distancia = np.abs(yt - yp)
    metricas["accuracy_adjacent"] = float((distancia <= 1).mean())
    metricas["mae_ordinal"] = float(distancia.mean())
    metricas["qwk"] = float(
        cohen_kappa_score(yt, yp, labels=list(range(len(CLASES))), weights="quadratic")
    )
    return metricas


def metricas_a_fila(nombre: str, metricas: dict, **extra) -> dict:
    """Convierte el diccionario de métricas en una fila serializable."""
    fila = {
        "modelo": nombre,
        **extra,
        **{
            clave: valor
            for clave, valor in metricas.items()
            if clave != "matriz_confusion"
        },
    }
    return fila


def _paso_preprocesador(modelo: Pipeline):
    for nombre in ("preprocesador", "prep"):
        if nombre in modelo.named_steps:
            return modelo.named_steps[nombre]
    raise ValueError("El pipeline no contiene un paso de preprocesamiento.")


def importancia_variables(modelo: Pipeline) -> pd.DataFrame:
    """Extrae la importancia de variables de árboles o coeficientes lineales."""
    preprocesador = _paso_preprocesador(modelo)
    estimador = modelo.named_steps["modelo"]
    nombres = np.asarray(preprocesador.get_feature_names_out(), dtype=object)

    if hasattr(estimador, "feature_importances_"):
        importancias = np.asarray(estimador.feature_importances_, dtype=float)
    elif hasattr(estimador, "coef_"):
        coeficientes = np.asarray(estimador.coef_, dtype=float)
        if coeficientes.ndim == 1:
            importancias = np.abs(coeficientes)
        elif coeficientes.shape[0] == len(nombres):
            importancias = np.abs(coeficientes).mean(axis=1)
        elif coeficientes.shape[-1] == len(nombres):
            importancias = np.abs(coeficientes).mean(axis=0)
        else:
            importancias = np.abs(coeficientes).ravel()[: len(nombres)]
    else:
        raise ValueError("El modelo no expone feature_importances_ ni coef_.")

    if len(importancias) != len(nombres):
        raise ValueError(
            f"Desajuste: {len(importancias)} importancias vs {len(nombres)} variables."
        )

    return (
        pd.DataFrame({"variable": nombres, "importancia": importancias})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
