"""Preprocesamiento aditivo para los escenarios geoespacial y extendido.

Este módulo NO reemplaza ``src/preprocessing.py``. Añade un ``ColumnTransformer``
que trata ``cluster_zona`` (y los clusters adicionales) como variables
categóricas, mantiene las distancias como numéricas y ofrece:

* imputación opcional (necesaria para el dataset extendido),
* escalado opcional (necesario para el modelo ordinal),
* target encoding con validación cruzada interna (sin fuga de información).

Las columnas derivadas del precio se excluyen siempre por seguridad.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocessing import (
    CATEGORICAS,
    COLUMNAS_DESCARTADAS,
    COLUMNA_PRECIO,
    TARGET,
)
from .utils import RAIZ, SEED

FEATURES_PATH = RAIZ / "data" / "processed" / "airprice_features.csv"
FEATURES_EXT_PATH = RAIZ / "data" / "processed" / "airprice_features_ext.csv"

#: Distancias generadas en ``src/features.py``.
COLUMNAS_GEO_NUMERICAS = [
    "distancia_centro_km",
    "distancia_heathrow_km",
    "distancia_kings_cross_km",
    "distancia_hyde_park_km",
    "distancia_canary_wharf_km",
]

#: Distancias y agregados generados por ``src/etl_features_ext.py``.
COLUMNAS_GEO_EXT = [
    "distancia_min_poi_km",
    "distancia_metro_km",
    "distancia_landmark_km",
    "distancia_aeropuerto_km",
    "pois_2km",
    "vecinos_1km",
    "distancia_vecino_km",
    "densidad_vecindario",
    "densidad_cluster",
    "ratio_densidad_1km",
]

#: ``cluster_zona`` es un entero en el CSV, pero semánticamente es categórico.
COLUMNAS_GEO_CATEGORICAS = [
    "cluster_zona",
    "cluster_zona_k5",
    "cluster_zona_k8",
    "cluster_zona_k10",
]

#: Categóricas que solo aparecen en el dataset extendido.
COLUMNAS_EXT_CATEGORICAS = [
    "host_response_time",
    "neighbourhood",
]

#: Columnas que nunca deben entrar al modelo por fuga de datos o por ser
#: identificadores/texto libre.
COLUMNAS_FUGA = [
    COLUMNA_PRECIO,
    TARGET,
    "price",
    "price_quote_checkin_date",
    "price_quote_checkout_date",
    "price_quote_total_price",
    "price_quote_price_per_night",
    "price_quote_raw",
    # ``estimated_revenue_l365d`` y ``estimated_occupancy_l365d`` son
    # estimaciones de Inside Airbnb construidas a partir de precio/ocupación.
    "estimated_revenue_l365d",
    "estimated_occupancy_l365d",
    "id",
    "scrape_id",
    "listing_url",
    "picture_url",
    "host_url",
    "host_profile_url",
    "host_thumbnail_url",
    "host_picture_url",
    "last_scraped",
    "calendar_last_scraped",
    "calendar_updated",
]

#: Columnas de texto/fecha que se transforman o descartan en el ETL extendido.
COLUMNAS_NO_MODELABLES = [
    "amenities",
    "host_since",
    "first_review",
    "last_review",
    "host_verifications",
    "bathrooms_text",
    "host_neighbourhood",
    "license",
    "name",
    "description",
    "neighborhood_overview",
]


def cargar_features(ruta: str | Path = FEATURES_PATH) -> pd.DataFrame:
    """Carga el dataset con features geoespaciales."""
    return pd.read_csv(ruta, low_memory=False)


def preparar_dataset_features(
    df: pd.DataFrame,
    descartar_baseline: bool = True,
) -> pd.DataFrame:
    """Limpia columnas de fuga y, opcionalmente, las descartadas en la baseline.

    ``descartar_baseline=True`` garantiza que la comparación base vs geo use
    exactamente las mismas variables base (más las geo). Para el escenario
    extendido se puede usar ``False`` para recuperar variables descartadas en
    el EDA baseline.

    Args:
        df: dataset crudo o enriquecido.
        descartar_baseline: si es ``True`` elimina ``COLUMNAS_DESCARTADAS``.

    Returns:
        Copia del dataframe lista para el split.
    """
    df = df.copy()
    # El target se conserva: ``split_estratificado`` lo necesita para estratificar.
    a_eliminar = {
        c
        for c in COLUMNAS_FUGA + COLUMNAS_NO_MODELABLES
        if c in df.columns and c != TARGET
    }
    if descartar_baseline:
        a_eliminar.update(c for c in COLUMNAS_DESCARTADAS if c in df.columns)
    df = df.drop(columns=sorted(a_eliminar))
    # Columnas 100 % vacías en este scrape (p. ej. host_since o
    # host_total_listings_count) no aportan y harían fallar/avisar al imputador.
    return df.dropna(axis=1, how="all")


def columnas_geo(X: pd.DataFrame) -> list[str]:
    """Devuelve las columnas geoespaciales presentes en ``X``."""
    candidatas = COLUMNAS_GEO_NUMERICAS + COLUMNAS_GEO_EXT + COLUMNAS_GEO_CATEGORICAS
    return [c for c in candidatas if c in X.columns]


def columnas_categoricas_features(X: pd.DataFrame) -> list[str]:
    """Detecta categóricas incluyendo explícitamente los clusters.

    No se apoya solo en ``dtype == object`` porque ``cluster_zona`` llega como
    entero desde el CSV y debe codificarse con One-Hot.
    """
    explicitas = list(CATEGORICAS) + COLUMNAS_EXT_CATEGORICAS + COLUMNAS_GEO_CATEGORICAS
    categoricas = [c for c in explicitas if c in X.columns]
    for columna in X.columns:
        if columna in categoricas:
            continue
        dtype = X[columna].dtype
        # pandas 3 usa dtype ``str`` para texto, no ``object``.
        if is_object_dtype(dtype) or is_string_dtype(dtype):
            categoricas.append(columna)
        elif isinstance(dtype, pd.CategoricalDtype):
            categoricas.append(columna)
    # ``dict.fromkeys`` conserva el orden y elimina duplicados.
    return list(dict.fromkeys(categoricas))


class TargetEncoderCV(BaseEstimator, TransformerMixin):
    """Target encoding multiclase con validación cruzada interna.

    ``fit_transform`` devuelve codificaciones *out-of-fold* para el conjunto de
    entrenamiento, de modo que el estimador nunca ve la media del target de su
    propia fila. ``transform`` usa el mapeo completo del train para validación
    y test. La suavización evita sobreajuste en categorías poco frecuentes.

    Args:
        smoothing: fuerza de la suavización hacia el prior de clase.
        n_splits: número de folds para las codificaciones out-of-fold.
        seed: semilla de los folds.
    """

    def __init__(self, smoothing: float = 10.0, n_splits: int = 5, seed: int = SEED):
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.seed = seed

    @staticmethod
    def _como_dataframe(X, columnas=None) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(np.asarray(X), columns=columnas)

    @staticmethod
    def _categorias(serie: pd.Series) -> pd.Series:
        return serie.astype(object).where(serie.notna(), "__NA__").astype(str)

    def _mapeo(self, serie: pd.Series, y: np.ndarray) -> dict:
        categorias = self._categorias(serie)
        prior = np.array([(y == clase).mean() for clase in self.classes_], dtype=float)
        indicadores = pd.DataFrame({"categoria": categorias})
        for i, clase in enumerate(self.classes_):
            indicadores[f"y{i}"] = (y == clase).astype(float)
        sumas = indicadores.groupby("categoria", observed=True).sum()
        conteos = indicadores.groupby("categoria", observed=True).size()
        mapeo = {}
        for categoria, fila in sumas.iterrows():
            n = float(conteos.loc[categoria])
            mapeo[categoria] = (fila.to_numpy(dtype=float) + prior * self.smoothing) / (
                n + self.smoothing
            )
        return {"prior": prior, "mapa": mapeo}

    def _aplicar(self, mapeo: dict, serie: pd.Series) -> np.ndarray:
        categorias = self._categorias(serie).to_numpy()
        prior = mapeo["prior"]
        return np.vstack([mapeo["mapa"].get(cat, prior) for cat in categorias])

    def fit(self, X, y):
        X = self._como_dataframe(X)
        y = np.asarray(y)
        self.feature_names_in_ = list(X.columns)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.mapeos_ = {col: self._mapeo(X[col], y) for col in self.feature_names_in_}
        self.feature_names_out_ = [
            f"{col}__{clase}" for col in self.feature_names_in_ for clase in self.classes_
        ]
        return self

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            raise ValueError("TargetEncoderCV requiere y para codificar el target.")
        self.fit(X, y)
        X = self._como_dataframe(X)
        y = np.asarray(y)
        n_clases = len(self.classes_)
        salida = np.zeros((len(X), len(self.feature_names_out_)), dtype=float)

        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        for train_idx, val_idx in kfold.split(X):
            for j, columna in enumerate(self.feature_names_in_):
                mapeo_fold = self._mapeo(X.iloc[train_idx][columna], y[train_idx])
                inicio = j * n_clases
                salida[val_idx, inicio : inicio + n_clases] = self._aplicar(
                    mapeo_fold, X.iloc[val_idx][columna]
                )
        return salida

    def transform(self, X):
        X = self._como_dataframe(X, getattr(self, "feature_names_in_", None))
        X = X[self.feature_names_in_]
        return np.hstack([self._aplicar(self.mapeos_[col], X[col]) for col in X.columns])

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)


def construir_preprocesador_features(
    X: pd.DataFrame,
    escalar: bool = False,
    imputar: bool = False,
    target_encoding: list[str] | None = None,
    min_frequency: float | None = None,
) -> ColumnTransformer:
    """Construye el ``ColumnTransformer`` para los escenarios con features.

    Args:
        X: matriz de features.
        escalar: aplica ``StandardScaler`` a las numéricas (modelo ordinal).
        imputar: imputa numéricas con la mediana y agrega indicadores de faltante
            (necesario en el dataset extendido).
        target_encoding: columnas categóricas que se codifican con
            :class:`TargetEncoderCV` en lugar de One-Hot.
        min_frequency: agrupa categorías raras de One-Hot si se indica.

    Returns:
        ``ColumnTransformer`` listo para insertarse en un ``Pipeline``.
    """
    target_encoding = list(target_encoding or [])
    # Puede haber columnas 100 % vacías que ``preparar_dataset_features``
    # elimina (p. ej. ``neighbourhood`` en este scrape). Se ignoran.
    target_encoding = [c for c in target_encoding if c in X.columns]
    categoricas = [
        c for c in columnas_categoricas_features(X) if c not in target_encoding
    ]
    numericas = [
        c for c in X.columns if c not in categoricas and c not in target_encoding
    ]

    if imputar and escalar:
        transformador_numerico = Pipeline(
            [
                ("imputador", SimpleImputer(strategy="median", add_indicator=True)),
                ("escalador", StandardScaler()),
            ]
        )
    elif imputar:
        transformador_numerico = SimpleImputer(strategy="median", add_indicator=True)
    elif escalar:
        transformador_numerico = StandardScaler()
    else:
        transformador_numerico = "passthrough"

    transformadores = [
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                min_frequency=min_frequency,
            ),
            categoricas,
        )
    ]
    if target_encoding:
        transformadores.append(("te", TargetEncoderCV(), target_encoding))
    transformadores.append(("num", transformador_numerico, numericas))

    return ColumnTransformer(
        transformers=transformadores,
        remainder="drop",
    )
