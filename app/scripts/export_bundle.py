"""Genera el bundle de inferencia a partir del modelo y del dataset extendido.

Uso:
    python -m app.scripts.export_bundle

El bundle contiene todo lo necesario para reconstruir las 151 features que
espera ``lightgbm_ext.joblib``: plantilla, medianas/moda, KMeans, BallTree de
densidad, POIs, esquema y métricas.
"""

from __future__ import annotations

import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.etl_features_ext import AEROPUERTOS, AMENITIES, ESTACIONES, LUGARES  # noqa: E402
from src.features import POIS as POIS_CINCO  # noqa: E402
from src.utils import CLASES, SEED, split_estratificado  # noqa: E402

EXT_PATH = RAIZ / "data" / "processed" / "airprice_features_ext.csv"
MODEL_PATH = RAIZ / "models" / "lightgbm_ext.joblib"
BUNDLE_PATH = RAIZ / "models" / "airprice_inference_bundle.joblib"
SCHEMA_PATH = RAIZ / "app" / "artifacts" / "feature_schema.json"
CARD_PATH = RAIZ / "app" / "artifacts" / "model_card.json"
METRICS_PATH = RAIZ / "reports" / "lightgbm_ext_metricas.csv"

CATEGORICAL_FEATURES = [
    "neighbourhood_cleansed",
    "property_type",
    "room_type",
    "host_is_superhost",
    "host_identity_verified",
    "instant_bookable",
]
CLUSTER_COLUMNS = {
    2: "cluster_zona",
    5: "cluster_zona_k5",
    8: "cluster_zona_k8",
    10: "cluster_zona_k10",
}
REQUIRED_FIELDS = [
    "room_type",
    "property_type",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "accommodates",
]
OPTIONAL_FIELDS = [
    "bedrooms",
    "beds",
    "bathrooms_num",
    "bathrooms_text",
    "minimum_nights",
    "maximum_nights",
    "host_listings_count",
    "calculated_host_listings_count",
    "amenities",
    "availability_365",
    "availability_30",
    "availability_60",
    "availability_90",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "number_of_reviews_l30d",
    "number_of_reviews_ly",
    "reviews_per_month",
    "review_scores_rating",
    "host_is_superhost",
    "host_identity_verified",
]


def _json_safe(valor):
    """Convierte tipos numpy/pandas a tipos serializables por JSON."""
    if isinstance(valor, dict):
        return {str(k): _json_safe(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_json_safe(v) for v in valor]
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating,)):
        return None if np.isnan(valor) else float(valor)
    if isinstance(valor, (np.bool_,)):
        return bool(valor)
    if isinstance(valor, float) and math.isnan(valor):
        return None
    return valor


def _default(serie: pd.Series, columna: str):
    """Mediana para continuas; moda para categóricas/binarias/clusters."""
    serie = serie.dropna()
    if serie.empty:
        return np.nan
    if (
        columna in CATEGORICAL_FEATURES
        or columna.startswith("cluster_zona")
        or columna.startswith("amenity_")
        or is_string_dtype(serie.dtype)
        or serie.nunique() <= 2
    ):
        moda = serie.mode()
        return moda.iloc[0] if not moda.empty else np.nan
    return serie.median()


def _metricas() -> dict:
    if not METRICS_PATH.exists():
        return {}
    fila = pd.read_csv(METRICS_PATH).iloc[0].to_dict()
    claves = [
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "accuracy_adjacent",
        "mae_ordinal",
        "qwk",
    ]
    return {
        clave: float(fila[clave])
        for clave in claves
        if clave in fila and pd.notna(fila[clave])
    }


def _ajustar_kmeans(df: pd.DataFrame, idx_train, feature_names: list[str]) -> dict:
    coords_train = df.loc[idx_train, ["latitude", "longitude"]].to_numpy()
    resultados = {}
    for k, columna in CLUSTER_COLUMNS.items():
        modelo = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(coords_train)
        asignados = modelo.labels_
        existentes = df.loc[idx_train, columna].to_numpy()
        mapping = {}
        for cluster in range(k):
            mascara = asignados == cluster
            if not mascara.any():
                mapping[int(cluster)] = int(cluster)
                continue
            valores, conteos = np.unique(existentes[mascara], return_counts=True)
            mapping[int(cluster)] = int(valores[np.argmax(conteos)])
        mapeados = np.array([mapping[int(c)] for c in asignados])
        fidelidad = float((mapeados == existentes).mean())
        resultados[str(k)] = {
            "model": modelo,
            "mapping": mapping,
            "fidelidad_train": fidelidad,
            "feature": columna,
        }
        print(f"  KMeans k={k}: fidelidad de etiquetas en train = {fidelidad:.4f}")
    return resultados


def _densidad(df: pd.DataFrame) -> dict:
    coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    arbol = BallTree(coords, metric="haversine")
    conteos = arbol.query_radius(coords, r=1.0 / 6371.0, count_only=True)
    vecinos = np.asarray(conteos, dtype=int) - 1
    por_vecindario = df.groupby("neighbourhood_cleansed").size().to_dict()
    por_cluster = df.groupby("cluster_zona").size().to_dict()
    return {
        "balltree": arbol,
        "median_vecinos": float(np.median(vecinos)),
        "by_neighbourhood": {str(k): int(v) for k, v in por_vecindario.items()},
        "by_cluster": {int(k): int(v) for k, v in por_cluster.items()},
        "default_vecindario": float(np.median(list(por_vecindario.values()))),
        "default_cluster": float(np.median(list(por_cluster.values()))),
    }


def _pois() -> dict:
    return {
        "cinco": POIS_CINCO,
        "estaciones": ESTACIONES,
        "lugares": LUGARES,
        "aeropuertos": AEROPUERTOS,
        "todos": {**LUGARES, **ESTACIONES, **AEROPUERTOS},
        "cercanos": {**LUGARES, **ESTACIONES},
    }


def _schema(df: pd.DataFrame, feature_names: list[str], template: dict, metrics: dict) -> dict:
    return {
        "api_version": "1.0.0",
        "description": (
            "Campos clave para predecir el rango de precio. Los campos no "
            "enviados se completan con medianas/moda del entrenamiento."
        ),
        "required_fields": REQUIRED_FIELDS,
        "optional_fields": OPTIONAL_FIELDS,
        "advanced_fields": [
            "maximum_nights",
            "host_listings_count",
            "calculated_host_listings_count",
            "availability_30",
            "availability_60",
            "availability_90",
            "number_of_reviews_ltm",
            "number_of_reviews_l30d",
            "number_of_reviews_ly",
            "reviews_per_month",
            "bathrooms_text",
        ],
        "defaults": {
            **{
                campo: _json_safe(template.get(campo))
                for campo in OPTIONAL_FIELDS
            },
            "amenities": [],
            "host_is_superhost": "UNKNOWN",
            "host_identity_verified": "UNKNOWN",
            "bathrooms_text": None,
        },
        "allowed_values": {
            "room_type": sorted(df["room_type"].dropna().unique().tolist()),
            "property_type": sorted(df["property_type"].dropna().unique().tolist()),
            "neighbourhood_cleansed": sorted(
                df["neighbourhood_cleansed"].dropna().unique().tolist()
            ),
            "host_is_superhost": ["t", "f", "UNKNOWN"],
            "host_identity_verified": ["t", "f", "UNKNOWN"],
            "amenities": sorted(AMENITIES.keys()),
        },
        "bounds": {
            "latitude": {"min": 51.2, "max": 51.7},
            "longitude": {"min": -0.6, "max": 0.3},
            "accommodates": {"min": 1, "max": 16},
            "bedrooms": {"min": 0, "max": 20},
            "beds": {"min": 0, "max": 30},
            "bathrooms_num": {"min": 0, "max": 20},
            "minimum_nights": {"min": 1, "max": 365},
            "maximum_nights": {"min": 1, "max": 2000},
            "host_listings_count": {"min": 0, "max": 1000},
            "calculated_host_listings_count": {"min": 0, "max": 1000},
            "availability_365": {"min": 0, "max": 365},
            "availability_30": {"min": 0, "max": 30},
            "availability_60": {"min": 0, "max": 60},
            "availability_90": {"min": 0, "max": 90},
            "review_scores_rating": {"min": 0, "max": 5},
        },
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "full_feature_defaults": {
            columna: _json_safe(template.get(columna)) for columna in feature_names
        },
        "model_metrics": metrics,
    }


def _model_card(metrics: dict) -> dict:
    fecha = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).date().isoformat()
    return {
        "name": "lightgbm_ext",
        "version": "1.0.0",
        "trained_at": fecha,
        "target": "price_range",
        "classes": list(CLASES),
        "metrics": metrics,
        "key_fields_benchmark": None,
        "key_fields": REQUIRED_FIELDS + OPTIONAL_FIELDS,
        "intended_use": [
            "Segmentar alojamientos en rangos de precio.",
            "Priorizar anuncios para análisis comercial.",
            "Demostración de inferencia con LightGBM.",
        ],
        "not_intended_for": [
            "Fijar el precio exacto de un alojamiento.",
            "Sustituir una tasación profesional.",
            "Usar el resultado como verdad absoluta sin revisión humana.",
        ],
        "limitations": [
            "Accuracy exacta de 4 clases ~0.73; accuracy adyacente ~0.98.",
            "El target se construyó por cuantiles de precio.",
            "Algunos campos de la API se imputan con medianas/moda.",
            "El dataset no incluye imágenes ni texto de descripción.",
        ],
        "leakage_controls": [
            "No se usan price, price_clean, price_quote_* ni estimated_revenue_l365d.",
            "El target price_range no entra como feature.",
            "Los clusters y densidades se ajustan con train.",
        ],
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "lightgbm": __import__("lightgbm").__version__,
            "joblib": joblib.__version__,
        },
    }


def main() -> None:
    print("Cargando modelo y dataset extendido...")
    modelo = joblib.load(MODEL_PATH)
    df = pd.read_csv(EXT_PATH, low_memory=False)
    feature_names = list(modelo.named_steps["preprocesador"].feature_names_in_)
    faltantes = [c for c in feature_names if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el dataset extendido: {faltantes}")

    X_train, _, _, _, _, _ = split_estratificado(
        df, target="price_range", excluir=["price_clean", "id"], seed=SEED
    )
    idx_train = X_train.index

    print("Calculando defaults de entrenamiento...")
    train = df.loc[idx_train, feature_names]
    template = {columna: _default(train[columna], columna) for columna in feature_names}
    medians = {
        columna: float(train[columna].median())
        for columna in feature_names
        if pd.api.types.is_numeric_dtype(train[columna])
    }
    modes = {}
    for columna in feature_names:
        moda = train[columna].mode()
        modes[columna] = moda.iloc[0] if not moda.empty else np.nan

    print("Ajustando KMeans de inferencia...")
    kmeans = _ajustar_kmeans(df, idx_train, feature_names)

    print("Construyendo BallTree de densidad...")
    densidad = _densidad(df)

    print("Guardando bundle y artefactos...")
    metrics = _metricas()
    schema = _schema(df, feature_names, template, metrics)
    card = _model_card(metrics)
    bundle = {
        "model": modelo,
        "feature_names": feature_names,
        "template": template,
        "medians": medians,
        "modes": modes,
        "categorical_features": CATEGORICAL_FEATURES,
        "kmeans": kmeans,
        "density": densidad,
        "pois": _pois(),
        "feature_schema": schema,
        "model_card": card,
    }

    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    CARD_PATH.write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Bundle: {BUNDLE_PATH} ({BUNDLE_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"Schema: {SCHEMA_PATH}")
    print(f"Model card: {CARD_PATH}")
    print(f"Features: {len(feature_names)}")


if __name__ == "__main__":
    main()
