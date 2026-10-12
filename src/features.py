"""Feature engineering geoespacial para AirPrice."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .utils import SEED

RADIO_TIERRA_KM = 6371.0

CENTRO_LONDRES = (51.5074, -0.1278)

POIS = {
    "centro": (51.5074, -0.1278),
    "heathrow": (51.4700, -0.4543),
    "kings_cross": (51.5308, -0.1238),
    "hyde_park": (51.5073, -0.1657),
    "canary_wharf": (51.5054, -0.0235),
}


def distancia_haversine(lat1, lon1, lat2, lon2):
    """Distancia en km entre dos puntos (o arreglos) con la fórmula de Haversine."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * RADIO_TIERRA_KM * np.arcsin(np.sqrt(a))


def distancia_centro(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega la distancia en km al centro de Londres."""
    df = df.copy()
    df["distancia_centro_km"] = distancia_haversine(
        df["latitude"], df["longitude"], *CENTRO_LONDRES
    )
    return df


def distancia_pois(df: pd.DataFrame, pois: dict | None = None) -> pd.DataFrame:
    """Agrega la distancia en km a cada POI (excepto el centro, ya cubierto)."""
    df = df.copy()
    pois = pois or POIS
    for nombre, (lat, lon) in pois.items():
        if nombre == "centro":
            continue
        df[f"distancia_{nombre}_km"] = distancia_haversine(
            df["latitude"], df["longitude"], lat, lon
        )
    return df


def evaluar_k(coords: pd.DataFrame, ks=range(2, 11), seed: int = SEED) -> pd.DataFrame:
    """Evalúa distintos valores de k con inercia (codo) y coeficiente de silueta."""
    resultados = []
    for k in ks:
        modelo = KMeans(n_clusters=k, random_state=seed, n_init=10)
        etiquetas = modelo.fit_predict(coords[["latitude", "longitude"]])
        resultados.append(
            {
                "k": k,
                "inercia": modelo.inertia_,
                "silueta": silhouette_score(
                    coords[["latitude", "longitude"]], etiquetas
                ),
            }
        )
    return pd.DataFrame(resultados)


def clustering_espacial(
    df: pd.DataFrame,
    coords_ajuste: pd.DataFrame,
    k: int = 8,
    seed: int = SEED,
):
    """Ajusta KMeans sobre ``coords_ajuste`` (train) y asigna ``cluster_zona``.

    Returns:
        Tupla ``(df_con_cluster, modelo_kmeans)``.
    """
    modelo = KMeans(n_clusters=k, random_state=seed, n_init=10)
    modelo.fit(coords_ajuste[["latitude", "longitude"]])

    df = df.copy()
    df["cluster_zona"] = modelo.predict(df[["latitude", "longitude"]])
    return df, modelo


def generar_features(
    df: pd.DataFrame,
    coords_ajuste: pd.DataFrame | None = None,
    k: int = 8,
    seed: int = SEED,
) -> pd.DataFrame:
    """Genera las features geoespaciales: distancias a centro/POIs y cluster."""
    df = distancia_centro(df)
    df = distancia_pois(df)
    coords_ajuste = coords_ajuste if coords_ajuste is not None else df
    df, _ = clustering_espacial(df, coords_ajuste, k=k, seed=seed)
    return df
