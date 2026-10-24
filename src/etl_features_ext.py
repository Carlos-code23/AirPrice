"""ETL aditivo para construir ``airprice_features_ext.csv``.

Estrategia
----------
El dataset limpio se derivó de ``data/listings.csv`` y todos sus ``id`` están
presentes en el original. Por eso el escenario extendido se construye con un
**merge por id** (más barato y reproducible que reprocesar el CSV completo),
recuperando las columnas que el ETL baseline descartó.

El archivo limpio y ``airprice_features.csv`` no se modifican.

Fuga de datos
-------------
Nunca se recuperan ``price``, ``price_quote_*`` ni las estimaciones de ingresos
de Inside Airbnb. ``estimated_occupancy_l365d`` también se excluye por ser una
estimación derivada del histórico y no una característica observable al publicar
el anuncio.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree

from .features import distancia_haversine
from .utils import RAIZ, SEED, split_estratificado

LISTINGS_PATH = RAIZ / "data" / "listings.csv"
FEATURES_PATH = RAIZ / "data" / "processed" / "airprice_features.csv"
SALIDA_PATH = RAIZ / "data" / "processed" / "airprice_features_ext.csv"
AUDIT_PATH = RAIZ / "data" / "processed" / "airprice_features_ext_audit.csv"
METADATA_PATH = RAIZ / "data" / "processed" / "airprice_features_ext_metadata.json"

#: Columnas del original que no están en ``airprice_features.csv``.
COLUMNAS_ORIGINALES = [
    "id",
    "last_scraped",
    "host_since",
    "host_response_time",
    "host_listings_count",
    "host_total_listings_count",
    "host_verifications",
    "neighbourhood",
    "bathrooms_text",
    "amenities",
    "maximum_nights",
    "minimum_minimum_nights",
    "maximum_minimum_nights",
    "minimum_maximum_nights",
    "maximum_maximum_nights",
    "minimum_nights_avg_ntm",
    "maximum_nights_avg_ntm",
    "has_availability",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_eoy",
    "number_of_reviews_ltm",
    "number_of_reviews_l30d",
    "number_of_reviews_ly",
    "first_review",
    "last_review",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
]

#: Solo para auditoría: columnas que se excluyen explícitamente por fuga.
COLUMNAS_FUGA_ORIGINAL = [
    "price",
    "price_quote_checkin_date",
    "price_quote_checkout_date",
    "price_quote_total_price",
    "price_quote_price_per_night",
    "price_quote_raw",
    "estimated_revenue_l365d",
    "estimated_occupancy_l365d",
]

#: Amenidades con señal esperada sobre el precio. Se buscan como subcadenas
#: normalizadas dentro de la lista textual ``amenities``.
AMENITIES = {
    "wifi": ["wifi", "wireless internet"],
    "kitchen": ["kitchen"],
    "air_conditioning": ["air conditioning"],
    "heating": ["heating"],
    "washer": ["washer"],
    "dryer": ["dryer"],
    "elevator": ["elevator"],
    "gym": ["gym", "fitness center", "fitness centre"],
    "pool": ["pool"],
    "hot_tub": ["hot tub"],
    "parking": ["parking"],
    "free_parking": ["free parking"],
    "balcony": ["balcony"],
    "garden": ["garden"],
    "patio": ["patio"],
    "dishwasher": ["dishwasher"],
    "tv": ["tv"],
    "workspace": ["workspace", "dedicated workspace"],
    "crib": ["crib"],
    "breakfast": ["breakfast"],
    "sauna": ["sauna"],
    "rooftop": ["rooftop"],
    "view": ["view"],
    "waterfront": ["waterfront", "beach"],
    "fireplace": ["fireplace"],
    "doorman": ["doorman"],
    "concierge": ["concierge"],
    "self_checkin": ["self check-in", "self checkin", "self-check-in"],
    "smart_lock": ["smart lock"],
    "bathtub": ["bathtub"],
    "coffee_maker": ["coffee maker"],
    "oven": ["oven"],
    "stove": ["stove"],
    "refrigerator": ["refrigerator"],
    "microwave": ["microwave"],
    "hair_dryer": ["hair dryer"],
    "iron": ["iron"],
    "hangers": ["hangers"],
    "shampoo": ["shampoo"],
    "essentials": ["essentials"],
    "bed_linens": ["bed linens"],
    "extra_pillows": ["extra pillows"],
    "private_entrance": ["private entrance"],
    "private_pool": ["private pool"],
    "shared_pool": ["shared pool"],
    "long_term": ["long term stays"],
    "pets": ["pets allowed"],
    "smoking": ["smoking allowed"],
    "events": ["events allowed"],
}

#: Estaciones y hubs de transporte (lat, lon).
ESTACIONES = {
    "victoria": (51.4952, -0.1441),
    "waterloo": (51.5031, -0.1132),
    "liverpool_street": (51.5178, -0.0823),
    "london_bridge": (51.5055, -0.0860),
    "paddington": (51.5154, -0.1755),
    "euston": (51.5282, -0.1337),
    "kings_cross": (51.5308, -0.1238),
    "oxford_circus": (51.5150, -0.1419),
    "green_park": (51.5067, -0.1428),
    "bond_street": (51.5142, -0.1494),
    "bank": (51.5133, -0.0886),
    "aldgate": (51.5141, -0.0755),
    "tower_hill": (51.5100, -0.0766),
    "shoreditch_high_street": (51.5238, -0.0755),
    "bethnal_green": (51.5270, -0.0549),
    "stratford": (51.5414, -0.0033),
    "canary_wharf": (51.5054, -0.0235),
    "north_greenwich": (51.5000, 0.0000),
    "greenwich": (51.4826, 0.0077),
    "lewisham": (51.4613, -0.0130),
    "brixton": (51.4613, -0.1156),
    "clapham_common": (51.4646, -0.1382),
    "balham": (51.4431, -0.1525),
    "wimbledon": (51.4215, -0.2065),
    "richmond": (51.4630, -0.3010),
    "hammersmith": (51.4927, -0.2240),
    "shepherds_bush": (51.5040, -0.2180),
    "notting_hill_gate": (51.5090, -0.1960),
    "bayswater": (51.5121, -0.1879),
    "camden_town": (51.5390, -0.1426),
    "angel": (51.5322, -0.1058),
    "highbury": (51.5520, -0.0970),
    "finsbury_park": (51.5640, -0.1060),
    "seven_sisters": (51.5820, -0.0750),
    "tottenham_hale": (51.5880, -0.0730),
    "walthamstow": (51.5830, -0.0190),
    "ilford": (51.5590, 0.0700),
    "barking": (51.5397, 0.0810),
    "woolwich": (51.4910, 0.0700),
    "romford": (51.5754, 0.1860),
    "ealing_broadway": (51.5130, -0.3080),
    "acton_town": (51.5230, -0.2650),
    "wembley_park": (51.5560, -0.2795),
    "harrow_on_the_hill": (51.5790, -0.3350),
    "uxbridge": (51.5460, -0.4780),
    "kingston": (51.4123, -0.3000),
    "croydon": (51.3762, -0.0982),
}

#: Principales atractivos/centralidades.
LUGARES = {
    "centro": (51.5074, -0.1278),
    "buckingham_palace": (51.5014, -0.1419),
    "westminster": (51.5007, -0.1246),
    "trafalgar_square": (51.5080, -0.1281),
    "national_gallery": (51.5089, -0.1283),
    "british_museum": (51.5194, -0.1270),
    "st_pauls": (51.5138, -0.0984),
    "tower_of_london": (51.5081, -0.0759),
    "tower_bridge": (51.5055, -0.0754),
    "london_eye": (51.5033, -0.1196),
    "tate_modern": (51.5076, -0.0994),
    "the_shard": (51.5045, -0.0865),
    "borough_market": (51.5055, -0.0910),
    "covent_garden": (51.5129, -0.1243),
    "soho": (51.5137, -0.1337),
    "mayfair": (51.5100, -0.1470),
    "hyde_park": (51.5073, -0.1657),
    "regents_park": (51.5313, -0.1569),
    "green_park": (51.5048, -0.1650),
    "st_james_park": (51.5027, -0.1340),
    "battersea_park": (51.4790, -0.1580),
    "kew_gardens": (51.4787, -0.2956),
    "richmond_park": (51.4425, -0.2733),
    "hampstead_heath": (51.5608, -0.1630),
    "olympic_park": (51.5461, -0.0185),
    "o2_arena": (51.5030, 0.0032),
    "excel": (51.5079, 0.0290),
    "westfield_london": (51.5074, -0.2213),
    "westfield_stratford": (51.5433, -0.0067),
    "portobello_market": (51.5170, -0.2056),
    "brick_lane": (51.5215, -0.0717),
    "harrods": (51.4994, -0.1633),
    "selfridges": (51.5145, -0.1527),
    "royal_albert_hall": (51.5009, -0.1774),
    "victoria_and_albert": (51.4966, -0.1722),
    "natural_history_museum": (51.4967, -0.1764),
    "science_museum": (51.4978, -0.1745),
    "knightsbridge": (51.5015, -0.1600),
    "chelsea": (51.4875, -0.1680),
    "fulham": (51.4800, -0.1950),
    "notting_hill": (51.5090, -0.1960),
    "camden_market": (51.5414, -0.1460),
    "dalston": (51.5460, -0.0750),
    "peckham": (51.4739, -0.0690),
    "greenwich_market": (51.4813, -0.0090),
}

#: Aeropuertos que influyen en la demanda de alojamiento.
AEROPUERTOS = {
    "heathrow": (51.4700, -0.4543),
    "gatwick": (51.1537, -0.1821),
    "stansted": (51.8860, 0.2389),
    "luton": (51.8747, -0.3683),
    "city": (51.5048, 0.0495),
    "southend": (51.5714, 0.6956),
}


def _parse_bool(serie: pd.Series) -> pd.Series:
    return (
        serie.astype(str)
        .str.strip()
        .str.lower()
        .map({"t": 1, "true": 1, "1": 1, "f": 0, "false": 0, "0": 0})
        .astype("float")
    )


def _dias_desde(serie: pd.Series, referencia: pd.Series) -> pd.Series:
    fechas = pd.to_datetime(serie, errors="coerce")
    ref = pd.to_datetime(referencia, errors="coerce")
    return (ref - fechas).dt.days


def _contar_verificaciones(serie: pd.Series) -> pd.Series:
    texto = serie.fillna("").astype(str).str.strip("[]")

    def contar(valor: str) -> int:
        if not valor.strip():
            return 0
        return len([parte for parte in valor.split(",") if parte.strip()])

    return texto.apply(contar).astype("int16")


def _flags_amenities(serie: pd.Series) -> pd.DataFrame:
    texto = serie.fillna("").astype(str).str.lower()
    salida = pd.DataFrame(index=serie.index)
    for nombre, patrones in AMENITIES.items():
        patron = "|".join(re.escape(p) for p in patrones)
        salida[f"amenity_{nombre}"] = (
            texto.str.contains(patron, regex=True, na=False).astype("int8")
        )
    salida["amenities_disponibles"] = (texto.str.len() > 2).astype("int8")
    return salida


def _flags_bathrooms(serie: pd.Series) -> pd.DataFrame:
    texto = serie.fillna("").astype(str).str.lower()
    return pd.DataFrame(
        {
            "bathroom_private": texto.str.contains("private", na=False).astype("int8"),
            "bathroom_shared": texto.str.contains("shared", na=False).astype("int8"),
            "bathroom_half": texto.str.contains("half", na=False).astype("int8"),
        },
        index=serie.index,
    )


def _distancias_minimas(df: pd.DataFrame, puntos: dict) -> np.ndarray:
    distancias = [
        distancia_haversine(
            df["latitude"].to_numpy(),
            df["longitude"].to_numpy(),
            lat,
            lon,
        )
        for lat, lon in puntos.values()
    ]
    return np.min(np.vstack(distancias), axis=0)


def _agregar_geo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    todos_los_pois = {**LUGARES, **ESTACIONES, **AEROPUERTOS}
    df["distancia_min_poi_km"] = _distancias_minimas(df, todos_los_pois)
    df["distancia_metro_km"] = _distancias_minimas(df, ESTACIONES)
    df["distancia_landmark_km"] = _distancias_minimas(df, LUGARES)
    df["distancia_aeropuerto_km"] = _distancias_minimas(df, AEROPUERTOS)

    puntos_cercanos = {**LUGARES, **ESTACIONES}
    distancias = np.vstack(
        [
            distancia_haversine(
                df["latitude"].to_numpy(),
                df["longitude"].to_numpy(),
                lat,
                lon,
            )
            for lat, lon in puntos_cercanos.values()
        ]
    )
    df["pois_2km"] = (distancias <= 2.0).sum(axis=0).astype("int16")

    return df


def _agregar_densidad(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    coordenadas = np.radians(df[["latitude", "longitude"]].to_numpy())
    arbol = BallTree(coordenadas, metric="haversine")

    # Radio de 1 km en radianes (radio terrestre = 6371 km).
    conteos = arbol.query_radius(coordenadas, r=1.0 / 6371.0, count_only=True)
    df["vecinos_1km"] = np.asarray(conteos, dtype=int) - 1

    distancias, _ = arbol.query(coordenadas, k=2)
    df["distancia_vecino_km"] = distancias[:, 1] * 6371.0

    df["densidad_vecindario"] = df.groupby("neighbourhood_cleansed")[
        "id"
    ].transform("count")
    if "cluster_zona" in df.columns:
        df["densidad_cluster"] = df.groupby("cluster_zona")["id"].transform("count")

    mediana = max(float(df["vecinos_1km"].median()), 1.0)
    df["ratio_densidad_1km"] = df["vecinos_1km"] / mediana
    return df


def _agregar_clusters(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Ajusta K-Means solo con las coordenadas de train y asigna a todos."""
    df = df.copy()
    X_train, _, _, _, _, _ = split_estratificado(
        df,
        target="price_range",
        excluir=["price_clean", "id"],
        seed=seed,
    )
    coords_train = X_train[["latitude", "longitude"]]
    for k in (5, 8, 10):
        modelo = KMeans(n_clusters=k, random_state=seed, n_init=10)
        modelo.fit(coords_train)
        df[f"cluster_zona_k{k}"] = modelo.predict(df[["latitude", "longitude"]])
    return df


def _agregar_features_extra(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    referencia = df["last_scraped"]

    df["antiguedad_host_anios"] = (
        _dias_desde(df["host_since"], referencia) / 365.25
    )
    df["dias_desde_primer_review"] = _dias_desde(df["first_review"], referencia)
    df["dias_desde_ultimo_review"] = _dias_desde(df["last_review"], referencia)
    df["host_verifications_count"] = _contar_verificaciones(df["host_verifications"])
    df["has_availability"] = _parse_bool(df["has_availability"])

    columnas_reviews = [
        "review_scores_accuracy",
        "review_scores_cleanliness",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_value",
    ]
    reviews = df[columnas_reviews].apply(pd.to_numeric, errors="coerce")
    df["review_scores_mean"] = reviews.mean(axis=1)
    df["review_scores_std"] = reviews.std(axis=1)
    df["review_scores_min"] = reviews.min(axis=1)
    df["review_scores_max"] = reviews.max(axis=1)
    df["review_scores_count"] = reviews.notna().sum(axis=1)

    for horizonte in (30, 60, 90):
        df[f"availability_ratio_{horizonte}"] = (
            df[f"availability_{horizonte}"] / horizonte
        )
    df["availability_mean"] = df[
        ["availability_30", "availability_60", "availability_90"]
    ].mean(axis=1)

    df["reviews_ltm_ratio"] = df["number_of_reviews_ltm"] / (
        df["number_of_reviews"] + 1
    )
    df["reviews_l30d_ratio"] = df["number_of_reviews_l30d"] / (
        df["number_of_reviews"] + 1
    )
    df["reviews_ly_ratio"] = df["number_of_reviews_ly"] / (
        df["number_of_reviews"] + 1
    )

    df["host_listings_ratio"] = df["host_listings_count"] / (
        df["host_total_listings_count"] + 1
    )
    df["entire_home_ratio"] = df[
        "calculated_host_listings_count_entire_homes"
    ] / (df["calculated_host_listings_count"] + 1)
    df["private_room_ratio"] = df[
        "calculated_host_listings_count_private_rooms"
    ] / (df["calculated_host_listings_count"] + 1)
    df["shared_room_ratio"] = df[
        "calculated_host_listings_count_shared_rooms"
    ] / (df["calculated_host_listings_count"] + 1)

    df["nights_range"] = df["maximum_nights"] - df["minimum_nights"]
    df["min_max_nights_ratio"] = df["minimum_nights"] / (df["maximum_nights"] + 1)

    df["accommodates_per_bedroom"] = df["accommodates"] / (df["bedrooms"] + 1)
    df["beds_per_bedroom"] = df["beds"] / (df["bedrooms"] + 1)
    df["bathrooms_per_bedroom"] = df["bathrooms_num"] / (df["bedrooms"] + 1)
    df["beds_per_guest"] = df["beds"] / (df["accommodates"] + 1)

    df = pd.concat(
        [df, _flags_amenities(df["amenities"]), _flags_bathrooms(df["bathrooms_text"])],
        axis=1,
    )
    return df


def _auditar(
    df_base: pd.DataFrame,
    df_ext: pd.DataFrame,
    ruta_audit: Path,
    ruta_metadata: Path,
) -> None:
    tasa = (
        float(df_ext["host_listings_count"].notna().mean())
        if "host_listings_count" in df_ext.columns
        else float("nan")
    )
    fila = pd.DataFrame(
        [
            {
                "filas_base": len(df_base),
                "filas_ext": len(df_ext),
                "columnas_base": df_base.shape[1],
                "columnas_ext": df_ext.shape[1],
                "columnas_nuevas": df_ext.shape[1] - df_base.shape[1],
                "tasa_emparejamiento_id": tasa,
                "columnas_originales_recuperadas": ";".join(COLUMNAS_ORIGINALES),
                "columnas_excluidas_por_fuga": ";".join(COLUMNAS_FUGA_ORIGINAL),
                "kmeans_k": "5;8;10",
                "n_amenities_flags": len(AMENITIES),
            }
        ]
    )
    fila.to_csv(ruta_audit, index=False)

    metadata = {
        "filas": int(len(df_ext)),
        "columnas": int(df_ext.shape[1]),
        "filas_base": int(len(df_base)),
        "columnas_base": int(df_base.shape[1]),
        "tasa_emparejamiento_id": tasa,
        "columnas_originales_recuperadas": COLUMNAS_ORIGINALES,
        "columnas_excluidas_por_fuga": COLUMNAS_FUGA_ORIGINAL,
        "kmeans_k": [5, 8, 10],
        "amenities_flags": list(AMENITIES),
        "nota": (
            "El merge es uno-a-uno por id. Las columnas de precio y las "
            "estimaciones de ingresos/ocupación de Inside Airbnb se excluyen "
            "para evitar fuga de datos."
        ),
    }
    with open(ruta_metadata, "w", encoding="utf-8") as archivo:
        json.dump(metadata, archivo, ensure_ascii=False, indent=2)


def construir_dataset_ext(
    ruta_listings: str | Path = LISTINGS_PATH,
    ruta_features: str | Path = FEATURES_PATH,
    ruta_salida: str | Path = SALIDA_PATH,
    ruta_audit: str | Path = AUDIT_PATH,
    ruta_metadata: str | Path = METADATA_PATH,
    seed: int = SEED,
) -> pd.DataFrame:
    """Construye y guarda ``airprice_features_ext.csv``.

    Returns:
        DataFrame extendido (también escrito en ``ruta_salida``).
    """
    base = pd.read_csv(ruta_features, low_memory=False)
    original = pd.read_csv(
        ruta_listings,
        usecols=lambda columna: columna in COLUMNAS_ORIGINALES,
        low_memory=False,
    )

    original = (
        original.sort_values("last_scraped", na_position="first")
        .drop_duplicates(subset="id", keep="last")
        .reset_index(drop=True)
    )

    df = base.merge(
        original,
        on="id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_orig"),
    )
    if len(df) != len(base):
        raise ValueError(
            f"El merge cambió el número de filas: {len(base)} → {len(df)}."
        )
    if not df["id"].equals(base["id"]):
        raise ValueError(
            "El merge alteró el orden de id; el split dejaría de ser comparable."
        )

    df = _agregar_features_extra(df)
    df = _agregar_geo(df)
    df = _agregar_densidad(df)
    df = _agregar_clusters(df, seed=seed)

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta_salida, index=False, encoding="utf-8")

    _auditar(base, df, Path(ruta_audit), Path(ruta_metadata))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construye el dataset extendido de AirPrice (aditivo)."
    )
    parser.add_argument("--listings", default=str(LISTINGS_PATH))
    parser.add_argument("--features", default=str(FEATURES_PATH))
    parser.add_argument("--salida", default=str(SALIDA_PATH))
    args = parser.parse_args()

    df = construir_dataset_ext(
        ruta_listings=args.listings,
        ruta_features=args.features,
        ruta_salida=args.salida,
    )
    print(f"Dataset extendido: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"Guardado en: {args.salida}")


if __name__ == "__main__":
    main()
