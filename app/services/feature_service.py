"""Construcción de las 151 features que espera el modelo extendido."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.etl_features_ext import AMENITIES
from src.features import distancia_haversine

RADIO_TIERRA_KM = 6371.0
REVIEW_SUBSCORES = [
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
]


class InvalidInputError(ValueError):
    """El payload contiene una categoría o valor no admitido."""


def _is_missing(valor) -> bool:
    if valor is None:
        return True
    try:
        return bool(pd.isna(valor))
    except (TypeError, ValueError):
        return False


class FeatureBuilder:
    """Reconstruye el vector de features a partir de campos clave.

    Parte de una fila plantilla (medianas/moda del entrenamiento), sobrescribe
    los campos enviados por el usuario y recalcula las features derivadas:
    distancias, clusters, densidad, amenidades, ratios y agregados de reviews.
    """

    def __init__(self, bundle: dict):
        self.feature_names: list[str] = list(bundle["feature_names"])
        self.template: dict = dict(bundle["template"])
        self.categorical_features: list[str] = list(bundle.get("categorical_features", []))
        self.kmeans: dict = bundle["kmeans"]
        self.density: dict = bundle["density"]
        self.pois: dict = bundle["pois"]
        self.schema: dict = bundle["feature_schema"]

    def _optional(self, payload: dict, clave: str, missing: list[str]):
        valor = payload.get(clave)
        if valor is None:
            missing.append(clave)
            return self.template.get(clave)
        return valor

    def _distancia_minima(self, lat: float, lon: float, puntos: dict) -> float:
        distancias = [
            float(distancia_haversine(lat, lon, plat, plon))
            for plat, plon in puntos.values()
        ]
        return min(distancias) if distancias else float("nan")

    def _distancias(self, lat: float, lon: float) -> dict:
        valores = {}
        for nombre, (plat, plon) in self.pois["cinco"].items():
            valores[f"distancia_{nombre}_km"] = float(
                distancia_haversine(lat, lon, plat, plon)
            )

        valores["distancia_min_poi_km"] = self._distancia_minima(
            lat, lon, self.pois["todos"]
        )
        valores["distancia_metro_km"] = self._distancia_minima(
            lat, lon, self.pois["estaciones"]
        )
        valores["distancia_landmark_km"] = self._distancia_minima(
            lat, lon, self.pois["lugares"]
        )
        valores["distancia_aeropuerto_km"] = self._distancia_minima(
            lat, lon, self.pois["aeropuertos"]
        )

        valores["pois_2km"] = int(
            sum(
                float(distancia_haversine(lat, lon, plat, plon)) <= 2.0
                for plat, plon in self.pois["cercanos"].values()
            )
        )
        return valores

    def _clusters(self, lat: float, lon: float) -> dict:
        valores = {}
        coordenada = np.array([[lat, lon]], dtype=float)
        for k in (2, 5, 8, 10):
            info = self.kmeans[str(k)]
            bruto = int(info["model"].predict(coordenada)[0])
            mapping = info["mapping"]
            etiqueta = int(mapping.get(bruto, mapping.get(str(bruto), bruto)))
            clave = "cluster_zona" if k == 2 else f"cluster_zona_k{k}"
            valores[clave] = etiqueta
        return valores

    def _densidad(
        self, lat: float, lon: float, neighbourhood: str, cluster: int
    ) -> dict:
        coordenada = np.radians(np.array([[lat, lon]], dtype=float))
        arbol = self.density["balltree"]

        conteo = int(
            arbol.query_radius(coordenada, r=1.0 / RADIO_TIERRA_KM, count_only=True)[0]
        )
        vecinos = max(conteo - 1, 0)
        distancia_vecino = float(arbol.query(coordenada, k=2)[0][0, 1] * RADIO_TIERRA_KM)

        por_vecindario = self.density["by_neighbourhood"]
        por_cluster = self.density["by_cluster"]
        densidad_vecindario = float(
            por_vecindario.get(str(neighbourhood), self.density["default_vecindario"])
        )
        densidad_cluster = float(
            por_cluster.get(
                int(cluster),
                por_cluster.get(str(cluster), self.density["default_cluster"]),
            )
        )
        mediana = float(self.density["median_vecinos"]) or 1.0

        return {
            "vecinos_1km": vecinos,
            "distancia_vecino_km": distancia_vecino,
            "densidad_vecindario": densidad_vecindario,
            "densidad_cluster": densidad_cluster,
            "ratio_densidad_1km": vecinos / mediana,
        }

    def _flags_amenities(self, amenities: list[str]) -> dict:
        texto = " ".join(amenities).lower()
        return {
            f"amenity_{nombre}": int(
                any(patron in texto for patron in patrones)
            )
            for nombre, patrones in AMENITIES.items()
        }

    def _bathrooms(self, values: dict, payload: dict) -> None:
        texto = payload.get("bathrooms_text")
        if not texto:
            return
        texto = str(texto).lower()
        values["bathroom_private"] = int("private" in texto)
        values["bathroom_shared"] = int("shared" in texto)
        values["bathroom_half"] = int("half" in texto)
        if payload.get("bathrooms_num") is None:
            numero = re.search(r"(\d+(?:\.\d+)?)", texto)
            if numero:
                values["bathrooms_num"] = float(numero.group(1))
            elif "half" in texto:
                values["bathrooms_num"] = 0.5

    def _host_portfolio(self, values: dict, payload: dict, missing: list[str]) -> None:
        host_count = payload.get("host_listings_count")
        if host_count is None:
            missing.append("host_listings_count")
            host_count = values.get("host_listings_count")
        else:
            values["host_listings_count"] = host_count

        calculado = payload.get("calculated_host_listings_count")
        if calculado is None:
            if _is_missing(host_count):
                missing.append("calculated_host_listings_count")
                calculado = values.get("calculated_host_listings_count")
            else:
                calculado = host_count

        if _is_missing(calculado):
            return

        calculado = int(calculado)
        values["calculated_host_listings_count"] = calculado
        room_type = str(values.get("room_type", ""))
        values["calculated_host_listings_count_entire_homes"] = (
            calculado if room_type == "Entire home/apt" else 0
        )
        values["calculated_host_listings_count_private_rooms"] = (
            calculado if room_type == "Private room" else 0
        )
        values["calculated_host_listings_count_shared_rooms"] = (
            calculado if room_type == "Shared room" else 0
        )
        denominador = float(calculado) + 1.0
        values["entire_home_ratio"] = (
            values["calculated_host_listings_count_entire_homes"] / denominador
        )
        values["private_room_ratio"] = (
            values["calculated_host_listings_count_private_rooms"] / denominador
        )
        values["shared_room_ratio"] = (
            values["calculated_host_listings_count_shared_rooms"] / denominador
        )

    def _review_scores(self, values: dict, warnings: list[str]) -> None:
        rating = values.get("review_scores_rating")
        if not _is_missing(rating):
            rating = float(rating)
            for columna in REVIEW_SUBSCORES:
                values[columna] = rating
            values["review_scores_mean"] = rating
            values["review_scores_std"] = 0.0
            values["review_scores_min"] = rating
            values["review_scores_max"] = rating
            values["review_scores_count"] = 6
            values["review_scores_rating_missing"] = 0
            values["has_reviews"] = 1
        else:
            warnings.append(
                "review_scores_rating no proporcionado; se usan los valores "
                "de la plantilla para los scores de reviews"
            )
            values["review_scores_rating_missing"] = int(
                _is_missing(values.get("review_scores_rating"))
            )

    def _ratios(self, values: dict, warnings: list[str], payload: dict) -> None:
        minimum = values.get("minimum_nights")
        maximum = values.get("maximum_nights")
        if not _is_missing(minimum) and not _is_missing(maximum):
            values["nights_range"] = max(float(maximum) - float(minimum), 0.0)
            values["min_max_nights_ratio"] = float(minimum) / (float(maximum) + 1.0)

        accommodates = float(values.get("accommodates") or 1)
        bedrooms = values.get("bedrooms")
        beds = values.get("beds")
        bathrooms = values.get("bathrooms_num")
        bedrooms_num = float(bedrooms) if not _is_missing(bedrooms) else 0.0

        values["accommodates_per_bedroom"] = accommodates / (bedrooms_num + 1.0)
        if not _is_missing(beds):
            values["beds_per_bedroom"] = float(beds) / (bedrooms_num + 1.0)
            values["beds_per_guest"] = float(beds) / (accommodates + 1.0)
        if not _is_missing(bathrooms):
            values["bathrooms_per_bedroom"] = float(bathrooms) / (bedrooms_num + 1.0)

        availability_365_usuario = payload.get("availability_365")
        disponibles = []
        for horizonte in (30, 60, 90):
            valor = values.get(f"availability_{horizonte}")
            if _is_missing(valor):
                if _is_missing(availability_365_usuario):
                    continue
                valor = min(float(availability_365_usuario), float(horizonte))
                warnings.append(
                    f"availability_{horizonte} no proporcionado; se estima "
                    f"a partir de availability_365"
                )
            else:
                valor = float(valor)
                if not _is_missing(availability_365_usuario):
                    valor = min(valor, float(availability_365_usuario))
            values[f"availability_{horizonte}"] = valor
            values[f"availability_ratio_{horizonte}"] = valor / horizonte
            disponibles.append(valor)

        if disponibles:
            values["availability_mean"] = float(np.mean(disponibles))
        if not _is_missing(availability_365_usuario):
            values["has_availability"] = int(float(availability_365_usuario) > 0)
        elif disponibles:
            values["has_availability"] = int(any(valor > 0 for valor in disponibles))

        number_reviews = values.get("number_of_reviews")
        if not _is_missing(number_reviews):
            number_reviews = float(number_reviews)
            for columna in (
                "number_of_reviews_ltm",
                "number_of_reviews_l30d",
                "number_of_reviews_ly",
            ):
                valor = values.get(columna)
                valor = 0.0 if _is_missing(valor) else min(
                    float(valor), number_reviews
                )
                values[columna] = valor
            values["reviews_ltm_ratio"] = values["number_of_reviews_ltm"] / (
                number_reviews + 1.0
            )
            values["reviews_l30d_ratio"] = values["number_of_reviews_l30d"] / (
                number_reviews + 1.0
            )
            values["reviews_ly_ratio"] = values["number_of_reviews_ly"] / (
                number_reviews + 1.0
            )
            if number_reviews <= 0:
                values["reviews_per_month"] = 0.0
                values["reviews_per_month_missing"] = 0
                values["has_reviews"] = 0
            else:
                values["has_reviews"] = max(int(values.get("has_reviews", 0)), 1)

    def build(self, payload: dict) -> tuple[pd.DataFrame, list[str]]:
        """Devuelve ``(DataFrame de una fila, warnings)``."""
        warnings: list[str] = []
        missing: list[str] = []

        property_type = str(payload["property_type"])
        neighbourhood = str(payload["neighbourhood_cleansed"])
        if property_type not in self.schema["allowed_values"]["property_type"]:
            raise InvalidInputError(
                "property_type no reconocido. Consulta /api/v1/schema."
            )
        if neighbourhood not in self.schema["allowed_values"]["neighbourhood_cleansed"]:
            raise InvalidInputError(
                "neighbourhood_cleansed no reconocido. Consulta /api/v1/schema."
            )

        values = dict(self.template)
        lat = float(payload["latitude"])
        lon = float(payload["longitude"])

        values.update(
            {
                "room_type": payload["room_type"],
                "property_type": property_type,
                "neighbourhood_cleansed": neighbourhood,
                "latitude": lat,
                "longitude": lon,
                "accommodates": int(payload["accommodates"]),
            }
        )

        for campo in (
            "bedrooms",
            "beds",
            "bathrooms_num",
            "bathrooms_text",
            "minimum_nights",
            "maximum_nights",
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
        ):
            values[campo] = self._optional(payload, campo, missing)

        values["host_is_superhost"] = payload.get("host_is_superhost", "UNKNOWN")
        values["host_identity_verified"] = payload.get(
            "host_identity_verified", "UNKNOWN"
        )
        self._bathrooms(values, payload)
        self._host_portfolio(values, payload, missing)

        amenities = payload.get("amenities") or []
        if amenities:
            values.update(self._flags_amenities(amenities))
            values["amenities_count"] = len(amenities)
            values["amenities_disponibles"] = 1
        else:
            warnings.append(
                "amenities no proporcionado; se usan los flags de la plantilla"
            )

        values.update(self._distancias(lat, lon))
        values.update(self._clusters(lat, lon))
        values.update(
            self._densidad(lat, lon, neighbourhood, int(values["cluster_zona"]))
        )

        self._review_scores(values, warnings)
        self._ratios(values, warnings, payload)

        bathrooms_provided = not _is_missing(payload.get("bathrooms_num")) or bool(
            payload.get("bathrooms_text")
        )
        values["bathrooms_num_missing"] = 0 if bathrooms_provided else 1
        values["bedrooms_missing"] = int(_is_missing(payload.get("bedrooms")))
        values["beds_missing"] = int(_is_missing(payload.get("beds")))
        values["minimum_nights_missing"] = int(
            _is_missing(payload.get("minimum_nights"))
        )

        if payload.get("reviews_per_month") is not None:
            values["reviews_per_month_missing"] = 0
        elif (
            not _is_missing(values.get("number_of_reviews"))
            and float(values["number_of_reviews"]) <= 0
        ):
            values["reviews_per_month_missing"] = 0
        else:
            values["reviews_per_month_missing"] = 1

        if missing:
            warnings.insert(
                0,
                "Campos no proporcionados; se usaron defaults de entrenamiento: "
                + ", ".join(sorted(set(missing))),
            )

        fila = {columna: values.get(columna, self.template.get(columna)) for columna in self.feature_names}
        dataframe = pd.DataFrame([fila], columns=self.feature_names)

        for columna in self.categorical_features:
            if columna in dataframe.columns:
                dataframe[columna] = (
                    dataframe[columna].fillna("UNKNOWN").astype(str)
                )

        return dataframe, warnings
