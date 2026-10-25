"""Tests de predicción y validación."""

from __future__ import annotations

import copy


def test_predict(client, payload):
    respuesta = client.post("/api/v1/predict", json=payload)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["prediction"]["label"] in {
        "ECONOMICO",
        "MEDIO",
        "ALTO",
        "PREMIUM",
    }
    assert abs(sum(cuerpo["prediction"]["probabilities"].values()) - 1.0) < 1e-6
    assert "warnings" in cuerpo
    assert respuesta.headers.get("X-Request-ID")


def test_predict_con_campos_avanzados(client, payload):
    avanzado = copy.deepcopy(payload)
    avanzado.update(
        {
            "bathrooms_text": "1 private bath",
            "maximum_nights": 365,
            "host_listings_count": 12,
            "calculated_host_listings_count": 10,
            "availability_30": 20,
            "availability_60": 45,
            "availability_90": 70,
            "number_of_reviews_ltm": 18,
            "number_of_reviews_l30d": 2,
            "number_of_reviews_ly": 30,
            "reviews_per_month": 1.2,
        }
    )
    respuesta = client.post("/api/v1/predict", json=avanzado)
    assert respuesta.status_code == 200


def test_latitud_invalida(client, payload):
    invalido = copy.deepcopy(payload)
    invalido["latitude"] = 10.0
    respuesta = client.post("/api/v1/predict", json=invalido)
    assert respuesta.status_code == 422


def test_property_type_desconocido(client, payload):
    invalido = copy.deepcopy(payload)
    invalido["property_type"] = "Castillo flotante"
    respuesta = client.post("/api/v1/predict", json=invalido)
    assert respuesta.status_code == 422
    assert "property_type" in respuesta.json()["detail"]


def test_batch(client, payload):
    respuesta = client.post(
        "/api/v1/predict/batch", json={"listings": [payload, payload]}
    )
    assert respuesta.status_code == 200
    assert len(respuesta.json()["predictions"]) == 2


def test_batch_limite(client, payload):
    respuesta = client.post(
        "/api/v1/predict/batch", json={"listings": [payload] * 51}
    )
    assert respuesta.status_code == 422


def test_cors(client):
    respuesta = client.options(
        "/api/v1/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert respuesta.status_code == 200
    assert (
        respuesta.headers.get("access-control-allow-origin")
        == "http://localhost:5173"
    )


def test_feature_builder_shape(client, payload):
    builder = client.app.state.model_service.builder
    dataframe, warnings = builder.build(payload)
    assert dataframe.shape == (1, 151)
    assert isinstance(warnings, list)
    assert dataframe["cluster_zona"].notna().all()
