"""Tests de salud e información del modelo."""

from __future__ import annotations


def test_root(client):
    respuesta = client.get("/")
    assert respuesta.status_code == 200
    assert respuesta.json()["predict"] == "/api/v1/predict"


def test_health(client):
    respuesta = client.get("/health")
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["status"] == "ok"
    assert cuerpo["model_loaded"] is True


def test_model_info(client):
    respuesta = client.get("/api/v1/model/info")
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["name"] == "lightgbm_ext"
    assert cuerpo["n_features"] == 151
    assert "accuracy" in cuerpo["metrics"]
    assert "key_fields" in cuerpo["key_fields_benchmark"]


def test_openapi(client):
    respuesta = client.get("/api/v1/openapi.json")
    assert respuesta.status_code == 200
    assert "paths" in respuesta.json()
