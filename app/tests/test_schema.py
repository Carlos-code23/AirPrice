"""Tests del endpoint de esquema."""

from __future__ import annotations


def test_schema(client):
    respuesta = client.get("/api/v1/schema")
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert "room_type" in cuerpo["required_fields"]
    assert "property_type" in cuerpo["allowed_values"]
    assert "Westminster" in cuerpo["allowed_values"]["neighbourhood_cleansed"]
    assert "amenities" in cuerpo["optional_fields"]
    assert cuerpo["n_features"] == 151
    assert "advanced_fields" in cuerpo
