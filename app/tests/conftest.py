"""Fixtures compartidas de pytest."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="session")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def payload() -> dict:
    return {
        "room_type": "Entire home/apt",
        "property_type": "Entire rental unit",
        "neighbourhood_cleansed": "Westminster",
        "latitude": 51.5074,
        "longitude": -0.1278,
        "accommodates": 2,
        "bedrooms": 1,
        "beds": 1,
        "bathrooms_num": 1,
        "minimum_nights": 2,
        "amenities": ["Wifi", "Kitchen", "Heating", "Elevator"],
        "availability_365": 120,
        "number_of_reviews": 45,
        "review_scores_rating": 4.8,
        "host_is_superhost": "t",
        "host_identity_verified": "t",
    }
