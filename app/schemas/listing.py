"""Esquema de entrada público de la API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RoomType = Literal["Entire home/apt", "Private room", "Shared room", "Hotel room"]
HostFlag = Literal["t", "f", "UNKNOWN"]


class ListingInput(BaseModel):
    """Datos mínimos de un alojamiento para predecir su rango de precio.

    Solo se piden los campos con mayor señal. El resto se completa con
    medianas/moda del entrenamiento y se informa en ``warnings``.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
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
            ]
        },
    )

    room_type: RoomType = Field(..., description="Tipo de habitación.")
    property_type: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="Tipo de propiedad; debe existir en /api/v1/schema.",
    )
    neighbourhood_cleansed: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="Barrio limpio de Londres; debe existir en /api/v1/schema.",
    )
    latitude: float = Field(..., ge=51.2, le=51.7, description="Latitud en Londres.")
    longitude: float = Field(..., ge=-0.6, le=0.3, description="Longitud en Londres.")
    accommodates: int = Field(..., ge=1, le=16, description="Número de huéspedes.")

    bedrooms: int | None = Field(None, ge=0, le=20, description="Dormitorios.")
    beds: int | None = Field(None, ge=0, le=30, description="Camas.")
    bathrooms_num: float | None = Field(None, ge=0, le=20, description="Baños.")
    bathrooms_text: str | None = Field(
        None,
        max_length=120,
        description="Texto de baños, p. ej. '1 shared bath' o 'Half-bath'.",
    )
    minimum_nights: int | None = Field(None, ge=1, le=365, description="Noches mínimas.")
    maximum_nights: int | None = Field(
        None, ge=1, le=2000, description="Noches máximas."
    )
    host_listings_count: int | None = Field(
        None, ge=0, le=1000, description="Anuncios del host."
    )
    calculated_host_listings_count: int | None = Field(
        None, ge=0, le=1000, description="Anuncios calculados del host."
    )

    amenities: list[str] = Field(
        default_factory=list,
        max_length=100,
        description="Lista de amenidades, p. ej. ['Wifi', 'Kitchen', 'Pool'].",
    )
    availability_365: int | None = Field(
        None, ge=0, le=365, description="Días disponibles en los próximos 365."
    )
    availability_30: int | None = Field(None, ge=0, le=30)
    availability_60: int | None = Field(None, ge=0, le=60)
    availability_90: int | None = Field(None, ge=0, le=90)
    number_of_reviews: int | None = Field(
        None, ge=0, le=10000, description="Número total de reviews."
    )
    number_of_reviews_ltm: int | None = Field(None, ge=0, le=10000)
    number_of_reviews_l30d: int | None = Field(None, ge=0, le=10000)
    number_of_reviews_ly: int | None = Field(None, ge=0, le=10000)
    reviews_per_month: float | None = Field(None, ge=0, le=100)
    review_scores_rating: float | None = Field(
        None, ge=0, le=5, description="Puntuación media de reviews (0-5)."
    )
    host_is_superhost: HostFlag = Field(
        "UNKNOWN", description="Anfitrión superhost: t, f o UNKNOWN."
    )
    host_identity_verified: HostFlag = Field(
        "UNKNOWN", description="Identidad verificada: t, f o UNKNOWN."
    )

    @field_validator("amenities")
    @classmethod
    def limpiar_amenities(cls, valores: list[str]) -> list[str]:
        limpios = [valor.strip() for valor in valores if valor and valor.strip()]
        return limpios[:100]


class BatchListingInput(BaseModel):
    """Lote de alojamientos para predicción."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "listings": [
                    {
                        "room_type": "Entire home/apt",
                        "property_type": "Entire rental unit",
                        "neighbourhood_cleansed": "Westminster",
                        "latitude": 51.5074,
                        "longitude": -0.1278,
                        "accommodates": 2,
                    }
                ]
            }
        }
    )

    listings: list[ListingInput] = Field(..., min_length=1)
