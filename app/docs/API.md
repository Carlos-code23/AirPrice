# API AirPrice — Referencia

Base URL local: `http://localhost:8000`

Documentación interactiva:

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/api/v1/openapi.json`

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Información del servicio |
| `GET` | `/health` | Estado y modelo cargado |
| `GET` | `/api/v1/model/info` | Modelo, métricas y benchmark |
| `GET` | `/api/v1/schema` | Campos, defaults, categorías y amenidades |
| `POST` | `/api/v1/predict` | Predicción individual |
| `POST` | `/api/v1/predict/batch` | Predicción por lote (máx. 50) |

## `POST /api/v1/predict`

### Campos obligatorios

| Campo | Tipo | Ejemplo |
|---|---|---|
| `room_type` | enum | `Entire home/apt` |
| `property_type` | string | `Entire rental unit` |
| `neighbourhood_cleansed` | string | `Westminster` |
| `latitude` | float | `51.5074` |
| `longitude` | float | `-0.1278` |
| `accommodates` | int | `2` |

### Campos opcionales

`bedrooms`, `beds`, `bathrooms_num`, `bathrooms_text`, `minimum_nights`,
`maximum_nights`, `host_listings_count`, `calculated_host_listings_count`,
`amenities`, `availability_365`, `availability_30`, `availability_60`,
`availability_90`, `number_of_reviews`, `number_of_reviews_ltm`,
`number_of_reviews_l30d`, `number_of_reviews_ly`, `reviews_per_month`,
`review_scores_rating`, `host_is_superhost`, `host_identity_verified`.

Los valores válidos de `property_type`, `neighbourhood_cleansed`, `room_type` y
las amenidades están en `GET /api/v1/schema`.

### Ejemplo mínimo

```json
{
  "room_type": "Entire home/apt",
  "property_type": "Entire rental unit",
  "neighbourhood_cleansed": "Westminster",
  "latitude": 51.5074,
  "longitude": -0.1278,
  "accommodates": 2
}
```

### Ejemplo con campos avanzados

```json
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
  "bathrooms_text": "1 private bath",
  "minimum_nights": 2,
  "maximum_nights": 365,
  "host_listings_count": 12,
  "calculated_host_listings_count": 10,
  "amenities": ["Wifi", "Kitchen", "Heating", "Elevator"],
  "availability_365": 120,
  "availability_30": 20,
  "availability_60": 45,
  "availability_90": 70,
  "number_of_reviews": 45,
  "number_of_reviews_ltm": 18,
  "number_of_reviews_l30d": 2,
  "number_of_reviews_ly": 30,
  "reviews_per_month": 1.2,
  "review_scores_rating": 4.8,
  "host_is_superhost": "t",
  "host_identity_verified": "t"
}
```

### Respuesta

```json
{
  "request_id": "7f0aa9ab-585d-44fc-ac0a-0f015840e8de",
  "model": {
    "name": "lightgbm_ext",
    "version": "1.0.0",
    "trained_at": "2026-10-24",
    "classes": ["ECONOMICO", "MEDIO", "ALTO", "PREMIUM"],
    "n_features": 151,
    "metrics": {
      "accuracy": 0.7297,
      "f1_macro": 0.7288,
      "accuracy_adjacent": 0.9784,
      "qwk": 0.8593
    }
  },
  "prediction": {
    "label": "ALTO",
    "index": 2,
    "confidence": 0.84,
    "probabilities": {
      "ECONOMICO": 0.01,
      "MEDIO": 0.12,
      "ALTO": 0.84,
      "PREMIUM": 0.03
    }
  },
  "warnings": [
    "Campos no proporcionados; se usaron defaults de entrenamiento: ..."
  ]
}
```

`warnings` indica qué campos se imputaron. No es un error: es transparencia.

## `POST /api/v1/predict/batch`

```json
{
  "listings": [
    { "room_type": "Entire home/apt", "property_type": "Entire rental unit", "neighbourhood_cleansed": "Westminster", "latitude": 51.5074, "longitude": -0.1278, "accommodates": 2 },
    { "room_type": "Private room", "property_type": "Private room in home", "neighbourhood_cleansed": "Camden", "latitude": 51.5390, "longitude": -0.1426, "accommodates": 1 }
  ]
}
```

Máximo 50 alojamientos por lote.

## `GET /api/v1/schema`

Devuelve:

- `required_fields` y `optional_fields`.
- `advanced_fields`: campos que mejoran la precisión si el frontend los envía.
- `defaults`: valor que se usará si no se envía el campo.
- `allowed_values`: categorías válidas y vocabulario de amenidades.
- `bounds`: rangos numéricos.
- `feature_names`: las 151 features internas.
- `model_metrics`.

## Errores

| Código | Significado |
|---|---|
| `422` | Campo inválido, categoría desconocida o lote demasiado grande |
| `429` | Rate limit excedido |
| `503` | Modelo no cargado |
| `500` | Error interno |

Formato:

```json
{
  "detail": "property_type no reconocido. Consulta /api/v1/schema.",
  "request_id": "..."
}
```

## Límites

- `POST /predict`: 30 peticiones/minuto por IP.
- `POST /predict/batch`: 10 peticiones/minuto por IP.
- Lote máximo: 50.

## CORS

Configura los orígenes permitidos con `AIRPRICE_ALLOWED_ORIGINS`:

```env
AIRPRICE_ALLOWED_ORIGINS=http://localhost:5173,https://tu-frontend.com
```

## Métricas de referencia

| Configuración | Accuracy | F1-Macro | Adyacente | QWK |
|---|---:|---:|---:|---:|
| 151 features | 0.7297 | 0.7288 | 0.9784 | 0.8593 |
| Campos core + defaults | 0.654 | 0.647 | 0.963 | 0.802 |
| Campos core + avanzados | 0.705 | 0.695 | 0.974 | 0.839 |

El benchmark se calculó sobre 1.000 filas de test. Enviar los campos avanzados
mejora la precisión, pero la API funciona solo con los obligatorios.
