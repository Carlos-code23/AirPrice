# AirPrice API

API pública de inferencia para clasificar alojamientos de Airbnb en Londres en
los rangos **ECONOMICO**, **MEDIO**, **ALTO** y **PREMIUM**.

- Modelo: `lightgbm_ext` (LightGBM, escenario extendido).
- Accuracy test: **0.7297**.
- Accuracy adyacente: **0.9784**.
- Documentación interactiva: `/docs` y `/redoc`.
- OpenAPI: `/api/v1/openapi.json`.

## Requisitos

```bash
pip install -r app/requirements-api.txt
```

El entorno de referencia usa Python 3.12, pandas 3.0.6, scikit-learn 1.9.1 y
LightGBM 4.7.0. Mantén esas versiones para cargar el `.joblib` sin problemas.

## Artefactos

La API necesita dos artefactos:

```text
models/airprice_inference_bundle.joblib   # modelo + KMeans + BallTree + defaults
app/artifacts/feature_schema.json         # esquema para el frontend
app/artifacts/model_card.json             # métricas y limitaciones
```

El bundle no se versiona (está en `.gitignore`). Genéralo con:

```bash
python -m app.scripts.export_bundle
```

Benchmark de campos clave:

```bash
python -m app.scripts.benchmark_key_fields
```

## Ejecutar

```bash
cp app/.env.example .env   # ajusta CORS y rutas
uvicorn app.main:app --reload --port 8000
```

Abre:

- `http://localhost:8000/docs`
- `http://localhost:8000/redoc`
- `http://localhost:8000/health`

## Ejemplo rápido

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "room_type": "Entire home/apt",
    "property_type": "Entire rental unit",
    "neighbourhood_cleansed": "Westminster",
    "latitude": 51.5074,
    "longitude": -0.1278,
    "accommodates": 2,
    "bedrooms": 1,
    "beds": 1,
    "bathrooms_num": 1,
    "amenities": ["Wifi", "Kitchen", "Heating"],
    "availability_365": 120,
    "number_of_reviews": 45,
    "review_scores_rating": 4.8
  }'
```

## Tests

```bash
python -m pytest app/tests -q
```

## Documentación

- [API.md](docs/API.md): endpoints, campos, errores y ejemplos.
- [REACT_INTEGRATION.md](docs/REACT_INTEGRATION.md): consumo desde React.
- [MODEL_CARD.md](docs/MODEL_CARD.md): métricas, límites y uso previsto.
- [DEPLOYMENT.md](docs/DEPLOYMENT.md): Docker y variables de entorno.
