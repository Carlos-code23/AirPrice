# Despliegue de la API

## Local

```bash
pip install -r app/requirements-api.txt
python -m app.scripts.export_bundle
uvicorn app.main:app --reload --port 8000
```

## Docker

El Dockerfile está en `app/Dockerfile` y se construye desde la raíz del repo:

```bash
docker build -f app/Dockerfile -t airprice-api .
docker run --rm -p 8000:8000 --env-file .env airprice-api
```

El contenedor necesita `models/airprice_inference_bundle.joblib` y
`app/artifacts/`. El bundle no se versiona; se genera antes del build.

## Variables de entorno

```env
AIRPRICE_ALLOWED_ORIGINS=http://localhost:5173,https://tu-frontend.com
AIRPRICE_MODEL_BUNDLE_PATH=models/airprice_inference_bundle.joblib
AIRPRICE_FEATURE_SCHEMA_PATH=app/artifacts/feature_schema.json
AIRPRICE_MODEL_CARD_PATH=app/artifacts/model_card.json
AIRPRICE_MAX_BATCH_SIZE=50
AIRPRICE_RATE_LIMIT_SINGLE=30/minute
AIRPRICE_RATE_LIMIT_BATCH=10/minute
AIRPRICE_LOG_LEVEL=INFO
```

## Healthcheck

```bash
curl http://localhost:8000/health
```

Respuesta esperada:

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0",
  "loaded_at": "..."
}
```

## Producción

- Usa `uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2`.
- Coloca la API detrás de Nginx/Caddy con HTTPS.
- Configura CORS con el dominio real del frontend.
- Mantén el bundle en un volumen o cópialo en la imagen.
- No expongas `models/` ni `app/artifacts/` como archivos estáticos.
- Añade rate limit por IP (ya incluido) y, si crece el tráfico, un API gateway.
- Monitorea `/health`, latencia y tasa de errores.
- Versiona el modelo (`/api/v1/model/info`) y guarda el bundle usado en cada deploy.

## Recursos

- Bundle: ~6 MB.
- Memoria de la API: ~200–400 MB.
- Latencia esperada por predicción: decenas de milisegundos.
- El `BallTree` y los KMeans se cargan una sola vez al arrancar.
