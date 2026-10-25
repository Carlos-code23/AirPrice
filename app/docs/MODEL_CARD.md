# Model Card — AirPrice `lightgbm_ext`

## Modelo

- Algoritmo: LightGBM multiclase.
- Target: `price_range` ordinal (`ECONOMICO` < `MEDIO` < `ALTO` < `PREMIUM`).
- Clases: 4, definidas por cuantiles del precio.
- Features: 151 tras feature engineering (32 base + geo + variables recuperadas).
- Artefacto: `models/lightgbm_ext.joblib`.
- Bundle de inferencia: `models/airprice_inference_bundle.joblib`.

## Métricas de test

| Métrica | Valor |
|---|---:|
| Accuracy exacta | 0.7297 |
| Balanced accuracy | 0.7294 |
| F1-Macro | 0.7288 |
| F1-Weighted | 0.7291 |
| Accuracy adyacente (±1 clase) | 0.9784 |
| MAE ordinal | 0.2956 |
| QWK | 0.8593 |

Las métricas exactas están limitadas por el target: los cortes por cuantiles
hacen que anuncios con precios muy cercanos caigan en clases distintas.

## Benchmark de la API

Sobre 1.000 filas de test:

| Configuración | Accuracy | F1-Macro | Adyacente | QWK |
|---|---:|---:|---:|---:|
| 151 features originales | 0.714 | 0.705 | 0.973 | 0.849 |
| Campos core + defaults | 0.654 | 0.647 | 0.963 | 0.802 |
| Campos core + avanzados | 0.705 | 0.695 | 0.974 | 0.839 |

Enviar los campos avanzados reduce la pérdida por imputación a ~1 pp.

## Uso previsto

- Segmentar anuncios en rangos de precio.
- Priorizar anuncios para análisis comercial.
- Demostración de inferencia tabular con LightGBM.

## No usar para

- Fijar el precio exacto de un alojamiento.
- Sustituir una tasación profesional.
- Tomar decisiones sin revisión humana.
- Inferir precios de ciudades distintas a Londres.

## Limitaciones

- Accuracy exacta ~0.73; no alcanza 0.90.
- El target se construyó por cuantiles; las fronteras son ambiguas.
- Algunos campos de la API se imputan con medianas/moda.
- No se usan imágenes, texto de descripción ni datos de calendario.
- El modelo se entrenó con un scrape concreto; puede degradarse con el tiempo.
- El split es aleatorio; para producción conviene validar con split temporal o
  espacial.

## Control de fuga

- No se usan `price`, `price_clean`, `price_quote_*` ni
  `estimated_revenue_l365d`.
- `price_range` no entra como feature.
- Clusters y densidades se ajustan con train.
- Target encoding, si se usa, es out-of-fold.

## Variables más relevantes

- Capacidad/tipo: `room_type`, `accommodates`, `bedrooms`, `bathrooms_num`.
- Disponibilidad: `availability_365`, `availability_30/60/90`, `availability_eoy`.
- Reviews: `reviews_per_month`, `number_of_reviews`, `review_scores_*`.
- Host: `host_listings_count`, `calculated_host_listings_count*`.
- Geo/densidad: distancias a POIs/estaciones, `cluster_zona*`, `vecinos_1km`.
- Amenidades: flags `amenity_*`.

## Versiones

- Python 3.12.2
- pandas 3.0.6
- numpy 2.5.3
- scikit-learn 1.9.1
- LightGBM 4.7.0
- joblib 1.6.0

## Reentrenamiento

El modelo se reentrena en `notebooks/models_features_ext/01_modelos_extendido.ipynb`.
El bundle de inferencia se regenera con:

```bash
python -m app.scripts.export_bundle
python -m app.scripts.benchmark_key_fields
```
