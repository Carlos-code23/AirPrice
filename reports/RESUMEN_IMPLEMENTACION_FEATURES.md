# AirPrice — Implementación de features geoespaciales y escenario extendido

## 1. Resumen ejecutivo

Se implementó una infraestructura **aditiva** para comparar tres escenarios sin tocar el baseline:

| Escenario | Dataset | Variables | Modelos |
|---|---|---|---|
| Base | `airprice_clean.csv` | 32 | RF, XGBoost, LightGBM, Ordinal |
| Geo | `airprice_features.csv` | 32 + 6 geo | RF, XGBoost, LightGBM, Ordinal |
| Extendido | `airprice_features_ext.csv` | 32 + variables recuperadas + geo adicional | RF, XGBoost, LightGBM, Ordinal |
| Extendido + tuning | idem | idem | LightGBM con `RandomizedSearchCV`, target encoding y `class_weight` |
| Extendido + ensamble | idem | idem | Soft-voting RF + XGBoost + LightGBM con pesos en validación |

El split es el mismo en los tres escenarios: `split_estratificado(..., seed=42)`, 70/15/15.

## 2. Diagnóstico inicial

### 2.1 Alineación de datasets

- `airprice_clean.csv` y `airprice_features.csv` tienen **61.617 filas**.
- El orden de `id` y `price_range` es **idéntico** en todas las filas.
- El split estratificado produce exactamente los mismos índices de train/validación/test.
- `airprice_features.csv` conserva las 32 columnas base en el mismo orden y añade 6 columnas geo.

### 2.2 Solapamiento con el dataset original

- `data/listings.csv` tiene **90 columnas** en esta versión del scrape (no 106).
- El dataset limpio se derivó de ese archivo; todos los `id` del clean están presentes en el original.
- Por eso se usa **merge uno-a-uno por `id`** en lugar de reprocesar el CSV completo. El merge no cambia el número de filas ni el orden.

### 2.3 Variables del plan que no existen en este scrape

El plan original menciona `guests_included`, `extra_people`, `security_deposit` y `cleaning_fee`. Estas columnas **no están disponibles** en las 90 columnas de `data/listings.csv`. No se inventan ni se imputan.

Sí se recuperan: `host_listings_count`, `host_total_listings_count`, `review_scores_*` (6 subpuntuaciones), `availability_30/60/90`, `number_of_reviews_ltm/l30d/ly`, `minimum/maximum_nights_avg_ntm`, `calculated_host_listings_count*`, `host_since`, `first_review`, `last_review` y `amenities`.

## 3. Por qué hasta ahora no se alcanza el máximo de eficiencia

1. **El target es una discretización de precio en cuantiles.** Dos anuncios con precios 178 y 182 caen en clases distintas (ALTO/PREMIUM) pero pueden ser casi idénticos en sus features. El techo práctico de accuracy exacta para 4 clases ordinales con datos tabulares de Airbnb está lejos de 0.90; la literatura y los resultados del proyecto apuntan a 0.68–0.78.
2. **Las geo features no añaden información nueva.** `latitude`, `longitude` y `neighbourhood_cleansed` ya estaban en la baseline. Las 5 distancias son transformaciones deterministas de lat/lon, y `cluster_zona` (k=2) también. El aporte esperado es pequeño; sirven para representar mejor la ubicación, no para descubrirla.
3. **Se descartaron variables con señal.** El ETL baseline eliminó review scores, disponibilidad a 30/60/90 días, tamaño del portafolio del host, antigüedad y, sobre todo, la lista textual de `amenities`. El escenario extendido las recupera.
4. **El tuning baseline fue mínimo.** Cada modelo ajustó esencialmente un hiperparámetro en un único split de validación. Los grids pequeños y la selección sobre 15 % tienen ruido de ±0.005; las diferencias entre LightGBM (0.688) y RF (0.681) no son concluyentes.
5. **El modelo ordinal es lineal.** `mord.LogisticAT` no captura interacciones entre capacidad, tipo de habitación y ubicación. Es esperable que quede por debajo de los árboles.
6. **Las clases están balanceadas (~25 %).** `class_weight` no debería mejorar la accuracy; se prueba solo para documentarlo.
7. **El 90 % de accuracy exacta no es el objetivo correcto para un target ordinal.** La métrica adecuada es `accuracy_adjacent` (acierto exacto o en la clase vecina) y `qwk`. Es muy probable que `accuracy_adjacent` supere 0.90 incluso con accuracy exacta ≈ 0.69.

## 4. Arquitectura implementada

```text
src/
├── preprocessing_features.py   # One-Hot de clusters, imputación, escalado, TargetEncoderCV
├── model_features.py           # pipelines RF/XGB/LGBM/Ordinal + métricas ordinales + importancia
└── etl_features_ext.py         # merge por id + amenidades + geo adicional + auditoría

notebooks/
├── etl/
│   └── 01_features_ext.ipynb
├── models_features/
│   ├── 01_random_forest_features.ipynb
│   ├── 02_xgboost_features.ipynb
│   ├── 03_lightgbm_features.ipynb
│   └── 04_regresion_logistica_ordinal_features.ipynb
├── models_features_ext/
│   ├── 01_modelos_extendido.ipynb
│   ├── 02_tuning_extendido.ipynb
│   └── 03_ensamble_extendido.ipynb
└── evaluation/
    ├── 01_evaluacion_modelos.ipynb
    ├── 02_evaluacion_modelos_features.ipynb
    └── 03_comparacion_escenarios.ipynb
```

### Decisiones de diseño

- **No se modifica** `src/preprocessing.py`, `src/model_*.py` ni los notebooks baseline.
- **Misma semilla y split** en todos los escenarios.
- **Mismo conjunto de columnas base** en base vs geo: se aplican las mismas `COLUMNAS_DESCARTADAS` para que la única diferencia sean las 6 geo.
- **Sin fuga:** se excluyen `price_clean`, `price`, `price_quote_*`, `estimated_revenue_l365d` y `estimated_occupancy_l365d`.
- **Target encoding sin fuga:** `TargetEncoderCV` usa `KFold` interno; `fit_transform` devuelve codificaciones out-of-fold y `transform` usa el mapeo completo de train.
- **K-Means sin fuga:** los clusters k=5/8/10 se ajustan solo con coordenadas de train y luego se asignan a todos.
- **Métricas homogéneas:** `src.utils.evaluar_modelo` + `accuracy_adjacent`, `mae_ordinal` y `qwk`.

## 5. Orden de ejecución

1. `notebooks/etl/01_features_ext.ipynb` → genera `data/processed/airprice_features_ext.csv` (puede tardar por el CSV de 224 MB y `BallTree`).
2. `notebooks/models_features/01..04` → escenario geo.
3. `notebooks/models_features_ext/01_modelos_extendido.ipynb` → escenario extendido.
4. `notebooks/models_features_ext/02_tuning_extendido.ipynb` → tuning y variantes (el más costoso).
5. `notebooks/models_features_ext/03_ensamble_extendido.ipynb` → soft-voting con pesos elegidos en validación.
6. `notebooks/evaluation/01_evaluacion_modelos.ipynb` → base consolidado (re-entrena con los hiperparámetros baseline).
7. `notebooks/evaluation/02_evaluacion_modelos_features.ipynb` → diagnóstico y comparación geo.
8. `notebooks/evaluation/03_comparacion_escenarios.ipynb` → tabla final y reporte.

> El orden 1–5 puede cambiarse, pero 6–8 deben ejecutarse al final.

## 6. Resultados baseline conocidos

| Modelo | Accuracy | Balanced Accuracy | F1-Macro | F1-Weighted |
|---|---:|---:|---:|---:|
| LightGBM | 0.6883 | 0.6880 | 0.6865 | 0.6868 |
| Random Forest | 0.6814 | 0.6810 | 0.6779 | 0.6782 |
| XGBoost | 0.6711 | 0.6707 | 0.6685 | 0.6688 |
| Ordinal Logístico | 0.6254 | 0.6252 | 0.6279 | 0.6282 |

Los resultados geo, extendidos y de tuning se generan al ejecutar los notebooks. La tabla final queda en `reports/comparacion_escenarios.csv` y `reports/comparacion_escenarios.md`.

## 7. Qué esperar

- **Geo vs base:** delta pequeño (probablemente < 0.01). La ablación `reports/features_ablation_geo.csv` lo cuantifica.
- **Extendido vs base:** la mayor ganancia debería venir de `amenities`, review scores, disponibilidad y densidad. Un salto realista es +0.02 a +0.06 de accuracy exacta; llegar a 0.90 es improbable.
- **Tuning:** +0.00 a +0.01 sobre el extendido. Si el target encoding no mejora, se conserva One-Hot por simplicidad e interpretabilidad.
- **Ensamble:** soft-voting de RF + XGBoost + LightGBM con pesos elegidos en validación. La ganancia esperada es pequeña; si no supera al mejor individual, se descarta.
- **Accuracy adyacente:** probablemente > 0.95. Es la métrica que demuestra que el modelo ordinal es útil aunque la accuracy exacta no sea 0.90.

## 8. Commits sugeridos

```text
feat(features): agregar preprocesador para variables geoespaciales
feat(etl): recuperar variables originales y construir dataset extendido
feat(model): agregar pipelines y notebooks con features geoespaciales
feat(model): agregar escenario extendido y tuning con validacion cruzada
feat(eval): agregar evaluacion base, geo y comparacion de escenarios
docs(reports): documentar diagnostico, plan y resultados esperados
```
