# AirPrice — Análisis de resultados: base vs geo vs extendido

## 1. Resumen ejecutivo

Se completaron los tres escenarios y el mejor esfuerzo de tuning/ensamble. El resultado central:

- **Base:** LightGBM 0.6883 accuracy exacta / 0.6865 F1-Macro.
- **Geo:** LightGBM 0.6906 / 0.6892. La mejora es pequeña y no robusta.
- **Extendido:** LightGBM 0.7297 / 0.7288. Es la mejora real: **+4.14 pp de accuracy y +4.23 pp de F1-Macro** sobre la baseline.
- **Tuning:** 0.7283 / 0.7278. No mejora al extendido simple; la selección en validación no generalizó.
- **Ensamble:** 0.7303 / 0.7295. Empate técnico con LightGBM extendido; no justifica la complejidad.

**Ningún escenario alcanza 0.90 de accuracy exacta.** El máximo es 0.7303. En cambio, la **accuracy adyacente** (acierto exacto o en la clase vecina) llega a **0.9796**, y el QWK a **0.8602**. El objetivo de 0.90 debe replantearse sobre la métrica ordinal, no sobre la exacta.

## 2. Tabla final

| Escenario | Mejor modelo | Accuracy | F1-Macro | Accuracy adyacente | QWK |
|---|---|---:|---:|---:|---:|
| Base | LightGBM | 0.6883 | 0.6865 | 0.9707 | 0.8306 |
| Geo | LightGBM | 0.6906 | 0.6892 | 0.9705 | 0.8300 |
| Extendido | LightGBM | **0.7297** | **0.7288** | 0.9784 | 0.8593 |
| Extendido + tuning | LightGBM (target encoding) | 0.7283 | 0.7278 | **0.9796** | **0.8602** |
| Extendido + ensamble | RF 0.1 / XGB 0.4 / LGBM 0.5 | **0.7303** | **0.7295** | 0.9788 | 0.8599 |

Lectura: la diferencia entre extendido, tuning y ensamble es **≤ 0.0015**, muy por debajo del error de muestreo del test (n = 9.243; IC 95 % ≈ ±0.009). Son estadísticamente indistinguibles.

## 3. Base → geo → extendido por modelo

| Modelo | Accuracy base | Accuracy geo | Δ geo | Accuracy ext | Δ ext |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.6883 | 0.6906 | +0.0023 | 0.7297 | **+0.0414** |
| XGBoost | 0.6711 | 0.6798 | +0.0087 | 0.7242 | **+0.0531** |
| Random Forest | 0.6814 | 0.6870 | +0.0056 | 0.7117 | **+0.0303** |
| Ordinal logístico | 0.6254 | 0.6395 | +0.0141 | 0.6738 | **+0.0484** |

F1-Macro:

| Modelo | Base | Geo | Δ geo | Extendido | Δ ext |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.6865 | 0.6892 | +0.0027 | 0.7288 | +0.0423 |
| XGBoost | 0.6685 | 0.6780 | +0.0095 | 0.7239 | +0.0554 |
| Random Forest | 0.6779 | 0.6846 | +0.0067 | 0.7101 | +0.0322 |
| Ordinal logístico | 0.6279 | 0.6420 | +0.0140 | 0.6768 | +0.0488 |

Conclusiones:

- **Geo ayuda más a los modelos débiles/lineales** (ordinal +1.41 pp, XGBoost +0.87 pp) porque no pueden explotar bien lat/lon por sí solos. A LightGBM le aporta casi nada (+0.23 pp).
- **Extendido mejora a los cuatro modelos de forma consistente** (+3.0 a +5.5 pp). Es el cambio que sí mueve la aguja.

## 4. ¿Por qué geo no despega?

- `latitude`, `longitude` y `neighbourhood_cleansed` ya estaban en la baseline.
- Las 5 distancias y `cluster_zona` son transformaciones deterministas de lat/lon. Añaden representación, no información nueva.
- Información mutua con el target: distancias 0.06–0.12, `cluster_zona` 0.02. Hay señal univariada, pero redundante.
- Correlación de Spearman: Hyde Park −0.40, centro −0.38, Kings Cross −0.32, Heathrow −0.14, Canary Wharf −0.10, cluster 0.18.
- **Ablación directa con LightGBM:** sin geo F1 0.68565; con geo F1 0.68473; **Δ = −0.00092**. No hay ganancia.
- La importancia agregada de geo es 27.8 % en RF, 27.7 % en LightGBM, 17.7 % en ordinal y 6.7 % en XGBoost, pero es importancia de uso, no valor incremental: hay muchas columnas geo correlacionadas.

**Veredicto geo:** mejora marginal, dentro del ruido. No es el camino para el 90 %.

## 5. ¿De dónde viene la mejora extendida?

El dataset extendido tiene 166 columnas (128 nuevas). Tras limpiar fugas, columnas vacías y no modelables quedan 151 features, de las cuales 8 son constantes (143 efectivas).

Importancia de uso en LightGBM por grupo:

| Grupo | % de importancia |
|---|---:|
| Geo/densidad | 31.2 % |
| Reviews/recencia | 23.2 % |
| Capacidad/tipo | 12.1 % |
| Disponibilidad | 11.5 % |
| Otras | 8.8 % |
| Host/portafolio | 6.6 % |
| Amenidades | 6.5 % |

Top LightGBM: `distancia_vecino_km`, `distancia_landmark_km`, `availability_365`, `latitude`, `distancia_hyde_park_km`, `amenities_count`, `availability_eoy`, `distancia_metro_km`, `distancia_min_poi_km`, `vecinos_1km`, `dias_desde_ultimo_review`, `minimum_nights_avg_ntm`, `reviews_per_month`, `entire_home_ratio`, `review_scores_cleanliness`, `review_scores_location`.

Top XGBoost: `room_type_Entire home/apt` (25.7 %), `room_type_Private room` (8.4 %), `bedrooms`, `accommodates`, `bathrooms_num`, `bathroom_shared`, `amenity_air_conditioning`, `pois_2km`, `distancia_hyde_park_km`, `amenity_dishwasher`.

La mejora no viene de una sola variable: es la combinación de **amenidades, reviews, disponibilidad, densidad y capacidad/tipo**. La importancia geo en el modelo extendido es alta porque hay muchas columnas geo correlacionadas, pero la ablación geo-only demuestra que por sí solas no explican el salto.

## 6. Matriz de confusión: dónde se equivoca

LightGBM extendido, test (n = 9.243):

| Clase real | Recall aprox. | Confusión principal |
|---|---:|---|
| ECONOMICO | 87.0 % | → MEDIO (277) |
| MEDIO | 62.9 % | → ALTO (451) |
| ALTO | 63.5 % | → MEDIO (439), → PREMIUM (360) |
| PREMIUM | 78.4 % | → ALTO (397) |

- Los errores graves (ECONOMICO↔PREMIUM) son rarísimos (7 y 27 casos).
- El problema está en las **fronteras MEDIO/ALTO** y **ALTO/PREMIUM**.
- Esto es exactamente lo esperable cuando el target se construye cortando el precio en cuantiles: anuncios de 178 y 182 £ caen en clases distintas pero son casi idénticos.
- El modelo base también confunde sobre todo MEDIO↔ALTO; el extendido mejora todas las clases, especialmente MEDIO (+168 aciertos) y ALTO (+82).

## 7. Tuning y ensamble: no aportan

### Tuning

- `RandomizedSearchCV` (3 folds, submuestra de 25k): mejor F1-Macro CV = 0.7059 ± 0.0025.
- Mejor configuración: 127 hojas, 500 árboles, lr 0.08, colsample 0.6, subsample 0.9, reg_lambda 5.
- Validación: F1 0.7344; test: accuracy 0.7283 / F1 0.7278.
- **Contra el extendido simple:** −0.0014 accuracy y −0.0010 F1. El tuning sobreajustó validación; no generalizó.
- Variantes en validación: target encoding 0.73437, One-Hot 0.73405, One-Hot agrupado 0.73363, `class_weight='balanced'` 0.73239. Todas dentro de ±0.002; `class_weight` no ayuda porque las clases ya están balanceadas.

### Ensamble

- Pesos elegidos en validación: RF 0.1, XGBoost 0.4, LightGBM 0.5.
- Test: accuracy 0.73028, F1 0.72947, adyacente 0.97879, QWK 0.85989.
- **Contra LightGBM extendido:** +0.00054 accuracy y +0.00066 F1. Es ruido, no una mejora real.

**Recomendación:** quedarse con **LightGBM extendido** por simplicidad y reproducibilidad. El ensamble solo tendría sentido si se buscara exprimir el último decimal en una competencia, no en un sistema mantenible.

## 8. Objetivo 0.90: veredicto

- Mejor accuracy exacta: **0.7303** (ensamble) / 0.7297 (LightGBM extendido). Brecha al 0.90: **≈ 17 pp**.
- Mejor accuracy adyacente: **0.9796** (tuning). Supera 0.90.
- Mejor QWK: **0.8602** (tuning).

El 0.90 exacto **no es alcanzable** con este dataset, este target de 4 cuantiles y features tabulares, sin fuga de precio. Para lograrlo habría que:

1. Cambiar el target (binario, o 3 clases, o regresión con tolerancia).
2. Aceptar la métrica ordinal (±1 clase) como criterio principal.
3. Incorporar datos más ricos (texto de descripción, fotos, amenidades detalladas, datos de reservas/ocupación).
4. Usar leakage de precio, lo cual está prohibido.

La opción defendible para la entrega es la 2: reportar accuracy exacta ≈ 0.73 y accuracy adyacente ≈ 0.98.

## 9. Calidad de datos y observaciones

- El merge por `id` es correcto: **1.0 de emparejamiento**. La auditoría anterior decía 0.9989 porque usaba `host_listings_count.notna()`; 66 filas tienen ese campo nulo, pero el `id` sí existe. Corregido en `src/etl_features_ext.py`.
- Columnas 100 % vacías en este scrape: `host_since`, `host_total_listings_count`, `host_response_time`, `host_verifications`, `neighbourhood`. Sus derivadas (`antiguedad_host_anios`, `host_listings_ratio`) también quedan vacías y se eliminan.
- Columnas constantes en el modelado: `instant_bookable`, `host_response_rate_num`, `host_acceptance_rate_num`, sus flags de faltante, `host_verifications_count`, `amenity_concierge`, `amenity_events`. No aportan; se pueden eliminar para limpiar.
- 49 flags de amenidades, 47 con varianza. Las 2 constantes son `amenity_concierge` y `amenity_events`.
- El dataset extendido pesa ~91 MB y tiene 166 columnas. No conviene versionarlo en Git sin LFS; es reproducible con `src/etl_features_ext.py`.
- No se detectó fuga: `price_clean`, `price`, `price_quote_*`, `estimated_revenue_l365d` y `estimated_occupancy_l365d` quedaron fuera del modelado.

## 10. Recomendaciones siguientes

1. **Cerrar el modelo con LightGBM extendido** (`models/lightgbm_ext.joblib`) y reportar la tabla base/geo/extendido.
2. **Reemplazar el criterio 0.90 exacto** por `accuracy_adjacent >= 0.90` y `QWK`; documentar el techo práctico.
3. **Limpiar columnas constantes/vacías** y volver a entrenar una vez para dejar el pipeline más limpio (no debería cambiar la accuracy).
4. Si se quiere atribuir la mejora extendida, hacer una **ablación por grupos** (sin amenidades, sin reviews, sin disponibilidad, sin densidad) en validación. Es el análisis que falta para saber qué grupo aporta más.
5. No invertir más en tuning/ensamble: la ganancia observada es menor que el error de muestreo.
6. Para la entrega, priorizar la interpretación ordinal y la comparación honesta de escenarios sobre perseguir un 0.90 que el dataset no soporta.
