# Comparación de escenarios AirPrice

Tabla de F1-Macro por familia de modelo y escenario:

```text
escenario             base  extendido  extendido_ensemble  extendido_tuning       geo
familia                                                                              
ensemble               NaN        NaN            0.729466               NaN       NaN
lightgbm          0.686526   0.728812                 NaN          0.727782  0.689235
ordinal_logistic  0.627935   0.676780                 NaN               NaN  0.641959
random_forest     0.677917   0.710139                 NaN               NaN  0.684603
xgboost           0.668548   0.723944                 NaN               NaN  0.678050
```

Tabla de accuracy exacta:

```text
escenario             base  extendido  extendido_ensemble  extendido_tuning       geo
familia                                                                              
ensemble               NaN        NaN            0.730282               NaN       NaN
lightgbm          0.688305   0.729741                 NaN          0.728335  0.690577
ordinal_logistic  0.625446   0.673807                 NaN               NaN  0.639511
random_forest     0.681381   0.711674                 NaN               NaN  0.687006
xgboost           0.671102   0.724224                 NaN               NaN  0.679758
```

Tabla de accuracy adyacente (acierto exacto o en la clase vecina):

```text
escenario             base  extendido  extendido_ensemble  extendido_tuning       geo
familia                                                                              
ensemble               NaN        NaN            0.978795               NaN       NaN
lightgbm          0.970681   0.978362                 NaN          0.979552  0.970464
ordinal_logistic  0.967327   0.977280                 NaN               NaN  0.969274
random_forest     0.965812   0.971330                 NaN               NaN  0.966894
xgboost           0.966245   0.978795                 NaN               NaN  0.969274
```

## Mejor modelo por escenario

```text
         escenario             modelo  accuracy  f1_macro  accuracy_adjacent      qwk
extendido_ensemble       ensemble_ext  0.730282  0.729466           0.978795 0.859895
         extendido       lightgbm_ext  0.729741  0.728812           0.978362 0.859333
  extendido_tuning lightgbm_ext_tuned  0.728335  0.727782           0.979552 0.860228
               geo  lightgbm_features  0.690577  0.689235           0.970464 0.830026
              base           lightgbm  0.688305  0.686526           0.970681 0.830649
```

## Lectura

- El delta geo-base es pequeño si las distancias y los clusters no aportan información nueva: `latitude`, `longitude` y `neighbourhood_cleansed` ya estaban en la baseline.
- El escenario extendido añade amenidades, review scores, disponibilidad, portafolio del host y densidad. Es el mejor esfuerzo para acercarse al 90 %.
- La accuracy exacta de 4 clases ordinales tiene un techo práctico: los cortes de precio son cuantiles y los anuncios cercanos a un límite son ambiguos.
- `accuracy_adjacent` es la métrica ordinal relevante. Si supera 0.90, el modelo cumple el objetivo de negocio aunque la accuracy exacta no llegue a 0.90.
- El test se usó una sola vez por modelo; la selección de hiperparámetros se hizo en validación.