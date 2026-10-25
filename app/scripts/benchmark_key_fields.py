"""Benchmark de la API usando solo los campos clave + defaults.

Compara, sobre el mismo conjunto de test:
1. El modelo con las 151 features reales.
2. El modelo con los campos clave y el resto imputado por la plantilla.

Uso:
    python -m app.scripts.benchmark_key_fields
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.model_features import evaluar_modelo_ordinal  # noqa: E402
from src.utils import SEED, split_estratificado  # noqa: E402

from app.services.feature_service import FeatureBuilder  # noqa: E402

EXT_PATH = RAIZ / "data" / "processed" / "airprice_features_ext.csv"
BUNDLE_PATH = RAIZ / "models" / "airprice_inference_bundle.joblib"
CARD_PATH = RAIZ / "app" / "artifacts" / "model_card.json"
BENCH_PATH = RAIZ / "app" / "artifacts" / "key_fields_benchmark.json"
REPORT_PATH = RAIZ / "reports" / "api_key_fields_benchmark.csv"


def _safe(valor):
    try:
        if pd.isna(valor):
            return None
    except (TypeError, ValueError):
        pass
    return valor


def _amenities(valor) -> list[str]:
    if not isinstance(valor, str):
        return []
    try:
        lista = ast.literal_eval(valor)
        if isinstance(lista, list):
            return [str(item) for item in lista]
    except (ValueError, SyntaxError):
        pass
    return []


def _payload(fila: pd.Series, advanced: bool = False) -> dict:
    payload = {
        "room_type": str(fila["room_type"]),
        "property_type": str(fila["property_type"]),
        "neighbourhood_cleansed": str(fila["neighbourhood_cleansed"]),
        "latitude": float(fila["latitude"]),
        "longitude": float(fila["longitude"]),
        "accommodates": int(fila["accommodates"]),
        "bedrooms": _safe(fila.get("bedrooms")),
        "beds": _safe(fila.get("beds")),
        "bathrooms_num": _safe(fila.get("bathrooms_num")),
        "minimum_nights": _safe(fila.get("minimum_nights")),
        "amenities": _amenities(fila.get("amenities")),
        "availability_365": _safe(fila.get("availability_365")),
        "number_of_reviews": _safe(fila.get("number_of_reviews")),
        "review_scores_rating": _safe(fila.get("review_scores_rating")),
        "host_is_superhost": str(fila.get("host_is_superhost", "UNKNOWN")),
        "host_identity_verified": str(fila.get("host_identity_verified", "UNKNOWN")),
    }
    if advanced:
        payload.update(
            {
                "bathrooms_text": _safe(fila.get("bathrooms_text")),
                "maximum_nights": _safe(fila.get("maximum_nights")),
                "host_listings_count": _safe(fila.get("host_listings_count")),
                "calculated_host_listings_count": _safe(
                    fila.get("calculated_host_listings_count")
                ),
                "availability_30": _safe(fila.get("availability_30")),
                "availability_60": _safe(fila.get("availability_60")),
                "availability_90": _safe(fila.get("availability_90")),
                "number_of_reviews_ltm": _safe(fila.get("number_of_reviews_ltm")),
                "number_of_reviews_l30d": _safe(fila.get("number_of_reviews_l30d")),
                "number_of_reviews_ly": _safe(fila.get("number_of_reviews_ly")),
                "reviews_per_month": _safe(fila.get("reviews_per_month")),
            }
        )
    return payload


def main() -> None:
    bundle = joblib.load(BUNDLE_PATH)
    modelo = bundle["model"]
    builder = FeatureBuilder(bundle)
    df = pd.read_csv(EXT_PATH, low_memory=False)

    X_train, _, X_test, _, _, y_test = split_estratificado(
        df, target="price_range", excluir=["price_clean", "id"], seed=SEED
    )
    n = min(1000, len(X_test))
    muestra = X_test.sample(n=n, random_state=SEED)
    y_muestra = y_test.loc[muestra.index]

    # 1) Con todas las features reales.
    proba_full = modelo.predict_proba(muestra[builder.feature_names])
    clases = [str(c) for c in modelo.named_steps["modelo"].classes_]
    pred_full = np.array([clases[int(i)] for i in proba_full.argmax(axis=1)])
    metricas_full = evaluar_modelo_ordinal(y_muestra, pred_full)

    # 2) Solo campos clave + defaults.
    def _evaluar(advanced: bool):
        predicciones = []
        fallos = 0
        for _, fila in df.loc[muestra.index].iterrows():
            try:
                X, _ = builder.build(_payload(fila, advanced=advanced))
                proba = np.asarray(modelo.predict_proba(X))[0]
                predicciones.append(clases[int(proba.argmax())])
            except Exception:
                fallos += 1
                predicciones.append("MEDIO")
        return evaluar_modelo_ordinal(y_muestra, np.array(predicciones)), fallos

    metricas_key, fallos_key = _evaluar(advanced=False)
    metricas_adv, fallos_adv = _evaluar(advanced=True)

    resultado = {
        "n": int(n),
        "fallos_construccion_key": int(fallos_key),
        "fallos_construccion_advanced": int(fallos_adv),
        "full_features": {
            k: float(v) for k, v in metricas_full.items() if k != "matriz_confusion"
        },
        "key_fields": {
            k: float(v) for k, v in metricas_key.items() if k != "matriz_confusion"
        },
        "advanced_fields": {
            k: float(v) for k, v in metricas_adv.items() if k != "matriz_confusion"
        },
        "delta_accuracy_key_vs_full": float(
            metricas_key["accuracy"] - metricas_full["accuracy"]
        ),
        "delta_accuracy_advanced_vs_full": float(
            metricas_adv["accuracy"] - metricas_full["accuracy"]
        ),
        "note": (
            "key_fields usa solo los campos core del formulario; advanced_fields "
            "añade los campos opcionales avanzados; full_features usa las 151 "
            "features originales. Los campos no enviados se imputan con "
            "medianas/moda del entrenamiento."
        ),
    }

    BENCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCH_PATH.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"configuracion": "full_features", **resultado["full_features"]},
            {"configuracion": "key_fields", **resultado["key_fields"]},
            {"configuracion": "advanced_fields", **resultado["advanced_fields"]},
        ]
    ).to_csv(REPORT_PATH, index=False)

    if CARD_PATH.exists():
        card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
        card["key_fields_benchmark"] = {
            "key_fields": {
                **resultado["key_fields"],
                "n": resultado["n"],
                "delta_accuracy_vs_full": resultado[
                    "delta_accuracy_key_vs_full"
                ],
            },
            "advanced_fields": {
                **resultado["advanced_fields"],
                "n": resultado["n"],
                "delta_accuracy_vs_full": resultado[
                    "delta_accuracy_advanced_vs_full"
                ],
            },
        }
        CARD_PATH.write_text(
            json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        bundle["model_card"] = card
        joblib.dump(bundle, BUNDLE_PATH, compress=3)

    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
