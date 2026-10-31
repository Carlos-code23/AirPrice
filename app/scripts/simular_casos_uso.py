"""Ejecuta los tres casos de uso contra la API real (HTTP) con alojamientos del conjunto de test.

Uso (desde la raíz del repo AirPrice, con la API corriendo en otra terminal):
    uvicorn app.main:app --port 8000
    python app/scripts/simular_casos_uso.py http://localhost:8000 reports/simulacion_casos_uso.json

- La partición de test se reproduce con split_estratificado (semilla 42) sobre airprice_clean.csv,
  que tiene las mismas filas, el mismo orden de id y el mismo target que el dataset extendido.
- CU1 envía solo los 6 campos obligatorios.
- CU2 envía los 16 campos reales del dataset limpio. Si existe data/processed/airprice_features_ext.csv
  (no versionado), CU2 envía los 27 campos del contrato con la misma función que el benchmark.
- CU3 envía 500 alojamientos por /predict/batch en bloques de 50 (igual que la vista /lote). Como la
  vista, valida cada fila antes de enviarla y excluye las que la API rechazaría, con su motivo.
"""

from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from src.utils import SEED, split_estratificado  # noqa: E402

EXT = Path('data/processed/airprice_features_ext.csv')
API = sys.argv[1].rstrip('/')
SALIDA = Path(sys.argv[2])
CLASES = ['ECONOMICO', 'MEDIO', 'ALTO', 'PREMIUM']
IDX = {c: i for i, c in enumerate(CLASES)}


def limpio(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def payload_core(f: pd.Series) -> dict:
    return {
        'room_type': str(f['room_type']),
        'property_type': str(f['property_type']),
        'neighbourhood_cleansed': str(f['neighbourhood_cleansed']),
        'latitude': float(f['latitude']),
        'longitude': float(f['longitude']),
        'accommodates': int(f['accommodates']),
    }


def payload_ampliado(f: pd.Series) -> dict:
    p = payload_core(f)
    extra = {
        'bedrooms': limpio(f['bedrooms']),
        'beds': limpio(f['beds']),
        'bathrooms_num': limpio(f['bathrooms_num']),
        'minimum_nights': limpio(f['minimum_nights']),
        'availability_365': limpio(f['availability_365']),
        'number_of_reviews': limpio(f['number_of_reviews']),
        'reviews_per_month': limpio(f['reviews_per_month']),
        'review_scores_rating': limpio(f['review_scores_rating']),
        'host_is_superhost': str(f['host_is_superhost']),
        'host_identity_verified': str(f['host_identity_verified']),
    }
    # Los campos que el dataset marca como imputados en la limpieza no se envían:
    # así la API aplica sus propios defaults, igual que haría un usuario que los deja vacíos.
    for campo in ['bedrooms', 'beds', 'bathrooms_num', 'minimum_nights', 'reviews_per_month',
                  'review_scores_rating']:
        if f.get(f'{campo}_missing', 0) == 1:
            extra[campo] = None
    p.update({k: v for k, v in extra.items() if v is not None})
    return p


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


def payload_completo(f: pd.Series) -> dict:
    """Los 27 campos del contrato, igual que _payload(advanced=True) de benchmark_key_fields.py.

    Se copia en lugar de importarse porque ese módulo carga xgboost, que no está en el entorno de la API.
    """
    return {
        **payload_core(f),
        'bedrooms': f.get('bedrooms'),
        'beds': f.get('beds'),
        'bathrooms_num': f.get('bathrooms_num'),
        'minimum_nights': f.get('minimum_nights'),
        'amenities': _amenities(f.get('amenities')),
        'availability_365': f.get('availability_365'),
        'number_of_reviews': f.get('number_of_reviews'),
        'review_scores_rating': f.get('review_scores_rating'),
        'host_is_superhost': str(f.get('host_is_superhost', 'UNKNOWN')),
        'host_identity_verified': str(f.get('host_identity_verified', 'UNKNOWN')),
        'bathrooms_text': f.get('bathrooms_text'),
        'maximum_nights': f.get('maximum_nights'),
        'host_listings_count': f.get('host_listings_count'),
        'calculated_host_listings_count': f.get('calculated_host_listings_count'),
        'availability_30': f.get('availability_30'),
        'availability_60': f.get('availability_60'),
        'availability_90': f.get('availability_90'),
        'number_of_reviews_ltm': f.get('number_of_reviews_ltm'),
        'number_of_reviews_l30d': f.get('number_of_reviews_l30d'),
        'number_of_reviews_ly': f.get('number_of_reviews_ly'),
        'reviews_per_month': f.get('reviews_per_month'),
    }


def metricas(reales: list[str], pred: list[str]) -> dict:
    r = np.array([IDX[c] for c in reales])
    p = np.array([IDX[c] for c in pred])
    d = np.abs(r - p)
    n = len(r)
    return {
        'n': n,
        'accuracy': round(float((d == 0).mean()), 4),
        'accuracy_adyacente': round(float((d <= 1).mean()), 4),
        'mae_ordinal': round(float(d.mean()), 4),
        'errores_por_distancia': {str(k): int((d == k).sum()) for k in range(4)},
    }


def main() -> None:
    global payload_ampliado
    if EXT.exists():
        # Mismo split: el dataset extendido conserva filas, orden de id y target del limpio.
        df = pd.read_csv(EXT, low_memory=False)
        def payload_ampliado(f: pd.Series) -> dict:  # noqa: F811
            return {k: limpio(v) for k, v in payload_completo(f).items() if limpio(v) is not None}

        fuente = 'airprice_features_ext.csv (27 campos)'
    else:
        df = pd.read_csv('data/processed/airprice_clean.csv', low_memory=False)
        fuente = 'airprice_clean.csv (16 campos)'
    print('CU2 usa', fuente, flush=True)
    _, _, X_test, _, _, y_test = split_estratificado(
        df, target='price_range', excluir=['price_clean', 'id'], seed=SEED
    )
    test = df.loc[X_test.index]
    assert (test['price_range'] == y_test).all()

    cliente = httpx.Client(base_url=API, timeout=60)
    salud = cliente.get('/health').json()
    resultado: dict = {'api': API, 'health': salud, 'n_test': int(len(test)), 'fuente_cu2': fuente}

    # ---------------- CU1 y CU2: comparación pareada, un alojamiento por clase real (al azar, semilla 42)
    pareados = []
    for clase in CLASES:
        fila = test[test['price_range'] == clase].sample(n=1, random_state=SEED).iloc[0]
        registro = {
            'id': str(fila['id']),
            'barrio': fila['neighbourhood_cleansed'],
            'property_type': fila['property_type'],
            'room_type': fila['room_type'],
            'accommodates': int(fila['accommodates']),
            'precio_real': float(fila['price_clean']),
            'clase_real': clase,
        }
        for nombre, constructor in [('cu1', payload_core), ('cu2', payload_ampliado)]:
            cuerpo = constructor(fila)
            t0 = time.perf_counter()
            r = cliente.post('/api/v1/predict', json=cuerpo)
            ms = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            j = r.json()
            registro[nombre] = {
                'campos_enviados': len(cuerpo),
                'payload': cuerpo,
                'status': r.status_code,
                'latencia_ms': round(ms, 1),
                'label': j['prediction']['label'],
                'confidence': round(j['prediction']['confidence'], 4),
                'probabilities': {k: round(v, 4) for k, v in j['prediction']['probabilities'].items()},
                'warnings': j['warnings'],
                'request_id': j['request_id'],
            }
        pareados.append(registro)
    resultado['pareados'] = pareados
    resultado['modelo'] = j['model']

    # ---------------- CU3: lote de 500 alojamientos de test (máximo de la vista /lote), bloques de 50
    muestra = test.sample(n=500, random_state=SEED)

    # Igual que la vista /lote: cada fila se valida antes de enviarla y las inválidas se excluyen con su motivo.
    from pydantic import ValidationError
    from app.schemas.listing import ListingInput

    excluidas, validas_idx = [], []
    for idx, f in muestra.iterrows():
        try:
            ListingInput(**payload_ampliado(f))
            validas_idx.append(idx)
        except ValidationError as error:
            primero = error.errors()[0]
            excluidas.append({'id': str(f['id']), 'campo': str(primero['loc'][0]),
                              'valor': primero.get('input'), 'motivo': primero['msg']})
    validas = muestra.loc[validas_idx]
    print(f'filas válidas con {fuente}: {len(validas)} de {len(muestra)}', flush=True)

    def correr_lote(constructor, etiqueta, filas):
        preds, conf, con_warning, latencias = [], [], 0, []
        t_total = time.perf_counter()
        for i in range(0, len(filas), 50):
            bloque = [constructor(f) for _, f in filas.iloc[i:i + 50].iterrows()]
            t0 = time.perf_counter()
            r = cliente.post('/api/v1/predict/batch', json={'listings': bloque})
            latencias.append((time.perf_counter() - t0) * 1000)
            if r.status_code != 200:
                raise SystemExit(f'{etiqueta}: estado {r.status_code} en el bloque {i // 50}: {r.text[:300]}')
            j = r.json()
            for k, pr in enumerate(j['predictions']):
                preds.append(pr['label'])
                conf.append(pr['confidence'])
                if j['warnings'][k]:
                    con_warning += 1
        segundos = time.perf_counter() - t_total
        m = metricas(filas['price_range'].tolist(), preds)
        m.update({
            'peticiones': len(latencias),
            'latencia_media_bloque_ms': round(float(np.mean(latencias)), 1),
            'tiempo_total_s': round(segundos, 2),
            'confianza_media': round(float(np.mean(conf)), 4),
            'confianza_baja_<0.5': int(sum(c < 0.5 for c in conf)),
            'filas_con_warnings': con_warning,
            'predicciones_devueltas': len(preds),
        })
        print(etiqueta, json.dumps(m), flush=True)
        return m, preds

    lote_amp, preds_amp = correr_lote(payload_ampliado, 'CU3 ampliado', validas)

    # Comprobación del límite de 10 lotes/minuto: la petición 11 dentro del mismo minuto debe dar 429.
    r429 = cliente.post('/api/v1/predict/batch', json={'listings': [payload_core(muestra.iloc[0])]})
    resultado['rate_limit_peticion_11'] = {'status': r429.status_code, 'body': r429.text[:200]}
    print('peticion 11 ->', r429.status_code, flush=True)

    # Validación: una categoría desconocida debe devolver 422 (lo que la vista /lote filtra antes de enviar).
    print('esperando 65 s para reiniciar el límite de lotes...', flush=True)
    time.sleep(65)
    malo = payload_core(muestra.iloc[0]) | {'property_type': 'Castle'}
    r422 = cliente.post('/api/v1/predict', json=malo)
    resultado['validacion_422'] = {'status': r422.status_code, 'body': r422.json()}

    lote_core, preds_core = correr_lote(payload_core, 'CU3 core', muestra)

    # Comparación justa: las mismas filas en ambas configuraciones.
    pos = {idx: i for i, idx in enumerate(muestra.index)}
    preds_core_validas = [preds_core[pos[idx]] for idx in validas.index]
    core_mismas = metricas(validas['price_range'].tolist(), preds_core_validas)
    cambios = sum(a != b for a, b in zip(preds_core_validas, preds_amp))
    print('CU3 core (mismas filas)', json.dumps(core_mismas), flush=True)
    resultado['cu3'] = {
        'n': len(muestra),
        'filas_validas_ampliado': len(validas),
        'filas_excluidas_ampliado': excluidas,
        'ampliado': lote_amp,
        'core': lote_core,
        'core_mismas_filas': core_mismas,
        'predicciones_que_cambian_al_ampliar': cambios,
        'distribucion_real': {c: muestra['price_range'].tolist().count(c) for c in CLASES},
    }

    SALIDA.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding='utf-8')
    print('OK ->', SALIDA)


if __name__ == '__main__':
    main()
