# Integración desde React

La API está pensada para consumirse desde un proyecto React separado.

## Configuración

```env
VITE_API_URL=http://localhost:8000
```

En producción:

```env
VITE_API_URL=https://api.tu-dominio.com
```

El backend debe permitir el origen del frontend:

```env
AIRPRICE_ALLOWED_ORIGINS=http://localhost:5173,https://tu-frontend.com
```

## Flujo recomendado

1. Al montar el formulario, `GET /api/v1/schema` para poblar:
   - `room_type`
   - `property_type`
   - `neighbourhood_cleansed`
   - amenidades
   - defaults y rangos
2. Enviar `POST /api/v1/predict`.
3. Mostrar `prediction.label`, `prediction.probabilities` y `warnings`.

## Tipos TypeScript

```ts
export type PriceRange = "ECONOMICO" | "MEDIO" | "ALTO" | "PREMIUM";

export interface ListingInput {
  room_type: "Entire home/apt" | "Private room" | "Shared room" | "Hotel room";
  property_type: string;
  neighbourhood_cleansed: string;
  latitude: number;
  longitude: number;
  accommodates: number;
  bedrooms?: number;
  beds?: number;
  bathrooms_num?: number;
  bathrooms_text?: string;
  minimum_nights?: number;
  maximum_nights?: number;
  host_listings_count?: number;
  calculated_host_listings_count?: number;
  amenities?: string[];
  availability_365?: number;
  availability_30?: number;
  availability_60?: number;
  availability_90?: number;
  number_of_reviews?: number;
  number_of_reviews_ltm?: number;
  number_of_reviews_l30d?: number;
  number_of_reviews_ly?: number;
  reviews_per_month?: number;
  review_scores_rating?: number;
  host_is_superhost?: "t" | "f" | "UNKNOWN";
  host_identity_verified?: "t" | "f" | "UNKNOWN";
}

export interface PredictionResponse {
  request_id: string;
  model: {
    name: string;
    version: string;
    trained_at: string;
    classes: PriceRange[];
    n_features: number;
    metrics: Record<string, number>;
  };
  prediction: {
    label: PriceRange;
    index: number;
    confidence: number;
    probabilities: Record<PriceRange, number>;
  };
  warnings: string[];
}
```

## Obtener el esquema

```ts
const API = import.meta.env.VITE_API_URL;

export async function getSchema() {
  const res = await fetch(`${API}/api/v1/schema`);
  if (!res.ok) throw new Error("No se pudo cargar el esquema");
  return res.json();
}
```

## Predecir

```ts
export async function predict(listing: ListingInput): Promise<PredictionResponse> {
  const res = await fetch(`${API}/api/v1/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(listing),
  });

  if (res.status === 429) {
    throw new Error("Demasiadas solicitudes. Espera un minuto.");
  }
  if (!res.ok) {
    const error = await res.json().catch(() => ({}));
    throw new Error(error.detail ?? "Error al predecir");
  }
  return res.json();
}
```

## Ejemplo de componente

```tsx
import { useState } from "react";
import { predict, type ListingInput, type PredictionResponse } from "./api";

export function PredictForm() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);

    const form = new FormData(event.currentTarget);
    const payload: ListingInput = {
      room_type: "Entire home/apt",
      property_type: String(form.get("property_type")),
      neighbourhood_cleansed: String(form.get("neighbourhood")),
      latitude: Number(form.get("latitude")),
      longitude: Number(form.get("longitude")),
      accommodates: Number(form.get("accommodates")),
      bedrooms: Number(form.get("bedrooms")) || undefined,
      bathrooms_num: Number(form.get("bathrooms")) || undefined,
      amenities: String(form.get("amenities") ?? "")
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean),
      availability_365: Number(form.get("availability_365")) || undefined,
      number_of_reviews: Number(form.get("number_of_reviews")) || undefined,
      review_scores_rating: Number(form.get("rating")) || undefined,
    };

    try {
      setResult(await predict(payload));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error desconocido");
    }
  }

  return (
    <form onSubmit={onSubmit}>
      {/* inputs... */}
      <button type="submit">Predecir</button>

      {error && <p role="alert">{error}</p>}

      {result && (
        <section>
          <h2>{result.prediction.label}</h2>
          <p>Confianza: {(result.prediction.confidence * 100).toFixed(1)}%</p>
          {Object.entries(result.prediction.probabilities).map(([label, value]) => (
            <div key={label}>
              <span>{label}</span>
              <progress value={value} max={1} />
              <span>{(value * 100).toFixed(1)}%</span>
            </div>
          ))}
          {result.warnings.length > 0 && (
            <details>
              <summary>Campos imputados</summary>
              <ul>
                {result.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </details>
          )}
        </section>
      )}
    </form>
  );
}
```

## Recomendaciones de UX

- Muestra siempre las **probabilidades**, no solo la etiqueta.
- Si `confidence < 0.50`, muestra una advertencia.
- Muestra `warnings` en un desplegable: indican campos que se completaron con
  defaults.
- Usa un mapa para `latitude`/`longitude` y limita el punto a Londres.
- Carga los dropdowns desde `/api/v1/schema`; no los hardcodees.
- Los campos avanzados (`host_listings_count`, `availability_30/60/90`,
  `bathrooms_text`, etc.) mejoran la precisión; muéstralos en un panel
  “Opciones avanzadas”.
- No interpretes la salida como un precio exacto; es un rango.

## CORS y errores típicos

- `403/CORS`: revisa `AIRPRICE_ALLOWED_ORIGINS`.
- `422`: categoría no permitida; consulta `/api/v1/schema`.
- `429`: rate limit; implementa un pequeño backoff.
- `503`: modelo no cargado; revisa `/health`.
