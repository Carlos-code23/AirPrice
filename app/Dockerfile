FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY app/requirements-api.txt /app/app/requirements-api.txt
RUN pip install --no-cache-dir -r /app/app/requirements-api.txt

COPY app /app/app
COPY src /app/src
COPY models/airprice_inference_bundle.joblib /app/models/airprice_inference_bundle.joblib

RUN useradd --create-home --uid 1000 airprice \
    && chown -R airprice:airprice /app
USER airprice

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
