# ============================================================
# Fase 11 — Docker
# Imagen del servicio de predicción FastAPI (real_estate.api.app).
#
# Build multi-etapa:
#   1. `build`: crea un venv con las dependencias de serving
#      (requirements-api.txt) e instala el paquete real_estate.
#   2. `runtime`: copia solo el venv y el código necesario.
#
# El bundle de serving (models/modelo_precio_propiedades/) NO se
# copia a la imagen: es un artefacto entrenado gitignoreado, por eso
# docker-compose.yml lo monta como volumen de solo lectura. Antes de
# `docker compose up` hay que ejecutar `make export-model`.
# ============================================================

# ============================================================
# Etapa 1: build
# ============================================================

FROM python:3.12-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copia el manifiesto y las dependencias del runtime de serving.
COPY requirements-api.txt pyproject.toml README.md ./
COPY src/ ./src/

# Crea un venv con las dependencias del servicio de predicción.
# --no-deps en la instalación del paquete: las dependencias reales ya
# vienen de requirements-api.txt (evita arrastrar mlflow/shap/scraping).
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements-api.txt \
    && /opt/venv/bin/pip install --no-cache-dir --no-deps .

# ============================================================
# Etapa 2: runtime
# ============================================================

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODELO_DIR=/app/models/modelo_precio_propiedades

WORKDIR /app

# Copia el venv del stage build (contiene el paquete real_estate instalado).
COPY --from=build /opt/venv /opt/venv

EXPOSE 8000

# Healthcheck: consulta el endpoint /health cada 30 s.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "real_estate.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
