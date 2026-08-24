# Roadmap

> Checklist de todas las fases del proyecto. Cada fase documenta el objetivo,
> los componentes implementados y el estado.

---

## Fases completadas

### Fase 1 — Dataset separado de nuevas publicaciones ✔

- **Objetivo:** scrapear publicaciones nuevas en dataset propio
- **Script:** `scripts/scrape_nuevas.py`
- **Componentes:** `ingestion/scraper.py` con `revision_periodica`

### Fase 2 — Predicción sobre nuevas publicaciones ✔

- **Objetivo:** aplicar curación + features + modelo al dataset nuevo
- **Script:** `scripts/evaluar_nuevas.py`
- **Componentes:** `serving/evaluar.py`

### Fase 3 — Clasificación buena/mala compra ✔

- **Objetivo:** comparar precio predicho contra publicado
- **Script:** `scripts/clasificar_ofertas.py`
- **Componentes:** `serving/clasificacion.py` (ratio + zona 1±std)

### Fase 4 — Modelos lineales (Lasso/Ridge) ✔

- **Objetivo:** comparar contra XGBoost con mismo pipeline
- **Script:** `scripts/train_lineales.py`
- **Componentes:** `models/modelos_lineales.py` (StandardScaler + regresor)
- **Resultado:** Ridge test RMSE log 0.3372 (XGBoost: 0.3040)

### Fase 5 — Tuning de hiperparámetros ✔

- **Objetivo:** explorar espacio y elegir campeón
- **Script:** `scripts/train_tuning.py`
- **Componentes:** `models/tuning.py` (GridSearchCV/RandomizedSearchCV)
- **Resultado:** default ya era casi óptimo

### Fase 6 — MLflow integral ✔

- **Objetivo:** tracking completo + comparación + champion
- **Scripts:** `scripts/comparar_runs.py`
- **Componentes:** `tracking/comparacion.py` (comparar_runs → elegir_champion)

### Fase 7 — SHAP Explainability ✔

- **Objetivo:** explicabilidad de modelos
- **Script:** `scripts/explain.py`
- **Componentes:** `explainability/shap_analysis.py`
- **Top-3:** superficie_cubierta, barrio_ordinal, expensas_usd

### Fase 8 — Evaluación profunda ✔

- **Objetivo:** métricas detalladas, residuos, segmentos, sesgo
- **Script:** `scripts/evaluate.py`
- **Componentes:** `evaluacion/analisis.py`

### Fase 10 — Serving y API FastAPI ✔

- **Objetivo:** exponer modelo como servicio HTTP
- **Endpoints:** `/health`, `/predict`, `/oportunidades`, `/oportunidades/{id}`
- **Componentes:** `serving/` (modelo, persistencia, evaluar, clasificacion), `api/`
- **Mejoras:** rate limiting (SlowAPI), validación tipo_propiedad, checksum SHA-256

### Fase 11 — Docker ✔

- **Objetivo:** contenerizar el servicio
- **Componentes:** `Dockerfile` multi-stage, `docker-compose.yml`, `requirements-api.txt`
- **Mejora:** usuario non-root (appuser, UID 1000)

### Fase 12 — ETL periódico de oportunidades ✔

- **Objetivo:** automatizar detección de oportunidades con PostgreSQL
- **Script:** `scripts/etl_oportunidades.py`
- **Componentes:** `serving/etl_oportunidades.py`, `persistencia/` (config, db, esquema, repositorio)
- **Workflow:** `.github/workflows/etl_oportunidades.yml` (cron cada 4 días)

### Fase 13 — CD del champion ✔

- **Objetivo:** automatizar publicación y despliegue
- **Script:** `scripts/publicar_champion.py`
- **Workflow:** `.github/workflows/cd_champion.yml`
- **Mecanismo:** fingerprint determinístico → smoke test → Docker image → GHCR

---

## Pendientes abiertos

- [ ] Tarea #19: fecha de publicación real desde la página de detalle
- [ ] Definir cadencia exacta del cron de scraping (default: 7 días)
- [ ] Decidir si el scraping programado re-escaneará segmentos completos o páginas acotadas
- [ ] Integrar config centralizada (`configs/config.yaml`) en todos los módulos
- [ ] Health check con métricas de Prometheus
- [ ] Indexación de base de datos para consultas frecuentes
