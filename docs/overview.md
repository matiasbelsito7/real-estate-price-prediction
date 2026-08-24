# Overview — Real Estate Price Prediction

> **Proyecto portfolio end-to-end de Data Science / Machine Learning.**
> Predice precios de propiedades residenciales en CABA usando datos reales de
> Argenprop, y detecta oportunidades de compra comparando el precio predicho
> contra el publicado.

---

## Objetivo

| Objetivo | Descripción |
|---|---|
| Problema real | Predecir precios de propiedades residenciales (CABA) |
| Datos reales | Publicaciones de Argenprop obtenidas con scraper propio |
| Ciclo completo | Adquisición → Curación → EDA → Features → Modelo → Tracking → Evaluación → Explicabilidad → Deploy |
| Portfolio | Proyecto público, entendible para cualquier persona en GitHub |

---

## Ciclo de vida

```text
Data Acquisition (scraper)
      ↓
Data Curation (limpieza, moneda USD, indicadores missing)
      ↓
Feature Engineering (ordinal encoding, imputación, splits)
      ↓
Model Development (baseline, XGBoost, Lasso/Ridge, tuning)
      ↓
Experiment Tracking (MLflow: params, métricas, artefactos, Model Registry)
      ↓
Model Evaluation (métricas detalladas, residuos, segmentos, sesgo)
      ↓
Explainability (SHAP: valores, importancia, figuras)
      ↓
Serving (bundle de predicción + API FastAPI + Frontend)
      ↓
ETL periódico (scrape + predicción + clasificación + PostgreSQL)
      ↓
CD (fingerprint → smoke test → Docker image → GHCR)
```

---

## Filosofía y principios

El proyecto exige **calidad de software además de calidad de Machine Learning**.

- **Reproducibilidad** — configuración centralizada, versionado de datos (DVC), semillas
- **Testing** — pytest con unit + integration tests
- **Linting** — Ruff (format + check)
- **Type checking** — Mypy en modo `strict`
- **CI/CD** — GitHub Actions (lint, tests, ETL automatizado, CD del champion)
- **Containerization** — Docker multi-stage con usuario non-root
- **Separación de responsabilidades** — paquetes por capa en `src/`
- **Configuración centralizada** — `configs/config.yaml` + pydantic-settings

### Reglas

- No agregar herramientas sin justificación.
- Optuna queda excluido por decisión explícita.
- DVC es la dependencia (dev) encargada del versionado de datos.
- Incrementalidad: antes de implementar una pieza nueva hay que definir qué responsabilidad tiene.

---

## Stack tecnológico

| Área | Herramienta | Estado |
|---|---|---|
| Lenguaje | Python `>=3.11,<3.14` | ✔ |
| Datos | NumPy, Pandas | ✔ |
| Scraping | Requests, BeautifulSoup, lxml | ✔ |
| Machine Learning | Scikit-learn, XGBoost | ✔ |
| Experiment tracking | MLflow | ✔ |
| Explainable AI | SHAP | ✔ |
| Data versioning | DVC | ✔ |
| Testing | Pytest, pytest-cov | ✔ |
| Code quality | Ruff, Mypy | ✔ |
| CI/CD | GitHub Actions | ✔ |
| Containerization | Docker, Docker Compose | ✔ |
| API / Serving | FastAPI, Uvicorn | ✔ |
| Frontend | HTML5, CSS3, JavaScript vanilla | ✔ |
| Configuración | Pydantic Settings, python-dotenv, PyYAML | ✔ |
| Rate limiting | SlowAPI | ✔ |
| Base de datos | PostgreSQL, SQLAlchemy 2.0 (Core), psycopg | ✔ |

**Exclusiones explícitas:** Optuna.

---

## Documentación del proyecto

| Documento | Contenido |
|---|---|
| [overview.md](overview.md) | Este archivo — resumen y principios |
| [architecture.md](architecture.md) | Registro de componentes y estructura del repositorio |
| [data-pipeline.md](data-pipeline.md) | Ingestión, curación, features y DVC |
| [models.md](models.md) | Entrenamiento, tuning, evaluación y explicabilidad |
| [serving.md](serving.md) | Bundle de serving, API FastAPI, Docker y persistencia |
| [ci-cd.md](ci-cd.md) | Workflows de GitHub Actions y calidad de código |
| [roadmap.md](roadmap.md) | Roadmap de fases implementadas |
| [changelog.md](changelog.md) | Registro de mejoras recientes |
