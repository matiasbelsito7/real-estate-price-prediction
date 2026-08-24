# Architecture — Registro de Componentes

> **Documento vivo.** Registro de todos los componentes del proyecto con sus
> responsabilidades, dependencias y relaciones. Para visión general ver
> [overview.md](overview.md).

---

## 1. Estructura del repositorio

```text
real-estate-price-prediction/
├── README.md
├── pyproject.toml                 ✔ Configuración del proyecto
├── Makefile                       ✔ Targets de build/test/deploy
├── Dockerfile                     ✔ Multi-stage + usuario non-root
├── docker-compose.yml             ✔ API + PostgreSQL
├── requirements-api.txt           ✔ Dependencias de serving
│
├── configs/
│   └── config.yaml                ✔ Config centralizada (pydantic-settings)
│
├── data/
│   ├── raw/                       ✔ Dataset crudo (DVC)
│   ├── processed/                 ✔ Curado + features (DVC)
│   └── external/                  ✔ Tipo de cambio histórico (DVC)
│
├── src/real_estate/
│   ├── utils/                     ✔ Config + logging centralizado
│   ├── ingestion/                 ✔ Scraper de Argenprop
│   ├── curation/                  ✔ Limpieza, moneda, indicadores
│   ├── features/                  ✔ Selección, ordinal encoding, imputación
│   ├── models/                    ✔ Entrenamiento, lineales, tuning
│   ├── tracking/                  ✔ MLflow experiments + comparación
│   ├── explainability/            ✔ SHAP analysis
│   ├── evaluacion/                ✔ Métricas detalladas, residuos
│   ├── serving/                   ✔ Bundle, predicción, clasificación, ETL
│   ├── api/                       ✔ FastAPI (health, predict, oportunidades, explain)
│   └── persistencia/              ✔ PostgreSQL (config, db, esquema, repo)
│
├── frontend/                      ✔ Frontend estático (HTML/CSS/JS)
│   ├── index.html                 ✔ Tabla de oportunidades + modal SHAP
│   ├── style.css                  ✔ Diseño responsive
│   └── app.js                     ✔ Lógica: fetch, render, auto-refresh
│
├── scripts/                       ✔ Entry points CLI (16 scripts)
├── tests/                         ✔ Unit (15 archivos) + integration (4)
├── models/                        ✔ Bundle de serving (gitignored)
├── reports/                       ✔ Figures + metrics
├── mlruns/                        ✔ MLflow store (gitignored)
│
├── docs/
│   ├── overview.md                ✔ Resumen y principios
│   ├── architecture.md            ✔ Este archivo
│   ├── data-pipeline.md           ✔ Ingestión → features
│   ├── models.md                  ✔ Modelos y evaluación
│   ├── serving.md                 ✔ API, Docker, persistencia
│   ├── ci-cd.md                   ✔ Workflows y calidad
│   ├── roadmap.md                 ✔ Fases implementadas
│   └── changelog.md               ✔ Mejoras recientes
│
└── .github/workflows/
    ├── ci.yml                     ✔ Lint + tests
    ├── dvc.yml                    ✔ Validación pipeline datos
    ├── etl_oportunidades.yml      ✔ ETL periódico (cron 4 días)
    └── cd_champion.yml            ✔ CD del champion
```

---

## 2. Registro de componentes

### 2.1 Configuración

| Componente | Ruta | Responsabilidad |
|---|---|---|
| `pyproject.toml` | raíz | Metadata, dependencias, config de Ruff/Mypy/Pytest |
| `Makefile` | raíz | Targets: install, pipeline, quality, docker, dvc |
| `configs/config.yaml` | configs | Parámetros centralizados (scraper, curation, features, models, tracking, serving, postgres) |
| `.pre-commit-config.yaml` | raíz | Hooks de calidad antes de cada commit |
| `.env.example` | raíz | Documentación de env vars requeridas |

### 2.2 Utils (`src/real_estate/utils/`)

| Módulo | Responsabilidad |
|---|---|
| `logging.py` | `configurar_logging()` — formato centralizado, nivel configurable |
| `config.py` | `ConfiguracionProyecto` — pydantic-settings + YAML loader, env vars `RE_*` |

### 2.3 Ingestión (`src/real_estate/ingestion/`)

| Módulo | Responsabilidad |
|---|---|
| `scraper.py` | Scraping de Argenprop: paginación, parseo, segmentación por barrio/tipo, backoff 202, progreso JSON |

### 2.4 Curation (`src/real_estate/curation/`)

| Módulo | Responsabilidad |
|---|---|
| `cleaning.py` | Conversión de tipos (texto → numérico), limpieza de expensas |
| `transformations.py` | Normalización moneda USD, tipo de cambio, indicadores missing |
| `validation.py` | Reglas de coherencia (solo reporte, no modifica) |
| `pipeline.py` | Orquestador `curar_dataset()` |

### 2.5 Features (`src/real_estate/features/`)

| Módulo | Responsabilidad |
|---|---|
| `transformations.py` | Selección columnas, target log, ordinal encoding, imputación |
| `pipeline.py` | Orquestador `construir_features()`, splits train/val/test |

### 2.6 Models (`src/real_estate/models/`)

| Módulo | Responsabilidad |
|---|---|
| `entrenamiento.py` | Baseline, XGBoost, preprocesamiento sin fuga, métricas |
| `modelos_lineales.py` | Lasso/Ridge con StandardScaler |
| `tuning.py` | GridSearchCV/RandomizedSearchCV sobre XGBoost |

### 2.7 Tracking (`src/real_estate/tracking/`)

| Módulo | Responsabilidad |
|---|---|
| `experimentos.py` | MLflow: params, métricas, artefactos, Model Registry |
| `comparacion.py` | Comparar runs, elegir champion |

### 2.8 Explicabilidad (`src/real_estate/explainability/`)

| Módulo | Responsabilidad |
|---|---|
| `shap_analysis.py` | TreeExplainer, importancia global, figuras |

### 2.9 Evaluación (`src/real_estate/evaluacion/`)

| Módulo | Responsabilidad |
|---|---|
| `analisis.py` | Métricas detalladas, residuos, segmentos, sesgo por rango |

### 2.10 Serving (`src/real_estate/serving/`)

| Módulo | Responsabilidad |
|---|---|
| `modelo.py` | `ModeloPrediccion` — predicción log/USD |
| `persistencia.py` | `guardar_bundle`/`cargar_bundle` con checksum SHA-256 |
| `evaluar.py` | Predicción sobre nuevas publicaciones |
| `clasificacion.py` | Ratio predicho/publicado, clasificación buena/mala compra |
| `etl_oportunidades.py` | Orquestador del ETL periódico |

### 2.11 API (`src/real_estate/api/`)

| Módulo | Responsabilidad |
|---|---|
| `app.py` | FastAPI: lifespan, /health, /predict, /oportunidades, /oportunidades/{id}/explain, rate limiting, frontend estático |
| `schemas.py` | Pydantic models: entrada, predicción, oportunidad, explicación SHAP |
| `config.py` | `ConfiguracionServicio` (directorio del modelo) |

### 2.12 Frontend (`frontend/`)

| Archivo | Responsabilidad |
|---|---|
| `index.html` | Página principal: tabla de oportunidades, filtros, modal de explicabilidad |
| `style.css` | Diseño responsive con variables CSS, badges de clasificación |
| `app.js` | Lógica: fetch API, renderizado, auto-refresh cada 30s, modales SHAP |

**Servido por FastAPI** como archivos estáticos en `/frontend/`. El endpoint raíz `/` redirige al frontend.

### 2.12 Persistencia (`src/real_estate/persistencia/`)

| Módulo | Responsabilidad |
|---|---|
| `config.py` | `ConfiguracionPostgres` (env vars `POSTGRES_*`) |
| `db.py` | `crear_engine()` — SQLAlchemy Engine |
| `esquema.py` | Tabla `oportunidades` (SQLAlchemy Core) |
| `repositorio.py` | upsert, dedup, listado, detalle |
| `bundle.py` | `guardar_bundle`/`cargar_bundle` (alias de serving/persistencia) |

### 2.13 Scripts

| Script | Target | Descripción |
|---|---|---|
| `scrape.py` | `make scrape` | Adquisición de datos |
| `scrape_nuevas.py` | — | Adquisición de nuevas publicaciones |
| `curate.py` | `make curate` | Data Curation |
| `features.py` | `make features` | Feature Engineering |
| `train.py` | `make train` | Entrenamiento + MLflow |
| `train_lineales.py` | `make train-lineales` | Lasso/Ridge |
| `train_tuning.py` | `make tuning` | Tuning XGBoost |
| `comparar_runs.py` | `make compare` | Elegir champion |
| `exportar_modelo.py` | `make export-model` | Exportar bundle |
| `publicar_champion.py` | `make publicar-champion` | Publicar champion |
| `evaluate.py` | `make evaluate` | Evaluación profunda |
| `explain.py` | `make explain` | Explicabilidad SHAP |
| `evaluar_nuevas.py` | — | Predicción sobre nuevas |
| `clasificar_ofertas.py` | — | Clasificación oportunidades |
| `etl_oportunidades.py` | `make etl` / `make etl-scrape` | ETL periódico (con/sin scraping) |
| `download_tipo_cambio.py` | — | Descargar histórico dólar |

### 2.14 Tests

| Tipo | Archivos | Cantidad |
|---|---|---|
| Unit | cleaning, scraper, transformations, features, models, modelos_lineales, tracking, comparacion, tuning, explainability, evaluacion, serving, clasificacion, repositorio, publicar_champion | ~244 tests |
| Integration | pipeline, evaluar_nuevas, API, ETL oportunidades | ~29 tests |

### 2.15 Datos

| Ruta | Contenido | Estado |
|---|---|---|
| `data/raw/propiedades_argenprop.csv` | Dataset crudo, 2.146 registros | ✔ |
| `data/processed/..._curado.csv` | Curado, 2.146 filas × 32 columnas | ✔ |
| `data/processed/..._features.csv` | Features, 2.140 filas × 16 columnas | ✔ |
| `data/external/tipo_cambio_blue.csv` | Histórico dólar blue, 5.702 fechas | ✔ |

---

## 3. Mapa de relaciones

```text
scripts/scrape.py → ingestion/scraper.py → data/raw/
    ↓
scripts/curate.py → curation/ → data/processed/curado.csv
    ↓
scripts/features.py → features/ → data/processed/features.csv
    ↓
scripts/train.py → models/entrenamiento.py → tracking/
    ↓
scripts/comparar_runs.py → tracking/comparacion.py → champion
    ↓
scripts/exportar_modelo.py → serving/ → models/bundle/
    ↓
api/app.py → /health, /predict, /oportunidades
    ↓
scripts/etl_oportunidades.py → persistencia/ → PostgreSQL
    ↓
cd_champion.yml → Docker image → GHCR
```
