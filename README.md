# Predicción de precios de propiedades — Real Estate Price Prediction

Proyecto **end-to-end de Data Science / Machine Learning** que predice precios de
propiedades residenciales en **Capital Federal** usando datos reales obtenidos
mediante scraping de **Argenprop**.

El repositorio documenta el ciclo completo de vida de un proyecto de ML:

```text
Data Acquisition  →  Data Curation  →  EDA  →  Feature Engineering  →
Model Development  →  Experiment Tracking  →  Model Evaluation  →
Explainability  →  Deployment  →  Serving (FastAPI)  →  ETL periódico  →  CD del champion
```

---

## Tabla de contenidos

1. [Arquitectura general](#arquitectura-general)
2. [Instalación](#instalación)
3. [Comandos rápidos (MakeFile)](#comandos-rápidos-makefile)
4. [Flujo de datos](#flujo-de-datos)
5. [Fuente de datos: Argenprop](#fuente-de-datos-argenprop)
6. [Data Curation](#data-curation)
7. [Feature Engineering](#feature-engineering)
8. [Modelado](#modelado)
9. [Experiment tracking (MLflow)](#experiment-tracking-mlflow)
10. [Explainability (SHAP)](#explainability-shap)
11. [Model evaluation](#model-evaluation)
12. [Serving y API](#serving-y-api)
13. [ETL periódico de oportunidades](#etl-periódico-de-oportunidades)
14. [Docker](#docker)
15. [CI/CD](#cicd)
16. [Testing](#testing)
17. [Quality gates](#quality-gates)
18. [Documentación](#documentación)

---

## Arquitectura general

El proyecto sigue una **arquitectura por capas** en `src/real_estate/`:

| Capa | Responsabilidad |
|---|---|
| `ingestion/` | Scraper de Argenprop (segmentación por barrio/tipo, backoff 202, progreso reanudable) |
| `curation/` | Limpieza de tipos, normalización de moneda a USD, indicadores de missing, validación |
| `features/` | Selección de columnas, target logarítmico, codificación ordinal, imputación, splits |
| `models/` | Baseline, XGBoost, modelos lineales (Lasso/Ridge), tuning (GridSearchCV/RandomizedSearchCV) |
| `tracking/` | Integración con MLflow (params, métricas, artefactos, Model Registry, comparación de runs) |
| `explainability/` | SHAP (valores, base, importancia global, figuras) |
| `evaluacion/` | Métricas detalladas, residuos, error por segmento, sesgo por rango de precio |
| `serving/` | Bundle de serving, predicción, clasificación de oportunidades, ETL periódico |
| `api/` | API FastAPI (`/health`, `/predict`, `/oportunidades`) |
| `persistencia/` | Capa de persistencia PostgreSQL (SQLAlchemy Core, upsert multi-dialecto) |

**Exclusiones explícitas:** Optuna queda excluido por regla del proyecto; el
tuning de XGBoost usa `GridSearchCV`/`RandomizedSearchCV`.

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/matiasbelsito7/real-estate-price-prediction.git
cd real-estate-price-prediction

# Instalar dependencias (Python >=3.11,<3.14)
pip install -e ".[dev]"
```

Ver `pyproject.toml` para detalles de dependencias y configuración de herramientas
(python `>=3.11,<3.14`, Ruff, Mypy en modo `strict`, Pytest con cobertura).

---

## Comandos rápidos (MakeFile)

El archivo real se llama `MakeFile` (no `Makefile`). Los targets aceptan
`ARGS="--flag valor"`.

| Comando | Descripción |
|---|---|
| `make install` | Instalar dependencias |
| `make install-dev` | Instalar dependencias + dev |
| `make scrape` | Ejecutar el scraper de Argenprop |
| `make curate` | Ejecutar Data Curation |
| `make features` | Ejecutar Feature Engineering |
| `make train` | Entrenar Baseline + XGBoost con MLflow |
| `make train-lineales` | Entrenar modelos lineales Lasso/Ridge |
| `make tuning` | Tuning de hiperparámetros de XGBoost |
| `make compare` | Comparar runs de MLflow y elegir el champion |
| `make export-model` | Exportar el bundle de serving del champion |
| `make publicar-champion` | Publicar el champion en GitHub Releases (fase 13) |
| `make serve` | Levantar la API FastAPI (uvicorn) |
| `make etl` | Ejecutar el ETL periódico de oportunidades |
| `make docker-build` | Build de la imagen Docker |
| `make docker-up` | Levantar el stack con Docker Compose |
| `make check` | `ruff format --check` + `ruff check` + `mypy` + `pytest` |
| `make format` | Formatear con Ruff |
| `make lint` | Lint con Ruff |
| `make typecheck` | Type check con Mypy |
| `make test` | Tests con Pytest |
| `make coverage` | Tests con cobertura |

---

## Flujo de datos

```text
Argenprop
    ↓
scripts/scrape.py ──→ src/real_estate/ingestion/scraper.py
    ↓
data/raw/propiedades_argenprop.csv          ◄─ DVC
    ↓
scripts/curate.py ──→ src/real_estate/curation/
    ↓
data/processed/propiedades_argenprop_curado.csv  ◄─ DVC
    ↓
scripts/features.py ──→ src/real_estate/features/
    ↓
data/processed/propiedades_argenprop_features.csv  ◄─ DVC
    ↓
scripts/train.py ──→ src/real_estate/models/ ──→ src/real_estate/tracking/ (MLflow)
    ↓
scripts/comparar_runs.py ──→ src/real_estate/tracking/comparacion.py
    ↓
scripts/exportar_modelo.py ──→ src/real_estate/serving/ + src/real_estate/tracking/
    ↓
models/modelo_precio_propiedades/  (bundle de serving)
    ↓
src/real_estate/api/app.py  (FastAPI)
    ↓
make serve (uvicorn)
```

### Flujo del cron (detección de oportunidades)

```text
scripts/scrape_nuevas.py ──→ src/real_estate/ingestion/scraper.py
    ↓
data/raw/propiedades_nuevas.csv
    ↓
scripts/evaluar_nuevas.py ──→ src/real_estate/serving/ + curation (cura en memoria)
    ↓
data/processed/propiedades_nuevas_evaluadas.csv (precio_predicho_usd, fecha_prediccion)
    ↓
scripts/clasificar_ofertas.py ──→ src/real_estate/serving/clasificacion.py
    ↓
reports/ofertas.csv (ranking buena/mala compra)
```

### Flujo del ETL periódico (fase 12)

```text
scripts/etl_oportunidades.py scrape --todos-los-barrios
    ↓
data/raw/propiedades_nuevas.csv
    ↓
scripts/etl_oportunidades.py etl
    ├── persistencia/repositorio.ids_procesados()
    ├── serving/evaluar.py + clasificacion.py
    ├── persistencia/repositorio.upsert_oportunidades()
    └── reports/oportunidades_nuevas.csv
    ↓
PostgreSQL ──→ src/real_estate/api/app.py (GET /oportunidades + /{id})
```

---

## Fuente de datos: Argenprop

El scraper (`src/real_estate/ingestion/scraper.py`, ejecutado por
`scripts/scrape.py`) obtiene avisos de **venta** de toda clase de propiedad en
**Capital Federal**.

### Segmentación por barrio (v3)

Argenprop **corta toda búsqueda en la página 100** con un HTTP 202 vacío aproximadamente 2.000 avisos por búsqueda). Para superarlo, la búsqueda se **segmenta por barrio**
(54 barrios de CABA) y/o tipo de propiedad; cada segmento tiene su propio
paginación y su propio cap. El progreso por segmento se guarda en un JSON
(`--progreso`), de modo que una corrida interrumpida reanuda desde la última
página procesada.

### Uso del scraper

```bash
# Probar con pocas páginas
python scripts/scrape.py --max-paginas 5 --output data/raw/prueba.csv

# Scraping completo (segmentado por barrio + tipo)
python scripts/scrape.py --todos-los-barrios --tipo departamentos

# Scraping acotado a barrios específicos
python scripts/scrape.py --barrios palermo,recoleta,caballito
```

### Columnas del dataset raw

| Columna | Descripción |
|---|---|
| `id` | ID interno del aviso en Argenprop |
| `link` | URL del aviso |
| `titulo` | Título del aviso |
| `descripcion` | Copete/descripción corta |
| `tipo_propiedad` / `idtipopropiedad` | Tipo (departamento, casa, ph, etc.) |
| `barrio` / `sub_barrio` | Ubicación |
| `precio` / `moneda` | Precio (numérico) y moneda (USD/ARS) |
| `expensas` | Expensas mensuales (texto, con símbolos) |
| `superficie_cubierta` / `superficie_semicubierta` / `superficie_total` | Superficie en m² (texto) |
| `ambientes` | Cantidad (extraída de la URL) |
| `dormitorios` | Cantidad |
| `banos` | Cantidad (suele faltar — limitación del listado) |
| `cocheras` | Cantidad (casi siempre falta — limitación del listado) |
| `antiguedad` | Años (texto: "17 años", "A estrenar", etc.) |
| `fecha_scrape` | Fecha de descarga |

**Nota sobre `banos` y `cocheras`:** quedan mayormente vacíos por limitación del
listado (no es un bug). Completarlos exigiría visitar la página de detalle de cada
aviso. El scraper no limpia datos — eso pertenece a Data Curation.

---

## Data Curation

La curación (`src/real_estate/curation/`, ejecutada por `scripts/curate.py`)
transforma los datos crudos en un dataset limpio:

| Etapa | Responsabilidad |
|---|---|
| `cleaning.py` | Conversión de texto a numérico (limpieza de símbolos, separadores de miles, unidades) |
| `transformations.py` | Normalización de moneda a USD (con tipo de cambio histórico de `data/external/tipo_cambio_blue.csv`), indicadores `{columna}_informado` |
| `validation.py` | Validación de coherencia (precio > 0, superficie > 0, ambientes >= 1) — solo reporta |

```bash
make curate
# o directamente:
python scripts/curate.py
```

**Output:** `data/processed/propiedades_argenprop_curado.csv` (32 columnas).

---

## Feature Engineering

El feature engineering (`src/real_estate/features/`, ejecutado por `scripts/features.py`)
prepara la matriz para modelar:

- Target `log_precio_usd` (logarítmo del precio en USD)
- Codificación ordinal de `barrio` y `tipo_propiedad` por mediana de precio
- Imputación por mediana
- Split 80/10/10 reproducible (`random_state=42`)
- Filtro de precios inválidos (< 1.000 USD)

```bash
make features
# o directamente:
python scripts/features.py
```

**Output:** `data/processed/propiedades_argenprop_features.csv` (1.999 filas × 16 columnas, 0 faltantes).

---

## Modelado

`src/real_estate/models/` contiene los modelos implementados:

| Modelo | Script | Resultado sobre el dataset real |
|---|---|---|
| Baseline (mediana) | `scripts/train.py` | RMSE log (test): — |
| XGBoost (default) | `scripts/train.py` | **RMSE log 0.3040, R² 0.7830** en test |
| Lasso | `scripts/train_lineales.py` | RMSE log 0.6421 (alpha=1.0 degenera al baseline) |
| Ridge | `scripts/train_lineales.py` | RMSE log 0.3372, R² 0.7330 en test |
| XGBoost tuned | `scripts/train_tuning.py` | RMSE log 0.3020 (mejora marginal ~0.7 %) |

**XGBoost es el campeón** sobre el dataset real. El tuning (fase 5) muestra que
los hiperparámetros por defecto ya estaban casi óptimos.

### Sin fuga de información

El preprocesamiento (ordenes ordinales, imputación) se ajusta **solo sobre train**
y se reaplica a val/test. Categorías no vistas en train → `CODIGO_DESCONOCIDO = -1`.

---

## Experiment tracking (MLflow)

`src/real_estate/tracking/` gestiona el tracking con MLflow:

- **`experimentos.py`**: `registrar_resultado` (params, métricas, artefactos,
  Model Registry), `registrar_lineales` (una corrida por modelo lineal),
  `registrar_tuning` (run resumen + un run anidado por trial)
- **`comparacion.py`**: `comparar_runs` y `elegir_champion` (métrica default:
  `xgboost_test_rmse_log`)

El store local vive en `mlruns/` (gitignored). El experimento se llama
`prediccion_precios_propiedades`.

```bash
# Entrenar y trackear
make train

# Comparar runs y elegir champion
make compare
```

---

## Explainability (SHAP)

`src/real_estate/explainability/shap_analysis.py` explica el modelo con SHAP:

- Valores SHAP y base (`exp(base) ≈ USD 158.234`)
- Propiedad aditiva verificada (error máx. 8.6e-06)
- Importancia global, beeswarm y gráfico de barras
- Interpretación en USD vía `exp(contribución)`

**Top-3 features más importantes:** `superficie_cubierta` (0.30),
`barrio_ordinal` (0.14), `expensas_usd` (0.07).

El notebook `05_shap_analysis.ipynb` genera las figuras en `reports/figures/`.

---

## Model evaluation

`src/real_estate/evaluacion/analisis.py` evalúa el modelo en profundidad:

- Métricas detalladas: RMSE log, RMSE/MAE/MedAE/MAPE en USD, R²
- Tabla de residuos, resumen de errores, error por segmento, sesgo por rango
- Figuras en `reports/figures/`

**Resultado en test:** RMSE log 0.3040, RMSE USD $87.244, R² 0.7830; error
relativo absoluto mediano 17.73 %; sesgo medio +5.02 % (sobreestima levemente).
Sobreestima las propiedades baratas y subestima las caras.

El notebook `06_model_evaluation.ipynb` ejecuta el pipeline completo.

---

## Serving y API

El modelo se expone como servicio HTTP mediante FastAPI:

```bash
make export-model  # Genera models/modelo_precio_propiedades/
make serve         # uvicorn real_estate.api.app:app
```

### Bundle de serving (`models/modelo_precio_propiedades/`)

| Archivo | Contenido |
|---|---|
| `modelo_xgboost.json` | Modelo XGBoost guardado con formato JSON nativo |
| `preprocesamiento.json` | Órdenes ordinales + imputador aprendidos sobre train |
| `features.json` | Orden exacto de las 14 features esperadas |
| `metadata.json` | Métricas en test, tamaños de split, fecha de exportación |

### API FastAPI (`/health`, `/predict`, `/oportunidades`)

| Endpoint | Método | Descripción |
|---|---|---|
| `/health` | GET | Estado, modelo, versión, métricas del bundle |
| `/predict` | POST | Predicción de precio en USD + log_precio_usd |
| `/oportunidades` | GET | Listado paginado de oportunidades (fase 12) |
| `/oportunidades/{id}` | GET | Detalle de una oportunidad (fase 12) |

**Contrato de entrada:** 14 features — 6 numéricas imputables
(`superficie_cubierta`, `ambientes`, `dormitorios`, `banos`, `antiguedad`,
`expensas_usd`), 6 indicadores `*_informado` y 2 ordinales (`barrio_ordinal`,
`tipo_propiedad_ordinal`).

---

## ETL periódico de oportunidades

El ETL periódico (`scripts/etl_oportunidades.py`, fase 12) es el corazón del
sistema de detección de oportunidades de compra:

1. **Scraping:** `scripts/etl_oportunidades.py scrape` re-escanea las
   publicaciones nuevas de CABA (dataset separado
   `data/raw/propiedades_nuevas.csv`, con `revision_periodica`).
2. **Predicción + clasificación:** `scripts/etl_oportunidades.py etl` cura en
   memoria, predice con el bundle del champion, clasifica cada propiedad
   (`clasificar_por_diferencia`, umbral ±10 %) y persiste en PostgreSQL.
3. **Dedup:** solo se procesan propiedades nuevas (`ids_procesados` contra la base).

### Clasificación

| Ratio `precio_predicho_usd / precio_publicado_usd` | Clasificación |
|---|---|
| > 1.10 | **Buena compra** |
| 0.90 – 1.10 | **Precio justo** |
| < 0.90 | **Mala compra** |
| — | **Sin clasificar** (precio publicado inválido) |

### Persistencia PostgreSQL

`src/real_estate/persistencia/` (SQLAlchemy 2.0 Core):

- `config.py` — `ConfiguracionPostgres` (env `POSTGRES_*`, `dsn` opcional para tests)
- `db.py` — `crear_engine` (SQLite en memoria para tests, PostgreSQL con `pool_pre_ping`)
- `esquema.py` — Tabla `oportunidades` + `crear_tablas`
- `repositorio.py` — `upsert_oportunidades` (ON CONFLICT multi-dialecto),
  `ids_procesados`, `listar_oportunidades`, `obtener_oportunidad`

---

## Docker

```bash
# Build y levantar el stack (API + PostgreSQL)
make docker-build
make docker-up
make docker-logs
```

- **Dockerfile** multi-stage: build (`python:3.12-slim`) + runtime (venv slim)
- **requirements-api.txt**: solo el runtime de serving (fastapi, uvicorn,
  pydantic-settings, xgboost, numpy, pandas, scikit-learn)
- **docker-compose.yml** (fase 12): servicios `postgres` (healthcheck
  `pg_isready`, volumen persistente) + `api` (bundle montado como volumen de
  solo lectura, espera a que postgres esté sano)
- La API queda en `http://localhost:8000`

Para ejecutar el ETL dentro del stack:

```bash
docker compose run --rm api python scripts/etl_oportunidades.py --todos-los-barrios
```

---

## CI/CD

| Workflow | Propósito |
|---|---|
| `ci.yml` | Ruff (check + format), Mypy, Pytest con cobertura (Python 3.11/3.12) |
| `dvc.yml` | Valida el pipeline de datos (`dvc stage list`, `dvc status`, `dvc pull` best-effort) |
| `etl_oportunidades.yml` | ETL periódico: cron `0 11 */4 * *` (cada 4 días, 08:00 ART) + `workflow_dispatch`; PostgreSQL efímero; restaura el bundle del champion desde el secret `MODELO_BUNDLE_URL` |
| `cd_champion.yml` | CD del champion: dispara en push de `models/champion_actual.json` (fingerprint determinístico); restaura el bundle, hace smoke test y push de la imagen self-contained a GHCR |

### Publicación del champion (fase 13)

```bash
make export-model     # Generar el bundle del champion
make publicar-champion  # Publicar en GitHub Releases + escribir fingerprint
```

`scripts/publicar_champion.py` compara el fingerprint calculado con el
existente: si es idéntico, no re-publica (no re-dispara el CD). El fingerprint
es **determinístico** (sin marcas de tiempo), por lo que solo cambia si el
champion cambió de verdad. La URL del release es estable entre versiones.

---

## Testing

262 tests cubriendo unit e integración:

| Test | Cobertura |
|---|---|
| `tests/unit/test_cleaning.py` | Limpieza de tipos (14 formatos reales) |
| `tests/unit/test_scraper.py` | Parser, backoff 202, progreso, revision_periodica (42 tests) |
| `tests/unit/test_transformations.py` | Tipo de cambio, normalización moneda (22 tests) |
| `tests/unit/test_features.py` | Feature engineering, splits (22 tests) |
| `tests/unit/test_models.py` | Entrenamiento, métricas, XGBoost (15 tests) |
| `tests/unit/test_modelos_lineales.py` | Lasso/Ridge con escalado (9 tests) |
| `tests/unit/test_tracking.py` | MLflow: registro, Model Registry (10 tests) |
| `tests/unit/test_comparacion.py` | Comparación de runs y champion (7 tests) |
| `tests/unit/test_tuning.py` | Tuning de XGBoost (GridSearchCV) |
| `tests/unit/test_explainability.py` | SHAP (5 tests) |
| `tests/unit/test_evaluacion.py` | Evaluación profunda (14 tests) |
| `tests/unit/test_serving.py` | Bundle, ModeloPrediccion (9 tests) |
| `tests/unit/test_clasificacion.py` | Clasificación oportunidades (21 tests) |
| `tests/unit/test_repositorio.py` | Upsert multi-dialecto, dedup (11 tests) |
| `tests/unit/test_publicar_champion.py` | Fingerprint, publicación (15 tests) |
| `tests/integration/test_pipeline.py` | Curación end-to-end |
| `tests/integration/test_evaluar_nuevas.py` | Predicción sobre nuevas (6 tests) |
| `tests/integration/test_api.py` | API FastAPI completa (19 tests) |
| `tests/integration/test_etl_oportunidades.py` | ETL end-to-end con PostgreSQL (4 tests) |

Los tests que tocan red (tipo de cambio) usan `requests` mockado — ningún test
hace requests reales.

```bash
# Tests unitarios
python -m pytest tests/unit/

# Tests de integración
python -m pytest tests/integration/

# Todo + cobertura
make test
make coverage
```

---

## Quality gates

```text
Developer  →  Code  →  pre-commit  →  Ruff  →  Mypy  →  Git commit
    ↓
GitHub  →  GitHub Actions  →  CI  →  Pytest / Ruff / Mypy  →  Docker build
```

| Herramienta | Configuración |
|---|---|
| Ruff | `line-length=100`, target `py311`, reglas `E W F I B UP N SIM` |
| Mypy | `strict=true`, `python_version=3.12` |
| Pre-commit | Ruff (check + format), Mypy, check-yaml, check-toml, end-of-file-fixer |

```bash
make check   # ruff format --check + ruff check + mypy + pytest
make format  # ruff format
make lint    # ruff check
make typecheck  # mypy
```

---

## Documentación

| Archivo | Descripción |
|---|---|
| `docs/architecture.md` | Mapa vivo de la arquitectura, componentes, flujos de datos y estado actual |
| `docs/roadmap.md` | Roadmap de predicción de precios + detección de oportunidades (fases 1-6) |
| `notebooks/` | Notebooks 01-07 (EDA, feature engineering, model analysis, SHAP, evaluación, detección de oportunidades) |

### Versionado de datos (DVC)

`dvc.yaml` + `dvc.lock` versionan el pipeline de datos (raw → curation → features).
El remote local por defecto es `dvcstore/`.

```bash
dvc repro   # Reproducir el pipeline
dvc push    # Subir datos al remote
dvc pull    # Descargar datos del remote
dvc status  # Estado del workspace vs remote
```
