# Arquitectura del Proyecto — Real Estate Price Prediction

> **Documento vivo.** Este archivo es el mapa de arquitectura del proyecto.
> Se actualiza cada vez que se crea, modifica o elimina un archivo, o cuando
> aparece información relevante nueva. Cada entrada documenta: ruta, propósito,
> responsabilidades, dependencias, archivos que lo utilizan, archivos que
> utiliza, outputs y relaciones con otros componentes.

---

## 1. Resumen del proyecto

**`real-estate-price-prediction`** es un proyecto profesional de Data Science /
Machine Learning cuyo objetivo es **predecir precios de propiedades residenciales**
usando datos reales obtenidos mediante scraping de **Argenprop**.

Se concibe como un proyecto de **portfolio end-to-end** publicable en GitHub, que
demuestra todo el ciclo de vida de un proyecto de Machine Learning:

```text
Data Acquisition
      ↓
Data Curation
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Model Development
      ↓
Experiment Tracking
      ↓
Model Evaluation
      ↓
Explainability
      ↓
Deployment
```

### Objetivos

| Objetivo | Descripción |
|---|---|
| Problema real | Predecir precios de propiedades residenciales (CABA) |
| Datos reales | Publicaciones de Argenprop obtenidas con scraper propio |
| Ciclo completo | Adquisición → Curación → EDA → Features → Modelo → Tracking → Evaluación → Explicabilidad → Deploy |
| Portfolio | Proyecto público, entendible para cualquier persona en GitHub |

---

## 2. Filosofía y principios de diseño

El proyecto exige **calidad de software además de calidad de Machine Learning**.
No se acepta terminar con un notebook + un `model.pkl` sin más.

Se debe demostrar:

- **Reproducibilidad** (configuración centralizada, versionado de datos, semillas)
- **Versionado de datos** (DVC)
- **Testing** (pytest, unit + integration)
- **EDA** (análisis exploratorio, notebooks 01 y 02) ✔
- **Linting** (Ruff)
- **Type checking** (Mypy en modo `strict`)
- **CI** (GitHub Actions) ✔
- **Experiment tracking** (MLflow)
- **Model evaluation** (métricas rigurosas)
- **Explainability** (SHAP)
- **Containerization** (Docker / Docker Compose) ✔
- **Separación de responsabilidades** (paquetes por capa en `src/`)
- **Arquitectura clara y documentación técnica** (este documento como mapa vivo)

### Reglas

- **No agregar herramientas sin justificación.** Cada herramienta debe tener una
  responsabilidad clara dentro de la arquitectura.
- **Optuna queda excluido** por decisión explícita (no se agregará por popularidad).
- **DVC** es la dependencia (dev) encargada del versionado de datos. Vive como
  `dvc` en `pyproject.toml` y opera sobre `dvc.yaml` / `dvc.lock` (commit) más el
  cache local `.dvc/cache` (ignorado) y el remote local `dvcstore/` (ignorado).
- **Incrementalidad:** antes de implementar una pieza nueva hay que definir qué
  responsabilidad tiene, en qué capa vive, de qué depende, qué depende de ella,
  cómo afecta al flujo de datos, cómo se testea y cómo se documenta en este archivo.

---

## 3. Fuente de datos: Argenprop

### 3.1 Scraping

El scraper (`src/real_estate/ingestion/scraper.py`, invocado por
`scripts/scrape.py`) obtiene avisos de **venta** de todo tipo de propiedad
(departamentos, casas, PH, etc.) publicados en **Capital Federal**.

**Límite del sitio (v3):** Argenprop **corta toda búsqueda en la página 100**
con un HTTP 202 vacío (≈ 2.000 avisos por búsqueda), aunque el widget de
paginación muestre miles de páginas. Para superarlo se **segmenta la búsqueda**
por barrio (54 barrios de CABA) y/o tipo de propiedad; cada segmento tiene su
propio paginado y su propio cap de 100 páginas. El HTTP 202 también se usa como
**throttle anti-bot** ante rafagas de requests sostenidas; el scraper lo
distingue por página: en la página 100 es el cap del servidor (no se reintenta),
en páginas tempranas es un bloqueo transitorio (reintento con backoff
exponencial). El progreso por segmento se guarda en un JSON
(`--progreso`), de modo que una corrida interrumpida reanuda desde la última
página procesada y los segmentos completos se saltan.

**Decisión de diseño clave — separación de etapas:**

```text
SCRAPING
    ↓
RAW DATA
    ↓
DATA CURATION
```

El scraper **ya cumplió su responsabilidad**: obtener los datos disponibles de las
publicaciones. **No** limpia ni normaliza; eso pertenece a Data Curation. Por eso
el raw contiene valores como:

- `"300 m² cubie."`
- `"90 m² cubie."`
- `"17 años"`
- `"&plus; $2.200.000 expensas"`

Esto es intencional.

### 3.2 Dataset raw

**Ruta:** `data/raw/propiedades_argenprop.csv`

**Estado real:** 2.005 registros, 20 columnas (campaña de scraping actual;
el número crece con cada corrida incremental).

| Columna | Tipo actual | Notas |
|---|---|---|
| `id` | numérico | ID interno del aviso en Argenprop (atributo `data` del link, confiable) |
| `link` | str | URL del aviso |
| `titulo` | str | Título del aviso |
| `descripcion` | str | Copete/descripción corta |
| `tipo_propiedad` | str | departamento, casa, ph, etc. (inferido de la URL) |
| `idtipopropiedad` | numérico | Código numérico interno del sitio |
| `barrio` | str | Ubicación |
| `sub_barrio` | str | Ubicación (puede faltar) |
| `precio` | numérico | Ya viene numérico (atributo `data`, confiable) |
| `moneda` | str | `USD` o `ARS` |
| `expensas` | texto | Formato con símbolos y separadores; `NaN` si el aviso no las publica |
| `superficie_cubierta` | texto | Ej.: `"90 m² cubie."` |
| `superficie_semicubierta` | texto | Puede faltar |
| `superficie_total` | texto | Puede faltar |
| `ambientes` | numérico | Extraído de la URL del aviso (dato confiable) |
| `dormitorios` | numérico | Atributo del aviso; puede faltar si no se declaró |
| `banos` | numérico | **Suele quedar vacío** (el listado solo muestra 2-3 características por tarjeta) |
| `cocheras` | numérico | **Casi siempre vacío** (solo aparece en la página de detalle) |
| `antiguedad` | texto | Ej.: `"17 años"`, `"A estrenar"`, `"En pozo"` |
| `fecha_scrape` | fecha | Cuándo se bajó el aviso |

> **Nota sobre `banos` y `cocheras`:** los valores faltantes no son un bug del
> scraper, sino una limitación del listado. Completarlos exigiría visitar la página
> de detalle de cada aviso (una request extra por aviso). Decisión pendiente.

---

## 4. Ciclo de vida objetivo

```text
                         ┌─────────────────┐
                         │    Argenprop    │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    INGESTION    │
                         │ Scraper / Parser│
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    RAW DATA     │
                         │       DVC       │
                         └────────┬────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │      DATA CURATION       │
                    │ Cleaning / Validation /  │
                    │ Transformations          │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                         ┌─────────────────┐
                         │ PROCESSED DATA  │
                         │       DVC       │
                         └────────┬────────┘
                                  │
                         ┌────────┴────────┐
                         ▼                 ▼
                    ┌──────────┐    ┌───────────────┐
                    │   EDA    │    │    FEATURES   │
                    │ Notebooks│    │ Engineering   │
                    └────┬─────┘    └───────┬───────┘
                         │                   │
                         └─────────┬─────────┘
                                   ▼
                          ┌──────────────────┐
                          │ MODEL DEVELOPMENT│
                          │                  │
                          │ Baseline         │
                          │ XGBoost          │
                          └────────┬─────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │      MLFLOW      │
                          │                  │
                          │ Experiments      │
                          │ Parameters       │
                          │ Metrics          │
                          │ Artifacts        │
                          │ Model Registry   │
                          └────────┬─────────┘
                                   │
                         ┌─────────┴─────────┐
                         ▼                   ▼
                  ┌─────────────┐      ┌─────────────┐
                  │    SHAP     │      │    MODEL    │
                  │Explainability│     │   Artifact  │
                  └─────────────┘      └──────┬──────┘
                                              │
                                              ▼
                                      ┌───────────────┐
                                      │Serving bundle │
                                      │  (fase 10)    │
                                      └──────┬────────┘
                                             │
                                             ▼
                                      ┌───────────────┐
                                      │  FastAPI API  │
                                      │  (fase 10)    │
                                      └───────────────┘
```

---

## 5. Flujo de datos objetivo

```text
Argenprop
    ↓
Scraper
    ↓
data/raw/
    ↓
DVC
    ↓
Data Curation
    ↓
data/processed/
    ↓
Feature Engineering
    ↓
Train / Validation / Test
    ↓
Baseline
    ↓
XGBoost
    ↓
Evaluation
    ↓
MLflow
    ↓
Best Model
    ├──────────────→ SHAP
    │
    └──────────────→ Serving bundle (models/, fase 10)
            ↓
    FastAPI API (/health, /predict)
            ↓
    make serve (uvicorn)
```

---

## 6. Data Curation — definición conceptual

**Estado: IMPLEMENTADA en `src/real_estate/curation/`**, con los módulos
`cleaning.py`, `transformations.py`, `validation.py` y el orquestador
`pipeline.py` (invocado por `scripts/curate.py`). Esta sección documenta la
definición conceptual; el detalle de implementación está en la sección 9.

Responsabilidades definidas:

### 1. Limpieza de tipos

Convertir campos que vienen como texto a valores numéricos. Hay que contemplar
formatos reales: separadores de miles, símbolos de moneda, `m²`, texto adicional,
etc.

| Ejemplo crudo | → | Valor esperado |
|---|---|---|
| `"300 m² cubie."` | → | `300` |
| `"2.200.000"` | → | `2200000` |
| `"17 años"` | → | `17` |

### 2. Normalización de moneda

Trabajar con una moneda común, **probablemente USD**. La conversión debe
considerar el **tipo de cambio correspondiente a la fecha de scraping**
(`fecha_scrape`).

### 3. Manejo de valores faltantes

Columnas típicamente afectadas: `expensas`, `antiguedad`, `banos`,
`superficie_total`, `cocheras`.

**Implementación actual (sin imputación):** no se imputa nada automáticamente.
La ausencia de información es una señal válida ("el aviso no informó este dato").
`transformations.py` crea indicadores binarios `{columna}_informado` por cada
columna afectada, y los `NaN` quedan como están para que el modelo decida cómo
tratarlos. La estrategia de imputación (si existiera) se definiría en Feature
Engineering.

### 4. Validación

Validar que los datos tengan sentido y detectar anomalías:

- `precio > 0`
- `superficie > 0`
- `ambientes >= 1`
- Detección de valores anómalos o inconsistentes

---

## 7. Stack tecnológico

| Área | Herramienta | Responsabilidad | Estado |
|---|---|---|---|
| Lenguaje | Python | Lenguaje del proyecto | ✔ en `pyproject.toml` (`>=3.11,<3.14`) |
| Datos | NumPy, Pandas | Manipulación de datos | ✔ dependencia |
| Scraping | Requests, BeautifulSoup, lxml | Adquisición de datos | ✔ dependencia |
| Machine Learning | Scikit-learn, XGBoost | Modelos | ✔ dependencia |
| Experiment tracking | MLflow | Seguimiento de experimentos | ✔ dependencia |
| Explainable AI | SHAP | Explicabilidad de modelos | ✔ dependencia |
| Data versioning | DVC | Versionado de datos | ✔ dependencia dev (`dvc.yaml` + `dvc.lock`) |
| Testing | Pytest, pytest-cov | Tests | ✔ dependencia dev |
| Code quality | Ruff, Mypy | Lint + type check | ✔ dependencia dev |
| Git hooks | pre-commit | Gates antes del commit | ✔ dependencia dev |
| CI/CD | GitHub Actions | Integración continua | ✔ `.github/workflows/ci.yml` |
| Containerization | Docker, Docker Compose | Contenedores | ✔ fase 11 (`Dockerfile` multi-stage + `docker-compose.yml`) |
| API / Serving | FastAPI, Uvicorn | API de predicción (fase 10) | ✔ dependencia |
| Configuración | Pydantic Settings, python-dotenv | Configuración y env vars | ✔ dependencia |
| Jupyter | Jupyter, IPython kernel | Notebooks de análisis | ✔ dependencia dev |

**Exclusiones explícitas:** Optuna (no se usará).

---

## 8. Estructura del repositorio

Árbol conceptual anotado con el **estado real**:

- ✔ = existe / implementado
- 🏗 = carpeta o archivo creado pero vacío (esqueleto)
- ✘ = pendiente de crear

```text
real-estate-price-prediction/
│
├── README.md                      ✔ (actualizado: documenta scrape y curate vía scripts/)
├── pyproject.toml                 ✔
├── MakeFile                       ✔ (nota: el archivo real se llama MakeFile, no Makefile)
│
├── .gitignore                     ✔
├── .dockerignore                  ✔
├── .editorconfig                  ✔
├── .pre-commit-config.yaml        ✔
├── .env.example                   ✔
├── .env                           ✔ (local, NO versionar — ignorado por .gitignore)
│
├── Dockerfile                     ✔
├── docker-compose.yml             ✔
├── requirements-api.txt           ✔
│
├── configs/
│   └── config.yaml                🏗 carpeta creada, archivo pendiente
│
├── data/
│   ├── raw/    propiedades_argenprop.csv            ✔ (2.005 registros, trackeado por DVC)
│   │           propiedades_argenprop.csv.dvc        ✔ (pointer de DVC, sí se versiona)
│   ├── interim/                             🏗 vacía
│   ├── processed/ propiedades_argenprop_curado.csv  ✔ (32 columnas, output DVC etapa curar)
│   │              propiedades_argenprop_features.csv ✔ (16 columnas, output DVC etapa features)
│   └── external/ tipo_cambio_blue.csv       ✔ (histórico dólar blue, 5.702 fechas, DVC)
│                tipo_cambio_blue.csv.dvc    ✔ (pointer de DVC)
│
├── dvc.yaml                        ✔ (etapas curar y features)
├── dvc.lock                        ✔ (hashes md5 de deps y outs)
│
├── notebooks/                      ✔ 01_eda_estructura_y_calidad, 02_eda_precio_y_caracteristicas, 03_feature_engineering, 04_model_analysis, 05_shap_analysis, 06_model_evaluation
│
├── src/
│   └── real_estate/
│       ├── __init__.py            ✔
│       ├── ingestion/
│       │   ├── __init__.py         ✔
│       │   └── scraper.py          ✔ (lógica completa del scraper)
│       ├── curation/
│       │   ├── __init__.py         ✔
│       │   ├── cleaning.py         ✔ (tipos texto → número, fecha)
│       │   ├── transformations.py  ✔ (moneda → USD, indicadores missing)
│       │   ├── validation.py       ✔ (reglas de coherencia, solo reporte)
│       │   └── pipeline.py         ✔ (orquestador curar_dataset)
│       ├── features/
│       │   ├── __init__.py         ✔
│       │   ├── transformations.py  ✔ (selección, target log, ordinal, imputación)
│       │   └── pipeline.py         ✔ (orquestador construir_features + splits)
│       ├── models/                ✔ entrenamiento.py (baseline, XGBoost) + modelos_lineales.py
│       │                             (Lasso/Ridge con escalado, roadmap fase 4, sin fuga)
│       ├── explainability/        ✔ shap_analysis.py (SHAP: valores, base,
│       │                             figuras, guardado)
│       ├── evaluacion/            ✔ analisis.py (métricas detalladas, residuos,
│       │                             error por segmento, sesgo por rango)
│       ├── tracking/              ✔ experimentos.py (MLflow: params, métricas,
│       │                             artefactos, Model Registry; registrar_lineales)
│       ├── serving/               ✔ fase 10: modelo.py (ModeloPrediccion),
│       │                             persistencia.py (guardar/cargar bundle)
│       ├── api/                   ✔ fase 10: app.py (FastAPI, /health + /predict),
│       │                             schemas.py, config.py
│       └── utils/                 🏗 vacío
│
├── scripts/
│   ├── scrape.py                   ✔ (entry point de adquisición)
│   ├── scrape_nuevas.py            ✔ (adquisición de nuevas publicaciones a dataset separado, para corridas programadas)
│   ├── curate.py                   ✔ (entry point de curación)
│   ├── features.py                 ✔ (entry point de feature engineering)
│   ├── train.py                    ✔ (entry point de modelado + tracking MLflow)
│   ├── train_lineales.py           ✔ roadmap fase 4 (entry point de modelos lineales: Lasso/Ridge)
│   ├── exportar_modelo.py          ✔ fase 10 (entry point de exportación del bundle de serving)
│   ├── evaluar_nuevas.py           ✔ roadmap fase 2 (entry point de predicción sobre nuevas publicaciones)
│   └── clasificar_ofertas.py       ✔ roadmap fase 3 (entry point de clasificación buena/mala compra)
│   (se planifican evaluate/explain)
│
├── tests/
│   ├── unit/                       ✔ (cleaning, scraper, transformations, features, models, modelos_lineales, tracking, explainability, evaluacion, serving)
│   └── integration/                ✔ (pipeline de curación, API FastAPI)
│
├── models/
│   └── modelo_precio_propiedades/  ✔ fase 10: bundle de serving (modelo, preprocesamiento,
│                                       features, metadata; gitignored, se regenera con
│                                       `make export-model`)
│
├── reports/
│   ├── figures/                    ✔ en uso (figuras SHAP fase 7 y de evaluación fase 8; gitignored)
│   └── metrics/                    🏗 vacía
│
├── mlruns/                         ✔ (store local de MLflow, gitignored)
│
├── docs/
│   ├── architecture.md             ✔ (este documento)
│   └── roadmap.md                  ✔ (roadmap de predicción + detección de oportunidades de compra)
│
└── .github/
    └── workflows/
        ├── ci.yml                  ✔ (lint + type check + tests)
        └── dvc.yml                 ✔ (valida etapas, estado y pull best-effort)
```

> **Nota:** el repositorio está versionado en
> `https://github.com/matiasbelsito7/real-estate-price-prediction.git`; cada fase
> se integra con pre-commit, se commitea y se pushea (workflow en §14).

---

## 9. Registro de componentes (mapa vivo)

### 9.1 Configuración raíz

#### `pyproject.toml`

| Atributo | Valor |
|---|---|
| **Propósito** | Centralizar la configuración del proyecto Python |
| **Responsabilidades** | Metadata del proyecto, versión de Python, dependencias prod/dev, package discovery, config de Pytest, Ruff, Mypy y coverage |
| **Build** | setuptools (`setuptools>=75`), layout `src/` (`[tool.setuptools.packages.find] where=["src"]`) |
| **Python** | `>=3.11,<3.14` |
| **Dependencias prod** | numpy, pandas, requests, beautifulsoup4, lxml, scikit-learn, xgboost, mlflow, shap, fastapi, uvicorn, pydantic-settings, python-dotenv |
| **Dependencias dev** | pytest, pytest-cov, httpx, ruff, mypy, pandas-stubs, pre-commit, jupyter, ipykernel, matplotlib |
| **Pytest** | `testpaths=["tests"]`, `pythonpath=["src"]`, addopts `-ra --strict-markers --strict-config` |
| **Ruff** | `line-length=100`, target `py311`, reglas `E W F I B UP N SIM` (ignora `E501`), formatter: comillas dobles, espacios |
| **Mypy** | `strict=true`, `python_version=3.12`, files `["src","tests"]`, varias advertencias estrictas |
| **Coverage** | `branch=true`, `source=["src/real_estate"]`, exclude `pragma: no cover` / `TYPE_CHECKING` / `raise NotImplementedError` |
| **Dependencias** | `README.md` (readme), paquetes en `src/` |
| **Usado por** | `make install`, `make install-dev`, pre-commit, CI |
| **Outputs** | Entorno Python instalable |

#### `MakeFile` (nombre real del archivo)

| Atributo | Valor |
|---|---|
| **Propósito** | Interfaz estándar para ejecutar tareas del proyecto |
| **Responsabilidades** | Envolver comandos de instalación, calidad, testing y limpieza |
| **Targets actuales** | `install`, `install-dev`, `scrape`, `curate`, `features`, `train`, `train-lineales`, `export-model`, `serve`, `docker-build`, `docker-up`, `docker-down`, `docker-logs`, `dvc-repro`, `dvc-push`, `dvc-pull`, `dvc-status`, `format`, `lint`, `typecheck`, `test`, `coverage`, `check`, `clean` (además de `help`) |
| **Targets futuros** | `evaluate`, `explain` — solo cuando los componentes existan realmente |
| **Dependencias** | `pyproject.toml` (comandos pip/pytest/ruff/mypy) |
| **Usado por** | Desarrolladores, CI (futuro) |
| **Nota** | El esquema conceptual lo llama `Makefile`; el archivo real en disco es `MakeFile`. |

#### `pre-commit-config.yaml`

| Atributo | Valor |
|---|---|
| **Propósito** | Gates de calidad antes de cada commit |
| **Responsabilidades** | Ejecutar Ruff (check + format), Mypy, y validaciones generales |
| **Hooks** | `ruff-check` (`--fix`), `ruff-format`, `mypy` (+ `pandas-stubs`, `types-requests`, `pytest`), `check-yaml`, `check-toml`, `end-of-file-fixer`, `trailing-whitespace`, `check-added-large-files` (`--maxkb=1000`) |
| **Dependencias** | `pyproject.toml` (versiones de herramientas), repos pre-commit |
| **Usado por** | Flujo de calidad del desarrollador |

#### `.env.example`

| Atributo | Valor |
|---|---|
| **Propósito** | Documentar las variables de entorno requeridas |
| **Variables** | `APP_ENV`, `LOG_LEVEL`, `DATA_DIR`, `MLFLOW_TRACKING_URI`, `SCRAPER_REQUEST_TIMEOUT`, `SCRAPER_DELAY_SECONDS`, `MODELO_DIR` (fase 10) |
| **Usado por** | pydantic-settings / python-dotenv (config del proyecto) |

#### Otros archivos de configuración

| Archivo | Propósito |
|---|---|
| `.gitignore` | Ignora venvs, caches, `.env`, datasets generados localmente, ML artifacts, etc. |
| `.dockerignore` | Excluye del contexto Docker: Git, entornos locales, datasets locales, caches, ML artifacts, secrets |
| `.editorconfig` | UTF-8, LF, espacios, indentación consistente, final newline, config específica para Makefile |
| `.env` | Variables reales locales. **No versionar.** Existe actualmente con valores locales |

### 9.2 Capa de INGESTION — `src/real_estate/ingestion/`

#### `scraper.py`

| Atributo | Valor |
|---|---|
| **Ruta** | `src/real_estate/ingestion/scraper.py` |
| **Propósito** | Obtener avisos de venta de propiedades en CABA desde Argenprop |
| **Responsabilidades** | Paginar el listado, parsear tarjetas (Requests + BeautifulSoup + lxml), extraer los 20 campos, guardar CSV incrementalmente, segmentar la búsqueda para superar el cap de 100 páginas del sitio, manejar el bloqueo HTTP 202 con backoff y guardar el progreso por segmento para reanudar corridas interrumpidas |
| **Estado** | ✔ Funcional, probado con datos reales. **v3 (feb 2026):** segmentación por barrio/tipo, manejo del cap 202, reintentos con backoff y progreso JSON |
| **Dependencias** | `requests`, `beautifulsoup4`, `lxml` |
| **Constantes** | `BASE_URL`, `HEADERS`, `COLUMNS` (20), `TIPOS_PROPIEDAD` (18), `ICONO_A_COLUMNA`, `RE_AMBIENTES_EN_URL`, `STATUS_BLOQUEO` (202), `MAX_PAGINAS_SERVICIO` (100), `BARRIOS_CABA` (54 slugs) |
| **Funciones** | `construir_url_pagina` (con `base_url` para segmentos), `construir_url_segmento` (tipo/barrio), `texto_o_none`, `detectar_tipo_propiedad`, `extraer_ambientes_de_url`, `extraer_features_de_tarjeta`, `parsear_listing`, `cargar_ids_existentes`, `asegurar_encabezado`, `guardar_filas`, `cargar_progreso`, `guardar_progreso`, `pagina_de_reanudacion`, `scrapear` |
| **Comportamiento** | Resumen por `id`: si se corta, al volver a correr no re-baja avisos ya presentes. Precio/moneda desde atributos `data` del link; ambientes desde la URL del aviso; alerta si 3 páginas seguidas no traen features (cambio de estructura del sitio). **202:** si `pagina >= 100` es el cap del servidor → segmento completo; si no, reintenta con backoff exponencial (`backoff_202_inicial * 2^n`, máx 120 s), pausa larga (`pausa_bloqueo`) al agotar los reintentos (máx 2) y luego abandona el segmento como incompleto. 404 y página vacía → segmento completo. Guarda `{"pagina": ultima_ok, "completo": bool}` en el archivo de progreso tras cada página. **`revision_periodica` (bool, default False):** si es True, re-escaneea el segmento aunque el progreso lo tenga como completo (útil para corridas programadas que capturan publicaciones nuevas; el dedup por `id` evita duplicados y una corrida interrumpida sigue reanudando) |
| **Usado por** | `scripts/scrape.py` |
| **Outputs** | `data/raw/propiedades_argenprop.csv` (o `--output`), `data/raw/progreso_scrape.json` (o `--progreso`) |
| **Relación** | Alimenta la etapa de Data Curation. `banos`/`cocheras` quedan mayormente vacíos por limitación del listado (no es un bug) |

#### `scripts/scrape.py`

| Atributo | Valor |
|---|---|
| **Ruta** | `scripts/scrape.py` |
| **Propósito** | Entry point CLI de adquisición de datos |
| **Responsabilidades** | Parser de argumentos, bootstrap de `sys.path` para importar `real_estate` sin instalar el paquete, manejo de `KeyboardInterrupt`, orquestar segmentos (uno por barrio y/o tipo) |
| **CLI** | `--output`, `--max-paginas`, `--pagina-inicio`, `--delay-min`, `--delay-max`, `--html-debug`, `--tipo` (departamentos, casas, ph…), `--barrios` (slugs separados por coma), `--todos-los-barrios` (54 segmentos), `--progreso`, `--reintentos-202`, `--backoff-202`, `--pausa-bloqueo` |
| **Usa** | `real_estate.ingestion.scraper.scrapear`, `construir_url_segmento`, `BARRIOS_CABA` |
| **Usado por** | `make scrape` (MakeFile) |
| **Outputs** | CSV en `data/raw/` + JSON de progreso |
| **Uso típico** | `python scripts/scrape.py --todos-los-barrios --tipo departamentos` recorre 54 segmentos (≈ 108.000 avisos potenciales); `python scripts/scrape.py --barrios palermo,recoleta,caballito` acota a 3 segmentos |

### 9.3 Capa de CURATION — `src/real_estate/curation/`

#### `cleaning.py`

| Atributo | Valor |
|---|---|
| **Ruta** | `src/real_estate/curation/cleaning.py` |
| **Propósito** | Limpieza y conversión de tipos (texto crudo → numérico) |
| **Responsabilidades** | `limpiar_numero` (quita entidades HTML `&plus;`/`&nbsp;`, símbolo `$`, saltos de línea, toma el bloque numérico inicial y resuelve separadores de miles/decimales), `limpiar_columnas_numericas` (aplica a `NUMERIC_COLUMNS`), `limpiar_expensas`, `preparar_fecha` (`fecha_scrape` → datetime UTC) |
| **Estado** | ✔ Implementado. **Fix de bug preexistente:** el texto con sufijo (p. ej. `"&plus; $2.200.000 expensas"`) rompía el chequeo de separador de miles y devolvía valores 1000x menores (`330.0` en vez de `330000`); afectaba las expensas reales. Hoy produce el valor documentado `2200000.0` |
| **Dependencias** | `pandas`, `re` |
| **Usado por** | `pipeline.py` |
| **Outputs** | Columnas numéricas limpias |

#### `transformations.py`

| Atributo | Valor |
|---|---|
| **Ruta** | `src/real_estate/curation/transformations.py` |
| **Propósito** | Transformaciones de valor: moneda común y señales de datos informados |
| **Responsabilidades** | `obtener_tipo_cambio` (consulta histórica por fecha, retrocede días hábiles, usa cotización "venta"), `cargar_tipo_cambio_historico` (lee el CSV versionado a `{fecha: venta}`; si no existe, cae a la API), `construir_tabla_tipo_cambio` (una consulta por fecha única; usa el histórico local como fuente primaria y la API como fallback), `normalizar_moneda` (crea `tipo_cambio_ars_usd` y `precio_usd`; USD se copia, ARS se divide), `normalizar_expensas` (`expensas_usd`), `crear_indicadores_missing` (`{columna}_informado` int8) |
| **Estado** | ✔ Implementado. La conversión usa como fuente primaria el dataset versionado `data/external/tipo_cambio_blue.csv` (`RUTA_TIPO_CAMBIO_HISTORICO`); las fechas no cubiertas por el histórico caen a la API |
| **Dependencias** | `pandas`, `requests`, `csv`, `os` |
| **FX API** | `https://api.argentinadatos.com/v1/cotizaciones/dolares/{market}/{date}`, `FX_MARKET = "blue"` (opciones: oficial, blue, bolsa, contadoconliqui, mayorista, etc.). Fallback: solo se consulta cuando la fecha no está en el histórico local |
| **Usado por** | `pipeline.py` |
| **Outputs** | `precio_usd`, `expensas_usd`, `tipo_cambio_ars_usd`, `*_informado` |

#### `validation.py`

| Atributo | Valor |
|---|---|
| **Ruta** | `src/real_estate/curation/validation.py` |
| **Propósito** | Validar coherencia del dataset sin modificarlo (solo reporta) |
| **Responsabilidades** | `validar` devuelve un `DataFrame` resumen con reglas: `precio > 0`, `superficie > 0` (cubierta/semicubierta/total), `ambientes >= 1`. Los `NaN` no se consideran inválidos |
| **Estado** | ✔ Implementado |
| **Dependencias** | `pandas` |
| **Usado por** | `pipeline.py` |
| **Outputs** | Reporte de validación (impreso + DataFrame) |

#### `pipeline.py`

| Atributo | Valor |
|---|---|
| **Ruta** | `src/real_estate/curation/pipeline.py` |
| **Propósito** | Orquestar la curación completa |
| **Responsabilidades** | `curar_dataset(df)`: `preparar_fecha` → `limpiar_columnas_numericas` → `limpiar_expensas` → `normalizar_moneda` → `normalizar_expensas` → `crear_indicadores_missing` → `validar`. `curar_csv(input, output)` lee el CSV, ejecuta las etapas y guarda (crea `data/processed/` si falta). `mostrar_dataset` / `mostrar_dataset_curado` imprimen resúmenes |
| **Estado** | ✔ Implementado, probado end-to-end con datos reales (2.005 filas, 32 columnas) |
| **Dependencias** | `cleaning.py`, `transformations.py`, `validation.py`, `pandas` |
| **Usado por** | `scripts/curate.py` |
| **Outputs** | `data/processed/propiedades_argenprop_curado.csv` |

#### `scripts/curate.py`

| Atributo | Valor |
|---|---|
| **Ruta** | `scripts/curate.py` |
| **Propósito** | Entry point CLI de Data Curation |
| **Responsabilidades** | Parser de argumentos, bootstrap de `sys.path`, ejecución de `curar_csv` |
| **CLI** | `--input` (default `data/raw/propiedades_argenprop.csv`), `--output` (default `data/processed/propiedades_argenprop_curado.csv`) |
| **Usa** | `real_estate.curation.pipeline.curar_csv` |
| **Usado por** | `make curate` (MakeFile) |
| **Outputs** | CSV curado en `data/processed/` |

#### `scripts/download_tipo_cambio.py`

| Atributo | Valor |
|---|---|
| **Ruta** | `scripts/download_tipo_cambio.py` |
| **Propósito** | Descarga el histórico completo del dólar blue desde ArgentinaDatos y lo guarda como dataset versionado |
| **Responsabilidades** | `descargar_historico` (valida que la API devuelva una lista), `guardar_historico` (CSV `fecha,compra,venta`), `main` (imprime cantidad, rango y avisa fechas duplicadas) |
| **CLI** | `--output` (default `data/external/tipo_cambio_blue.csv`) |
| **Usado por** | Mantenimiento manual; el dataset resultante alimenta `normalizar_moneda` sin depender de la API |
| **Outputs** | `data/external/tipo_cambio_blue.csv` (5.702 fechas, 2011-01-03 → 2026-08-15, trackeado con DVC) |

### 9.4 Paquete `src/real_estate/` — subpaquetes

Paquete descubrible vía `pyproject.toml` (layout `src/`). Import como
`real_estate.*`. `ingestion/`, `curation/`, `features/`, `models/`,
`tracking/`, `explainability/` y `evaluacion/` ya están implementados
(secciones 9.2, 9.3, 9.4b, 9.4c, 9.4d, 9.4e y 9.4f). `utils/` existe pero
está vacío:

| Subpaquete | Responsabilidad |
|---|---|
| `features/` | ✔ Implementado: selección de columnas, target `log_precio_usd`, codificación ordinal por mediana de precio, imputación por mediana, splits train/val/test (sección 9.4b) |
| `models/` | ✔ Implementado: entrenamiento y evaluación con preprocesamiento sin fuga (sección 9.4c) |
| `tracking/` | ✔ Implementado: integración con MLflow — experimentos, params, métricas, artefactos y Model Registry (sección 9.4d) |
| `explainability/` | ✔ Implementado: SHAP — valores y base, importancia global, figuras (beeswarm/barras) y guardado PNG (sección 9.4e) |
| `evaluacion/` | ✔ Implementado: evaluación profunda — métricas detalladas, residuos por observación, error por segmento y sesgo por rango de precio (sección 9.4f) |
| `utils/` | Utilidades transversales (config con pydantic-settings, logging, etc.) |

#### 9.4b `features/` (implementado en la fase 4)

| Archivo | Contenido |
|---|---|
| `transformations.py` | `COLUMNAS_DESCARTAR`/`IMPUTAR`/`INDICADOR`/`CATEGORICAS`, `PRECIO_MINIMO_USD` (descarta artefactos del scraping, p. ej. precio 1 USD), `seleccionar_columnas`, `crear_target_log`, `crear_orden_mediana`, `codificar_ordinal` (`CODIGO_DESCONOCIDO = -1`), `crear_imputador`, `aplicar_imputacion` |
| `pipeline.py` | `construir_features` (orquesta las 4 etapas), `dividir_train_val_test` (80/10/10 reproducible vía `random_state`), `mostrar_features` |

Decisiones de diseño (de las conclusiones de EDA 01 y 02):
- Columnas sin señal descartadas: cobertura < 3 % (+ sus indicadores), `sub_barrio` (88,8 % faltante), texto libre, identificadores y redundancias de moneda.
- Target `log_precio_usd` (el log es aproximadamente normal); RMSE en log = error relativo.
- Codificación ordinal de `barrio` (44 categorías) y `tipo_propiedad` (7) por mediana de precio — apta para árboles (modelo planificado).
- API `crear_*` / `aplicar_*`: los ajustes se aprenden solo sobre train y se reaplican a val/test (sin fuga de información).
- Filtro de precios: se descartan < `PRECIO_MINIMO_USD` (1.000), incluye 6 filas del real (4 con precio ≤ 0 y 2 artefactos de scraping con precio 1 USD).

Artifactos: `data/processed/propiedades_argenprop_features.csv` (1.999 filas × 16 columnas, 0 faltantes).

#### 9.4c `models/` (implementado en la fase 5)

| Archivo | Contenido |
|---|---|
| `entrenamiento.py` | `Preprocesamiento` (ordenes ordinales + imputador, ajustados solo sobre train), `ResultadoEntrenamiento` (métricas y modelos), `ajustar_preprocesamiento` / `aplicar_preprocesamiento` (sin fuga), `separar_features_target`, `calcular_metricas` (RMSE log, RMSE USD, R²), `mostrar_metricas`, `entrenar_baseline` (mediana), `entrenar_xgboost`, `entrenar_y_evaluar` (pipeline completo) |
| `modelos_lineales.py` | `ALPHA_DEFAULT` (1.0), `MAX_ITER_DEFAULT` (10.000), `ResultadoLineales` (métricas de lasso/ridge en val, mejor en test, pipelines ajustados y `ajustes`), `crear_pipeline_lineal` (`StandardScaler` + regresor lineal), `entrenar_lasso`, `entrenar_ridge`, `entrenar_y_evaluar_lineales` (mismo preprocesamiento y features que XGBoost; entrena ambos, compara en val y evalúa al mejor en test) |

Decisiones de diseño:
- **Sin fuga de información:** no se reutiliza `construir_features` (codifica e imputa sobre todo el dataset). Se ajustan sobre train los ordenes ordinales (`crear_orden_mediana`) y la imputación por mediana (`crear_imputador`), y se reaplican a val/test con `aplicar_preprocesamiento`. Categorías no vistas en train -> `CODIGO_DESCONOCIDO (-1)`.
- **Split 80/10/10 reproducible** vía `dividir_train_val_test` (`random_state=42`).
- **Baseline (mediana)** como referencia mínima con `DummyRegressor(strategy="median")`.
- **XGBoost** con parámetros por defecto en `PARAMS_XGBOOST_DEFAULT` (300 árboles, depth 4, lr 0.05, regularización) y `random_state` para reproducibilidad.
- **Métricas sobre el target logarítmico:** RMSE log (≈ error relativo), RMSE USD (deshaciendo el log) y R².
- **Escalado dentro del pipeline (roadmap fase 4):** los lineales son sensibles a la escala; `StandardScaler` se ajusta junto con el modelo sobre train (sin fuga hacia val/test). Mismas features y mismo split que XGBoost → comparación justa.

Resultados sobre el dataset real (fase 5):

| Modelo | RMSE log (val) | RMSE USD (val) | R² (val) | RMSE log (test) | R² (test) |
|---|---|---|---|---|---|
| baseline (mediana) | 0.6423 | $185.135 | -0.0025 | — | — |
| XGBoost | 0.2718 | $112.963 | 0.8205 | 0.3040 | 0.7830 |

XGBoost reduce el RMSE log en ~58 % respecto del baseline; error relativo
mediano en test: 2,5 %.

Modelos lineales sobre el dataset real (roadmap fase 4,
`scripts/train_lineales.py`, una corrida MLflow por modelo):

| Modelo | RMSE log (val) | RMSE USD (val) | R² (val) | RMSE log (test) | R² (test) |
|---|---|---|---|---|---|
| Lasso (alpha=1.0) | 0.6421 | $183.046 | -0.0019 | — | — |
| Ridge (alpha=1.0) | 0.3370 | $133.440 | 0.7240 | 0.3372 | 0.7330 |

Ridge es el mejor lineal (gana en val y se evalúa en test), pero **XGBoost
sigue siendo el campeón** (test: RMSE log 0.3040 vs 0.3372). Lasso con
alpha=1.0 anula todos los coeficientes y degenera al baseline (R² ≈ 0); la
regularización se puede bajar con `--alpha-lasso` (el pipeline lo soporta).

#### 9.4d `tracking/` (implementado en la fase 6)

| Archivo | Contenido |
|---|---|
| `experimentos.py` | `configurar_tracking` (URI + experimento, creándolo si no existe), `registrar_resultado` (abre la corrida: loguea params, métricas, artefacto JSON `resumen_entrenamiento.json`, modelo XGBoost con firma vía `infer_signature` y lo versiona en el Model Registry; devuelve `(run_id, version)`), `registrar_lineales` (una corrida por modelo lineal: params, métricas `val_*` y `test_*` solo para el mejor, artefacto JSON `resumen_lineal.json` y pipeline con firma; devuelve `[(nombre, run_id)]`), `finalizar_corrida` |

Decisiones de diseño:
- **`models/entrenamiento.py` se mantiene puro:** el tracking se inyecta desde
  `tracking/` sin acoplarlo al entrenamiento.
- **Sin fuga también en el artefacto:** `registrar_resultado` reconstruye
  `x_train`/`y_train` reaplicando el preprocesamiento aprendido sobre train
  (vía `resultado.ajustes`), en vez de re-entrenar o duplicar lógica.
- **URI resoluble:** argumento explícito → `MLFLOW_TRACKING_URI` → store local
  `mlruns/` (gitignored). MLflow 3.x requiere `MLFLOW_ALLOW_FILE_STORE=true`
  para el store de archivos; se habilita por defecto en `configurar_tracking`.
- **MLflow 3.x (arquitectura modelo-céntrica):** `log_model` escribe el modelo
  en el repositorio de modelos del store (`models:/m-<uuid>`), no en los
  artefactos de la corrida; `register_model` crea la versión en el Model
  Registry.
- **Métricas prefijadas** (`baseline_val_*`, `xgboost_val_*`, `xgboost_test_*`)
  para evitar colisiones de nombre; params de XGBoost logueados completos vía
  `get_params()`.
- **Modelos lineales, sin versionar (roadmap fase 4):** `registrar_lineales` no
  toca el Model Registry — el champion se elige y registra recién en la fase 6.
  Cada corrida guarda su propio resumen (`resumen_lineal.json`).

Resultado sobre el dataset real (fase 6): experimento
`prediccion_precios_propiedades`, corrida con las métricas de la sección 9.4c
y modelo `modelo_precio_propiedades` en el Model Registry (versión 1, con
firma de entrada/salida para servir el modelo). La fase 4 agrega una corrida
por modelo lineal (lasso y ridge) con sus métricas y resumen propio.

#### 9.4e `explainability/` (implementado en la fase 7)

| Archivo | Contenido |
|---|---|
| `shap_analysis.py` | `ExplicacionSHAP` (dataclass inmutable: `valores` (n, p), `base`, `nombres`; `importancia_global` = media \|SHAP\| por feature, desc), `calcular_shap` (ajusta `shap.TreeExplainer` sobre el modelo ya entrenado y explica una matriz preprocesada), `grafico_resumen` (beeswarm), `grafico_barras` (media \|SHAP\|), `guardar_figuras` (PNG a un directorio, creándolo) |

Decisiones de diseño:
- **Espacio logarítmico:** el target es `log_precio_usd`, por lo que cada valor
  SHAP es la contribución al log precio y `exp(contribución)` es el factor
  multiplicativo en USD sobre el precio.
- **Propiedad aditiva como contrato:** `base + Σ valores ≈ predict(X)`; el
  test unitario `test_propiedad_aditiva_suma_mas_base_igual_prediccion` la
  verifica (`atol=1e-3`).
- **Desacoplado del entrenamiento:** recibe el modelo ya ajustado y la matriz
  ya preprocesada (misma codificación ordinal/imputación), sin tocar
  `models/entrenamiento.py` — no hay fuga ni transformaciones inconsistentes.
- **Figuras retornadas** como `Figure` de matplotlib (no `show()` dentro del
  paquete) para que el llamador decida mostrarlas o persistirlas; el notebook
  05 las guarda en `reports/figures/` (gitignored).

Resultado sobre el dataset real (fase 7): sobre validación la propiedad
aditiva se cumple con error máx. `8.6e-06`; top-3 por media |SHAP|
`superficie_cubierta` (0.30), `barrio_ordinal` (0.14) y `expensas_usd` (0.07);
base `exp(11.97) ≈ USD 158.234` (precio de referencia). Figuras
`shap_resumen.png` y `shap_importancia.png` en `reports/figures/`.

#### 9.4f `evaluacion/` (implementado en la fase 8)

| Archivo | Contenido |
|---|---|
| `analisis.py` | `metricas_detalladas` (RMSE log, RMSE/MAE/MedAE/MAPE en USD y R²), `tabla_residuos` (precio real/predicho en USD, `residuo_log = pred - real`, `residuo_usd`, `error_relativo = expm1(residuo_log)`), `resumen_errores` (sesgo medio en log y %, percentiles del error relativo absoluto), `metricas_por_segmento` (error por grupo de cualquier columna original, NaN aparte, ordenado por `n`), `bias_por_rango_precio` (`pd.qcut` → sesgo medio por banda), gráficos (`grafico_residuos`, `grafico_error_segmento`, `grafico_sesgo_rango`) y `guardar_figuras` (PNG) |

Decisiones de diseño:
- **Target logarítmico en todo el análisis:** los residuos se computan sobre
  `log_precio_usd` (`residuo_log = pred - real`; positivo = sobreestima) y se
  traducen a USD / porcentaje con `exp()` / `expm1()`.
- **Desacoplado del entrenamiento:** recibe targets reales y predichos (en
  log) ya calculados, por lo que se aplica igual sobre val o test y sobre
  cualquier modelo (baseline, XGBoost).
- **Segmentación sin fuga:** `metricas_por_segmento` agrupa por columnas
  originales (sin codificar) y conserva los NaN (`dropna=False`) como
  segmento propio, sin perder filas.
- **Sesgo por rango con `pd.qcut`:** bandas con el mismo número de
  observaciones (duplicados descartados), lo que evita bandas vacías y revela
  patrones tipo "sobreestimar lo barato y subestimar lo caro".
- **Figuras retornadas** como `Figure` (sin `show()` interno), igual que en
  explainability; el notebook 06 las persiste en `reports/figures/`.

Resultado sobre el dataset real (fase 8, test): RMSE log 0.3040, RMSE USD
$87.244, R² 0.7830; sesgo medio +5.02 % (sobreestima levemente); error
relativo **absoluto** mediano 17.73 % (el 2,5 % del notebook 04 es la mediana
del error relativo **firmado** — medidas distintas). Por segmento,
`departamento` (n=176) es el más estable (RMSE log 0.25) y los segmentos con
n < 10 (cochera, oficina, local) los menos confiables. Por rango de precio:
sobreestima las bandas bajas (sesgo +17.2 % por debajo de $85.400) y
subestima las altas (-15.8 % por encima de $235.000). Figuras
`evaluacion_residuos.png`, `evaluacion_error_segmento.png` y
`evaluacion_sesgo_rango.png` en `reports/figures/`.

### 9.5 `scripts/`

| Script | Estado | Responsabilidad |
|---|---|---|
| `scrape.py` | ✔ implementado | Orquestar la adquisición (usa `src/real_estate/ingestion`) |
| `scrape_nuevas.py` | ✔ implementado | Orquestar la adquisición de **nuevas publicaciones** a un dataset separado (`data/raw/propiedades_nuevas.csv`), pensado para corridas programadas: usa `scrapear(..., revision_periodica=True)` para re-escanear segmentos completos; el dedup es interno a ese dataset (no toca el de entrenamiento) |
| `curate.py` | ✔ implementado | Orquestar la curación (usa `src/real_estate/curation`) |
| `features.py` | ✔ implementado | Orquestar el feature engineering (usa `src/real_estate/features`) |
| `train.py` | ✔ implementado | Orquestar el entrenamiento + tracking MLflow (usa `src/real_estate/models` y `src/real_estate/tracking`; CLI `--input`, `--random-state` y `--no-tracking`) |
| `train_lineales.py` | ✔ implementado (roadmap fase 4) | Orquestar los modelos lineales: Lasso y Ridge con escalado (mismas features y split que XGBoost) + una corrida MLflow por modelo sin Model Registry (usa `src/real_estate/models` y `src/real_estate/tracking`; CLI `--input`, `--random-state`, `--alpha-lasso`, `--alpha-ridge` y `--no-tracking`) |
| `exportar_modelo.py` | ✔ implementado (fase 10) | Entrenar sobre el curado y exportar el bundle de serving a `models/modelo_precio_propiedades/` (usa `src/real_estate/serving`; CLI `--input`, `--output`, `--random-state`) |
| `evaluar_nuevas.py` | ✔ implementado (roadmap fase 2) | Predecir el precio de las nuevas publicaciones con el bundle de serving: cura el CSV de nuevas en memoria (`curar_dataset`), carga el modelo (`cargar_bundle`) y guarda `data/processed/propiedades_nuevas_evaluadas.csv` con `precio_predicho_usd` y `fecha_prediccion` (usa `src/real_estate/serving`; CLI `--input`, `--output`, `--modelo`) |
| `clasificar_ofertas.py` | ✔ implementado (roadmap fase 3) | Clasificar cada publicación evaluada como **buena/mala compra** (ratio `precio_predicho_usd / precio_usd` y zona neutra `1 ± std` del lote) y guardar el ranking de oportunidades `reports/ofertas.csv` ordenado por ratio descendente (usa `src/real_estate/serving`; CLI `--input`, `--output`; pensado para correrse después de `evaluar_nuevas.py` en el mismo cron) |
| `evaluate.py` | ✘ pendiente | Evaluación de modelos |
| `explain.py` | ✘ pendiente | Explicabilidad (SHAP) |

**Relación con MakeFile:** `make scrape`, `make curate`, `make features`,
`make train`, `make train-lineales`, `make export-model` y `make serve` ya
existen (aceptan `ARGS="..."`). A medida que existan los demás scripts, se
agregan targets `make evaluate`, `make explain`.

### 9.6 `notebooks/` (01..07 ✔)

| Notebook | Tema | Estado |
|---|---|---|
| `01_eda_estructura_y_calidad.ipynb` | Estructura (2005×32), datos faltantes, cobertura, identificadores y cardinalidad | ✔ ejecutado |
| `02_eda_precio_y_caracteristicas.ipynb` | Distribución del target, outliers, precio vs características, correlaciones | ✔ ejecutado |
| `03_feature_engineering.ipynb` | Construcción de la matriz de features (1.999×16), verificación ordinal, split 80/10/10 | ✔ ejecutado |
| `04_model_analysis.ipynb` | Pipeline train/val/test sin fuga, preprocesamiento ajustado solo en train, baseline vs XGBoost, comparación en val, modelo final en test, error relativo (mediana 2,5 %) e importancia de features | ✔ ejecutado |
| `05_shap_analysis.ipynb` | SHAP sobre validación: valores y base, propiedad aditiva verificada (error máx. 8.6e-06), importancia global, beeswarm/barras, interpretación en USD vía `exp()`, figuras en `reports/figures/` | ✔ ejecutado |
| `06_model_evaluation.ipynb` | Evaluación profunda sobre test: métricas detalladas (baseline vs XGBoost en val, XGBoost en test), tabla de residuos y resumen de errores, histograma de error relativo y real vs. predicho, error por segmento (tipo/barrio/ambientes), sesgo por rango de precio, figuras en `reports/figures/` | ✔ ejecutado |
| `07_deteccion_oportunidades.ipynb` | Roadmap fase 3: clasificación de nuevas publicaciones como buena/mala compra (ratio predicho/publicado, zona neutra `1 ± std`), histograma de ratios con límites y ranking de oportunidades por ratio descendente; usa dataset sintético si `propiedades_nuevas_evaluadas.csv` aún no existe; figura `reports/figures/oportunidades_ratios.png` | ✔ ejecutado |

Nota: los notebooks se ejecutan headless con
`python -m jupyter nbconvert --to notebook --execute --inplace` (backend Agg).

### 9.7 `tests/`

| Ruta | Contenido | Estado |
|---|---|---|
| `tests/unit/test_cleaning.py` | `limpiar_numero` (14 formatos reales de Argenprop), `limpiar_columnas_numericas`, `limpiar_expensas`, `preparar_fecha` | ✔ |
| `tests/unit/test_scraper.py` | `parsear_listing` (tarjeta completa y mínima, swap Capital Federal, fallback de moneda por `idmoneda`), helpers de URL/tipo/ambientes, ciclo CSV (header/append/ids), `construir_url_segmento` (tipo/barrio), progreso (guardar/cargar/corrupto/reanudación), `scrapear` con 202 (cap en página 100 → completo sin reintentar; 202 temprano → reintenta y avanza; bloqueo sostenido → incompleto y reanudable; segmento completo se saltea; reanudación desde última página guardada; cap respetado aunque `reintentos-202` sea alto), `revision_periodica` (reescaneea segmentos completos deduplicando; el default sigue salteándolos; reanuda corridas incompletas) — 42 tests | ✔ |
| `tests/unit/test_transformations.py` | `cargar_tipo_cambio_historico` (CSV válido/ruta inexistente/filas sin venta), `obtener_tipo_cambio` (mock de `requests`: dict/lista/sin venta/error de red/retroceso de día), `construir_tabla_tipo_cambio` (una consulta por fecha; histórico local sin consultar API; fallback a API; sin ruta → solo API), `normalizar_moneda` (USD/ARS/moneda desconocida/columnas faltantes), `normalizar_expensas`, `crear_indicadores_missing` — 22 tests | ✔ |
| `tests/unit/test_features.py` | `seleccionar_columnas`, `crear_target_log` (filtro de inválidos y de artefactos < 1.000 USD), `crear_orden_mediana`, `codificar_ordinal`, imputación, `construir_features` end-to-end y `dividir_train_val_test` (tamaños, disjunción, reproducibilidad) — 22 tests | ✔ |
| `tests/unit/test_models.py` | `ajustar_preprocesamiento` (aprende ordenes e imputador, ignora columnas ausentes), `aplicar_preprocesamiento` (codifica/imputa, categoría solo en val -> `CODIGO_DESCONOCIDO`, no modifica el original), `separar_features_target`, `calcular_metricas`, `entrenar_baseline` (mediana), `entrenar_xgboost` (shape, reproducibilidad, params) y `entrenar_y_evaluar` end-to-end (supera al baseline) — 15 tests | ✔ |
| `tests/unit/test_modelos_lineales.py` | Roadmap fase 4: `crear_pipeline_lineal` (escala antes del modelo, acepta Lasso y Ridge), `entrenar_lasso`/`entrenar_ridge` (shape consistente, predicciones finitas, respetan alpha) y `entrenar_y_evaluar_lineales` end-to-end (supera al baseline, el mejor en val se evalúa en test, ajustes reutilizables) — 9 tests | ✔ |
| `tests/unit/test_tracking.py` | `configurar_tracking` (crea experimento en store local, respeta env `MLFLOW_TRACKING_URI`), `registrar_resultado` (devuelve run_id + versión y cierra la corrida; loguea params, métricas, artefacto `resumen_entrenamiento.json`; modelo con firma en el Model Registry), `registrar_lineales` (una corrida por modelo, loguea params/métricas/artefacto `resumen_lineal.json`, no versiona en el Model Registry), versionado v1/v2 y `finalizar_corrida` — 10 tests | ✔ |
| `tests/unit/test_explainability.py` | `calcular_shap` (forma, base finita, nombres; propiedad aditiva base + Σ ≈ predicción), `importancia_global` (todas las features, no negativa, descendente), `grafico_resumen`/`grafico_barras` (devuelven `Figure` con ejes), `guardar_figuras` (escribe PNG no vacíos) — 5 tests | ✔ |
| `tests/unit/test_evaluacion.py` | `metricas_detalladas` (predicción perfecta → errores 0 y R² 1; consistencia con `calcular_metricas`), `tabla_residuos` (columnas, relaciones internas, exactitud), `resumen_errores` (claves, coherencia con la tabla, sesgo cero), `metricas_por_segmento` (n por grupo, NaN aparte), `bias_por_rango_precio` (bandas balanceadas y ordenadas), gráficos (devuelven `Figure`, PNG no vacíos) — 14 tests | ✔ |
| `tests/unit/test_serving.py` | Fase 10: round-trip `guardar_bundle`/`cargar_bundle` (archivos escritos, `preprocesamiento.json` válido), `ModeloPrediccion` (USD = `exp(log)`, equivale al pipeline de entrenamiento, invariante al reorden de columnas, categoría desconocida → `CODIGO_DESCONOCIDO`, NaN imputado con la mediana del bundle) — 9 tests | ✔ |
| `tests/unit/test_clasificacion.py` | Roadmap fase 3: cálculo del ratio predicho/publicado (incluidos precios publicados inválidos → NaN/sin clasificar), zona neutra `1 ± std` del lote (ddof=1) en las tres categorías, caso degenerado de std (1 ratio → compara contra 1), ratio inválido que no contamina la std, flujo `clasificar_y_exportar` (ranking ordenado por ratio descendente, crea directorio padre, `FileNotFoundError`) — 13 tests | ✔ |
| `tests/integration/test_pipeline.py` | `curar_csv` end-to-end sobre CSV sintético (columnas del scraper), con tipo de cambio mockeado: conversión USD/ARS, indicadores `*_informado`, conversión de tipos textuales, `FileNotFoundError` | ✔ |
| `tests/integration/test_evaluar_nuevas.py` | Roadmap fase 2: `evaluar_nuevas` end-to-end sobre CSV sintético de nuevas (bundle entrenado sobre datos sintéticos en `tmp_path`, tipo de cambio mockeado): columnas de salida, predicciones positivas y finitas, conserva `precio_usd` publicado (USD/ARS), `fecha_prediccion` = hoy, `FileNotFoundError` — 6 tests | ✔ |
| `tests/integration/test_api.py` | Fase 10: `/health` (200, estado/modelo/versión/métricas del bundle), `/predict` (precio razonable y finito, `log` consistente, estable ante reorden del payload, indicadores `*_informado` derivados, imputación de faltantes, categoría desconocida, 422 con campos faltantes / indicador fuera de rango / valor negativo), arranque falla sin bundle — 11 tests | ✔ |

Nota: las funciones de `transformations` que tocan la red se prueban con el
módulo `requests` mockeado; ningún test hace requests reales.

### 9.8 `configs/config.yaml` (pendiente)

Configuración centralizada del proyecto (parámetros de scraper, curación,
entrenamiento, MLflow). Se cargaría vía pydantic-settings.

### 9.9 DVC (versionado de datos)

`dvc.yaml` + `dvc.lock` versionan el pipeline y los datos (raw → curation →
processed). DVC es dependencia dev (`pyproject.toml`).

**Etapas del pipeline:**

| Etapa | Comando | Dependencias | Outputs |
|---|---|---|---|
| `curar` | `python scripts/curate.py` | `scripts/curate.py`, `src/real_estate/curation`, `data/raw/propiedades_argenprop.csv` | `data/processed/propiedades_argenprop_curado.csv` |
| `features` | `python scripts/features.py` | `scripts/features.py`, `src/real_estate/features`, `data/processed/propiedades_argenprop_curado.csv` | `data/processed/propiedades_argenprop_features.csv` |
| `evaluar_nuevas` | `python scripts/evaluar_nuevas.py` | `scripts/evaluar_nuevas.py`, `src/real_estate/serving`, `src/real_estate/curation`, `data/raw/propiedades_nuevas.csv`, `models/modelo_precio_propiedades/` | `data/processed/propiedades_nuevas_evaluadas.csv` |
| `clasificar_ofertas` | `python scripts/clasificar_ofertas.py` | `scripts/clasificar_ofertas.py`, `src/real_estate/serving`, `data/processed/propiedades_nuevas_evaluadas.csv` | `reports/ofertas.csv` |

**Almacenamiento:** el contenido de los datos vive en el cache local
`.dvc/cache` (gitignored) y en el remote por defecto `local` → `dvcstore/`
(gitignored). Los pointer files `data/raw/propiedades_argenprop.csv.dvc` y
`dvc.lock` sí se versionan en git (guardan los hashes md5 de cada archivo).

**Flujo típico:**

```text
dvc repro     # reproduce las etapas (solo si algo cambió)
dvc push      # sube datos al remote (local: dvcstore/)
dvc pull      # baja datos del remote (otra máquina / CI)
dvc status    # compara workspace vs remote
```

**Remote de producción:** para colaborar o CI real conviene cambiar el remote
por defecto a uno en la nube (S3, GCS, DAGsHub o Google Drive), p. ej.:
`dvc remote add -d storage s3://bucket/real-estate-dvc`. El remote local
`dvcstore/` queda como default para que el flujo funcione out-of-the-box.

### 9.10 Serving + API FastAPI (Fase 10)

El modelo se expone como servicio de predicción HTTP. El flujo es:
`make export-model` genera el **bundle de serving** en `models/modelo_precio_propiedades/`
(modelo + preprocesamiento + orden de features + metadata) y `make serve` levanta
la API FastAPI que lo carga al arrancar y predice sobre nuevas propiedades.

**Bundle de serving** (`models/modelo_precio_propiedades/`, gitignored):

| Archivo | Contenido |
|---|---|
| `modelo_xgboost.json` | XGBoost guardado con `save_model` nativo (formato JSON de XGBoost) |
| `preprocesamiento.json` | `Preprocesamiento` aprendido sobre train: ordenes ordinales por categoría (`barrio`, `tipo_propiedad`) + imputador por mediana |
| `features.json` | Orden exacto de las 14 features que espera el modelo |
| `metadata.json` | Métricas en test (RMSE log, R²), tamaños de split y fecha de exportación |

> **Por qué bundle propio y no el Model Registry de MLflow:** MLflow loguea solo
> el XGBoost (sin firma de preprocesamiento); el `Preprocesamiento` (codificación
> ordinal + imputación) no se persiste por MLflow, así que el serving usa un
> bundle a medida generado por `guardar_bundle`.

**Componentes:**

| Ruta | Propósito |
|---|---|
| `src/real_estate/serving/persistencia.py` | `guardar_bundle` / `cargar_bundle` (round-trip del bundle a disco) |
| `src/real_estate/serving/modelo.py` | `ModeloPrediccion` (dataclass): `_construir_matriz` replica el pipeline de entrenamiento (`seleccionar_columnas` → `aplicar_preprocesamiento` → reorden por `columnas_features`), `predecir_log` / `predecir_usd` |
| `scripts/exportar_modelo.py` | Entrena sobre `data/processed/propiedades_argenprop_curado.csv`, arma el bundle y escribe `resumen_bundle.json` |
| `src/real_estate/api/config.py` | `ConfiguracionServicio` (pydantic-settings; `modelo_dir` desde env `MODELO_DIR`, default `models/modelo_precio_propiedades`) |
| `src/real_estate/api/schemas.py` | `PropiedadEntrada` (tipo, barrio, 6 numéricas opcionales con `ge=0`, 6 indicadores `_informado` opcionales 0/1; `None` en un indicador = derivar del valor), `PrediccionSalida` |
| `src/real_estate/api/app.py` | `crear_app(config)` con lifespan que carga el bundle (falla con `RuntimeError` si no existe); `GET /health` (estado, modelo, versión, métricas), `POST /predict` (devuelve `precio_usd` y `log_precio_usd`); módulo `app` para uvicorn |

**Contrato de entrada del modelo:** 14 features — 6 numéricas imputables
(`superficie_cubierta`, `ambientes`, `dormitorios`, `banos`, `antiguedad`,
`expensas_usd`), 6 indicadores `*_informado` y 2 ordinales (`barrio_ordinal`,
`tipo_propiedad_ordinal`; categoría desconocida → `CODIGO_DESCONOCIDO = -1`).

**Uso:**

```bash
make export-model    # genera models/modelo_precio_propiedades/
make serve           # uvicorn real_estate.api.app:app --reload
```

Resultado sobre el dataset real (fase 10): split 1.599 train / 200 val / 200
test, RMSE log 0.304, R² 0.783 en test; `/predict` devuelve precios en USD.

### 9.11 Docker (implementado, fase 11)

Conteneriza el servicio de predicción FastAPI:

- **`Dockerfile`** multi-etapa:
  1. `build` (`python:3.12-slim`): crea un venv en `/opt/venv` con las
     dependencias de serving (`requirements-api.txt`) e instala el paquete
     `real_estate` con `--no-deps` (las deps reales vienen del requirements).
  2. `runtime` (`python:3.12-slim`): copia solo el venv, expone el puerto 8000,
     define `MODELO_DIR` y un `HEALTHCHECK` sobre `/health` (urllib, no hay
     curl en la imagen slim). Arranca con `uvicorn real_estate.api.app:app`.
- **`requirements-api.txt`**: solo el runtime de serving (fastapi, uvicorn,
  pydantic-settings, python-dotenv, xgboost, numpy, pandas, scikit-learn) — sin
  mlflow/shap/scraping para mantener la imagen liviana.
- **`docker-compose.yml`**: monta el bundle de serving
  (`models/modelo_precio_propiedades/`) como volumen de solo lectura — es un
  artefacto entrenado gitignoreado, no parte de la imagen.

**Uso:** `make export-model` (una vez, o tras reentrenar) → `make docker-build`
→ `make docker-up` → `make docker-logs`. La API queda en `http://localhost:8000`
(`/health` + `/predict`).

### 9.12 CI/CD

`ci.yml` (`.github/workflows/ci.yml`): ejecuta Ruff (check + formato), Mypy y
Pytest con cobertura en GitHub Actions. Un job de calidad (Python 3.12) y un
job de tests con matriz Python 3.11 / 3.12, instalando `pip install -e ".[dev]"`.

`dvc.yml` (`.github/workflows/dvc.yml`): valida el pipeline de datos. Lista las
etapas definidas (`dvc stage list`), verifica el estado de deps/outs contra el
lock (`dvc status`) e intenta restaurar los datos con `dvc pull`. El pull es
best-effort: si falla (el remote por defecto es local, `dvcstore/`, que no
existe en CI), el job igualmente termina en éxito con un aviso, porque la
validación real es de la definición del pipeline. Se dispara en push/PR a
`main` y manualmente (`workflow_dispatch`).

### 9.13 Datos

| Ruta | Contenido | Estado |
|---|---|---|
| `data/raw/propiedades_argenprop.csv` | Dataset crudo, 2.005 registros, 20 columnas (el cap de 100 páginas limitaba la campaña; con segmentación se apunta a ≥ 10.000) | ✔ |
| `data/interim/` | Datos intermedios (p. ej., entre curación y features) | 🏗 vacía |
| `data/processed/propiedades_argenprop_curado.csv` | Dataset curado, 2.005 filas, 32 columnas (listo para EDA/features) | ✔ |
| `data/processed/propiedades_argenprop_features.csv` | Matriz de features, 1.999 filas × 16 columnas, 0 faltantes (lista para modelar) | ✔ |
| `data/external/tipo_cambio_blue.csv` | Histórico del dólar blue (compra/venta por día hábil), 5.702 fechas (2011-01-03 → 2026-08-15), trackeado por DVC. Fuente primaria de `normalizar_moneda`; se descarga con `scripts/download_tipo_cambio.py` | ✔ |

### 9.14 Otros directorios

| Ruta | Estado | Nota |
|---|---|---|
| `models/` | ✔ en uso (fase 10) | Bundle de serving `modelo_precio_propiedades/` (gitignored; se regenera con `make export-model`) |
| `reports/figures/` | ✔ en uso | Figuras SHAP del notebook 05 (gitignored; se regeneran con el notebook) |
| `reports/metrics/` | 🏗 vacía | Métricas de evaluación |
| `mlruns/` | ✔ en uso | Store local de MLflow (experimentos, corridas y repositorio de modelos; gitignored) |
| `docs/architecture.md` | ✔ | Este documento (mapa vivo) |
| `docs/roadmap.md` | ✔ | Roadmap de predicción de precio + detección de oportunidades de compra (buena/mala compra, score relativo, modelos lineales, tuning XGBoost, MLflow) |

---

## 10. Estado actual — matriz OK / pendiente

### ✅ Hecho

- [x] Idea y problema definidos
- [x] Fuente de datos real seleccionada (Argenprop)
- [x] Scraper desarrollado y refactorizado (`src/real_estate/ingestion/scraper.py` + `scripts/scrape.py`), **v3**: segmentación por barrio/tipo para superar el cap de 100 páginas, manejo del 202 con backoff y progreso reanudable
- [x] Scraper probado con datos reales
- [x] Dataset raw generado (`data/raw/propiedades_argenprop.csv`)
- [x] Data Curation implementada (`src/real_estate/curation/`: cleaning, transformations, validation, pipeline + `scripts/curate.py`)
- [x] Dataset curado generado (`data/processed/propiedades_argenprop_curado.csv`)
- [x] Arquitectura general definida
- [x] Estructura de carpetas creada
- [x] `pyproject.toml` creado
- [x] Makefile creado (como `MakeFile`)
- [x] `docs/architecture.md` creado
- [x] `.gitignore`, `.dockerignore`, `.editorconfig`, `.pre-commit-config.yaml`, `.env.example` creados

### ❌ Pendiente

- [x] Tests del código del proyecto (`tests/`) — unit (cleaning, scraper, transformations, features) + integración (pipeline)
- [x] DVC (`dvc.yaml`, `dvc.lock`, dependencia dev, remote local `dvcstore/`)
- [x] EDA estructurado (notebooks 01 y 02, ejecutados headless)
- [x] Feature Engineering (`src/real_estate/features`, notebook 03)
- [x] Pipeline Train / Validation / Test
- [x] Baseline
- [x] XGBoost training pipeline
- [x] Modelos lineales Lasso/Ridge (roadmap fase 4) con escalado y tracking por modelo
- [x] MLflow tracking (`src/real_estate/tracking`)
- [x] SHAP analysis (`src/real_estate/explainability`, notebook 05)
- [x] Model evaluation (`src/real_estate/evaluacion`, notebook 06)
- [x] GitHub Actions (`ci.yml` + `dvc.yml`)
- [x] Serving bundle (`src/real_estate/serving` + `scripts/exportar_modelo.py`, bundle en `models/modelo_precio_propiedades/`)
- [x] API de predicción FastAPI (`src/real_estate/api`, `/health` + `/predict`, `make serve`)
- [x] Dockerfile multi-stage + `requirements-api.txt` (fase 11)
- [x] docker-compose (fase 11)

---

## 11. Flujo de calidad de código

```text
Developer
    ↓
Code
    ↓
pre-commit
    ↓
Ruff
    ↓
Mypy
    ↓
Git commit
    ↓
GitHub
    ↓
GitHub Actions
    ↓
CI
    ↓
Pytest / Ruff / Mypy
    ↓
Docker build
```

**Gate local:** `make check` = `ruff format --check` + `ruff check` + `mypy` + `pytest`.

---

## 12. Mapa de relaciones clave

### Flujo de datos (componentes de código)

```text
scripts/scrape.py ──→ src/real_estate/ingestion/scraper.py
        ↓
data/raw/propiedades_argenprop.csv                  ◄─ DVC (pointer .dvc + cache)
        ↓
scripts/curate.py ──→ src/real_estate/curation/pipeline.py
                      ├── cleaning.py
                      ├── transformations.py
                      └── validation.py
        ↓
data/processed/propiedades_argenprop_curado.csv     ◄─ DVC (output etapa curar)
        ↓
scripts/features.py ──→ src/real_estate/features/
                        ├── transformations.py
                        └── pipeline.py
        ↓
data/processed/propiedades_argenprop_features.csv   ◄─ DVC (output etapa features)
                                                     (1.999 × 16, sin faltantes)
        ↓
scripts/train.py ──→ src/real_estate/models/entrenamiento.py  (Baseline → XGBoost)
        ↓
src/real_estate/tracking (MLflow: params / métricas / artefactos + Model Registry)
        ↓
scripts/train_lineales.py ──→ src/real_estate/models/modelos_lineales.py  (Lasso/Ridge, fase 4)
                              └──→ src/real_estate/tracking (una corrida por modelo, sin registry)
        ↓
modelo_precio_propiedades (versión con firma; champion elegido en fase 6)
        ↓
notebooks/05_shap_analysis.ipynb ──→ src/real_estate/explainability/shap_analysis.py
                                      (TreeExplainer sobre val → valores/base → figuras)
        ↓
notebooks/06_model_evaluation.ipynb ──→ src/real_estate/evaluacion/analisis.py
                                         (residuos / error por segmento / sesgo por rango, sobre test)
        ↓
reports/figures/ (shap_*.png, evaluacion_*.png)
        ↓
scripts/exportar_modelo.py ──→ src/real_estate/serving/  (bundle de serving, fase 10)
        ↓
models/modelo_precio_propiedades/ (modelo + preprocesamiento + features + metadata)
        ↓
src/real_estate/api/app.py ──→ src/real_estate/api/schemas.py
                               (FastAPI: /health + /predict, fase 10)
        ↓
make serve (uvicorn real_estate.api.app:app)

Flujo del cron (roadmap, dataset separado de nuevas publicaciones):

```text
scripts/scrape_nuevas.py ──→ src/real_estate/ingestion/scraper.py (revision_periodica)
        ↓
data/raw/propiedades_nuevas.csv
        ↓
scripts/evaluar_nuevas.py ──→ src/real_estate/serving + curation (cura en memoria)
        ↓
data/processed/propiedades_nuevas_evaluadas.csv (precio_predicho_usd, fecha_prediccion)
        ↓
scripts/clasificar_ofertas.py ──→ src/real_estate/serving/clasificacion.py (ratio + zona 1±std)
        ↓
reports/ofertas.csv (ranking buena/mala compra) + notebooks/07 (figura oportunidades_ratios.png)
```

### Dependencias de software (ejemplo)

```text
train.py
        ↓
XGBoost
        ↓
MLflow
```

```text
scrape.py ──→ src/real_estate/ingestion ──→ requests / BeautifulSoup / lxml
curate.py ──→ src/real_estate/curation ──→ pandas / numpy / pydantic-settings
features.py ──→ src/real_estate/features ──→ scikit-learn / numpy / pandas
train.py  ──→ src/real_estate/models ──→ scikit-learn / xgboost
             └──→ src/real_estate/tracking ──→ mlflow
train_lineales.py ──→ src/real_estate/models ──→ scikit-learn
             └──→ src/real_estate/tracking ──→ mlflow
exportar_modelo.py ──→ src/real_estate/serving ──→ xgboost / pandas / numpy
evaluar_nuevas.py ──→ src/real_estate/serving ──→ xgboost / pandas / numpy
                 └──→ src/real_estate/curation ──→ pandas / numpy
clasificar_ofertas.py ──→ src/real_estate/serving ──→ pandas / numpy
api/app.py ──→ src/real_estate/api ──→ fastapi / uvicorn / pydantic-settings
             └──→ src/real_estate/serving ──→ xgboost / pandas / numpy
explain.py ──→ src/real_estate/explainability ──→ shap
evaluate.py ──→ src/real_estate/evaluacion ──→ scikit-learn / matplotlib
```

---

## 13. Diferencias entre la estructura conceptual y el estado real

> Registro explícito para no perder de vista la transición.

| Concepto | Realidad | Acción |
|---|---|---|
| `Makefile` | El archivo se llama `MakeFile` | Renombrar o aceptar el nombre actual |
| `scripts/scrape.py` | ✔ Migrado: `src/real_estate/ingestion/scraper.py` + `scripts/scrape.py`. La raíz quedó limpia. **v3:** segmentación por barrio/tipo (54 barrios), manejo del cap 202, backoff y progreso JSON reanudable | ✔ |
| `scripts/curate.py` | ✔ Implementado: `src/real_estate/curation/` (cleaning, transformations, validation, pipeline) + `scripts/curate.py` | ✔ |
| `scripts/` | `scrape.py`, `scrape_nuevas.py`, `curate.py`, `features.py`, `train.py`, `train_lineales.py`, `exportar_modelo.py`, `evaluar_nuevas.py`, `clasificar_ofertas.py` y `download_tipo_cambio.py` existen; faltan `evaluate.py`, `explain.py` | Parcial |
| `scripts/scrape_nuevas.py` | ✔ Implementado (fase 12 / roadmap): dataset separado de nuevas publicaciones con `revision_periodica` (re-escaneea segmentos completos, dedup interno al dataset de nuevas) | ✔ |
| `scripts/evaluar_nuevas.py` | ✔ Implementado (roadmap fase 2): predicción de precio sobre las nuevas publicaciones (`src/real_estate/serving/evaluar.py`): cura en memoria + carga del bundle + `precio_predicho_usd` y `fecha_prediccion` en `data/processed/propiedades_nuevas_evaluadas.csv` | ✔ |
| `scripts/clasificar_ofertas.py` | ✔ Implementado (roadmap fase 3): clasificación buena/mala compra (`src/real_estate/serving/clasificacion.py`): ratio `precio_predicho_usd / precio_usd` con zona neutra `1 ± std` del lote + ranking por ratio descendente en `reports/ofertas.csv` | ✔ |
| `scripts/train_lineales.py` | ✔ Implementado (roadmap fase 4): modelos lineales Lasso/Ridge con `StandardScaler` (`src/real_estate/models/modelos_lineales.py`): mismo preprocesamiento y features que XGBoost, comparación en val y test al mejor; una corrida MLflow por modelo sin Model Registry (`registrar_lineales`) | ✔ |
| `docs/roadmap.md` | ✔ Creado: roadmap de predicción de precio + detección de oportunidades (buena/mala compra con score relativo), fases 1-6 con decisiones tomadas. Fases 1 (scrape de nuevas) ✔, 2 (predicción) ✔, 3 (clasificación buena/mala compra) ✔ y 4 (modelos lineales Lasso/Ridge) ✔ | ✔ |
| `configs/config.yaml` | Carpeta creada, sin archivo | Pendiente |
| `dvc.yaml` / `dvc.lock` | ✔ Implementado: etapas `curar` y `features` con hashes md5. Remote por defecto `local` → `dvcstore/` | ✔ |
| `Dockerfile` / `docker-compose.yml` | ✔ Implementado (fase 11): `Dockerfile` multi-stage + `requirements-api.txt` + `docker-compose.yml` con bundle montado como volumen de solo lectura | ✔ |
| `.github/workflows/ci.yml` | ✔ Implementado: lint (Ruff), type check (Mypy) y tests (Pytest + cobertura) en Python 3.11/3.12 | ✔ |
| `.github/workflows/dvc.yml` | ✔ Implementado: valida etapas (`stage list`), estado (`status`) y `pull` best-effort | ✔ |
| `data/raw/` | Contiene `propiedades_argenprop.csv` (2.005 registros) | ✔ |
| `data/processed/` | Contiene `propiedades_argenprop_curado.csv` (32 columnas) y `propiedades_argenprop_features.csv` (16 columnas) | ✔ |
| `data/external/` | Contiene `tipo_cambio_blue.csv` (5.702 fechas de dólar blue, trackeado con DVC; fuente primaria de `normalizar_moneda`, fallback a la API) | ✔ |
| `tests/` | ✔ Implementado: unit (cleaning, scraper, transformations, features, models, modelos_lineales, tracking, explainability, evaluacion, serving, clasificacion) + integration (pipeline, evaluar_nuevas, API FastAPI) — 214 tests | ✔ |
| `src/real_estate/models` | ✔ Implementado en la fase 5: `entrenamiento.py` (baseline, XGBoost, evaluación sin fuga); roadmap fase 4: `modelos_lineales.py` (Lasso/Ridge con escalado dentro del pipeline) | ✔ |
| `src/real_estate/explainability` | ✔ Implementado en la fase 7: `shap_analysis.py` (valores SHAP, base, importancia global, figuras) | ✔ |
| `src/real_estate/evaluacion` | ✔ Implementado en la fase 8: `analisis.py` (métricas detalladas, residuos, error por segmento, sesgo por rango, figuras) | ✔ |
| `src/real_estate/serving` | ✔ Implementado en la fase 10: `modelo.py` (`ModeloPrediccion`) + `persistencia.py` (`guardar_bundle`/`cargar_bundle`); roadmap fase 3: `clasificacion.py` (ratio + zona `1 ± std` + ranking) | ✔ |
| `src/real_estate/api` | ✔ Implementado en la fase 10: `app.py` (FastAPI `/health` + `/predict`), `schemas.py`, `config.py` | ✔ |
| `notebooks/` | ✔ 01..07 ejecutados (estructura/calidad, precio/características, feature engineering, model analysis, shap analysis, model evaluation, detección de oportunidades) | ✔ |
| `reports/figures/` | ✔ En uso: figuras SHAP (05) y de evaluación (06) (gitignored) | ✔ |
| `models/` | ✔ Bundle de serving de la fase 10 (`modelo_precio_propiedades/`; gitignored, se regenera con `make export-model`) | ✔ |
| `mlruns/` | ✔ En uso: store local de MLflow (experimentos, corridas, repositorio de modelos) — gitignored | ✔ |
| Repositorio Git | ✔ Inicializado en `main`, remoto `origin` apuntando a GitHub (matiasbelsito7/real-estate-price-prediction), commit inicial pusheado | ✔ |
| `README.md` | ✔ Actualizado: documenta scrape y curate vía `scripts/`; ya no menciona `scraper_argenprop2.py` | ✔ |

> **Nota de entorno (resuelta):** el ambiente local tiene Python 3.14.7 +
> numpy 2.5.2, cuyos stubs usan `type` statements (PEP 695, solo Python ≥ 3.12).
> `pyproject.toml` fija `[tool.mypy] python_version = "3.12"` (dentro del rango
> soportado `>=3.11,<3.14`), lo que hace que `mypy` plano, el hook de pre-commit
> y el CI funcionen de forma consistente. El hook de mypy en pre-commit
> además declara `pandas-stubs`, `types-requests` y `pytest` como dependencias
> extra (pytest es necesario para tipar los decoradores `@pytest.fixture` y
> `@pytest.mark.parametrize` en los tests: el entorno aislado del hook no
> incluye las dependencias dev del proyecto).

---

## 14. Convenciones para mantener este documento

1. **Cada vez que se cree o modifique un archivo**, actualizar la sección 9
   (Registro de componentes) con: ruta, propósito, responsabilidades,
   dependencias, archivos que lo usan, archivos que usa, outputs y relaciones.
2. **Cada pieza nueva** debe responder antes de implementarse: ¿qué
   responsabilidad tiene? ¿en qué capa vive? ¿de qué archivos depende? ¿qué
   dependerá de ella? ¿cómo afecta al flujo de datos? ¿cómo se testea? ¿requiere
   cambios en `pyproject.toml`, `MakeFile` u otra configuración?
3. **No agregar herramientas sin justificación** ni arquitectura innecesaria sin
   discutir primero su razón de ser.
4. Actualizar las matrices de estado (sección 10) y las diferencias
   concepto/realidad (sección 13) cuando corresponda.
