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
- **Containerization** (Docker / Docker Compose)
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
                                      │     Docker    │
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
    └──────────────→ Docker / Deployment
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
| Containerization | Docker, Docker Compose | Contenedores | ✘ pendiente |
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
├── Dockerfile                     ✘
├── docker-compose.yml             ✘
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
│   └── external/                            🏗 vacía
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
│       ├── models/                ✔ entrenamiento.py (baseline, XGBoost, sin fuga)
│       ├── explainability/        ✔ shap_analysis.py (SHAP: valores, base,
│       │                             figuras, guardado)
│       ├── evaluacion/            ✔ analisis.py (métricas detalladas, residuos,
│       │                             error por segmento, sesgo por rango)
│       ├── tracking/              ✔ experimentos.py (MLflow: params, métricas,
│       │                             artefactos, Model Registry)
│       └── utils/                 🏗 vacío
│
├── scripts/
│   ├── scrape.py                   ✔ (entry point de adquisición)
│   ├── curate.py                   ✔ (entry point de curación)
│   ├── features.py                 ✔ (entry point de feature engineering)
│   └── train.py                    ✔ (entry point de modelado + tracking MLflow)
│   (se planifican evaluate/explain)
│
├── tests/
│   ├── unit/                       ✔ (cleaning, scraper, transformations, features, models, tracking, explainability, evaluacion)
│   └── integration/                ✔ (pipeline de curación)
│
├── models/                         🏗 vacía
│
├── reports/
│   ├── figures/                    ✔ en uso (figuras SHAP fase 7 y de evaluación fase 8; gitignored)
│   └── metrics/                    🏗 vacía
│
├── mlruns/                         ✔ (store local de MLflow, gitignored)
│
├── docs/
│   └── architecture.md             ✔ (este documento)
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
| **Dependencias prod** | numpy, pandas, requests, beautifulsoup4, lxml, scikit-learn, xgboost, mlflow, shap, pydantic-settings, python-dotenv |
| **Dependencias dev** | pytest, pytest-cov, ruff, mypy, pandas-stubs, pre-commit, jupyter, ipykernel, matplotlib |
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
| **Targets actuales** | `install`, `install-dev`, `scrape`, `curate`, `features`, `train`, `dvc-repro`, `dvc-push`, `dvc-pull`, `dvc-status`, `format`, `lint`, `typecheck`, `test`, `coverage`, `check`, `clean` (además de `help`) |
| **Targets futuros** | `evaluate`, `explain`, `docker-...` — solo cuando los componentes existan realmente |
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
| **Variables** | `APP_ENV`, `LOG_LEVEL`, `DATA_DIR`, `MLFLOW_TRACKING_URI`, `SCRAPER_REQUEST_TIMEOUT`, `SCRAPER_DELAY_SECONDS` |
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
| **Responsabilidades** | Paginar el listado, parsear tarjetas (Requests + BeautifulSoup + lxml), extraer los 20 campos, guardar CSV incrementalmente |
| **Estado** | ✔ Funcional, probado con datos reales |
| **Dependencias** | `requests`, `beautifulsoup4`, `lxml` |
| **Constantes** | `BASE_URL`, `HEADERS`, `COLUMNS` (20), `TIPOS_PROPIEDAD` (18), `ICONO_A_COLUMNA`, `RE_AMBIENTES_EN_URL` |
| **Funciones** | `construir_url_pagina`, `texto_o_none`, `detectar_tipo_propiedad`, `extraer_ambientes_de_url`, `extraer_features_de_tarjeta`, `parsear_listing`, `cargar_ids_existentes`, `asegurar_encabezado`, `guardar_filas`, `scrapear` |
| **Comportamiento** | Resumen por `id`: si se corta, al volver a correr no re-baja avisos ya presentes. Precio/moneda desde atributos `data` del link; ambientes desde la URL del aviso; alerta si 3 páginas seguidas no traen features (cambio de estructura del sitio) |
| **Usado por** | `scripts/scrape.py` |
| **Outputs** | `data/raw/propiedades_argenprop.csv` (o `--output`) |
| **Relación** | Alimenta la etapa de Data Curation. `banos`/`cocheras` quedan mayormente vacíos por limitación del listado (no es un bug) |

#### `scripts/scrape.py`

| Atributo | Valor |
|---|---|
| **Ruta** | `scripts/scrape.py` |
| **Propósito** | Entry point CLI de adquisición de datos |
| **Responsabilidades** | Parser de argumentos, bootstrap de `sys.path` para importar `real_estate` sin instalar el paquete, manejo de `KeyboardInterrupt` |
| **CLI** | `--output`, `--max-paginas`, `--pagina-inicio`, `--delay-min`, `--delay-max`, `--html-debug` |
| **Usa** | `real_estate.ingestion.scraper.scrapear` |
| **Usado por** | `make scrape` (MakeFile) |
| **Outputs** | CSV en `data/raw/` |

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
| **Responsabilidades** | `obtener_tipo_cambio` (consulta histórica por fecha, retrocede días hábiles, usa cotización "venta"), `construir_tabla_tipo_cambio` (una request por fecha única), `normalizar_moneda` (crea `tipo_cambio_ars_usd` y `precio_usd`; USD se copia, ARS se divide), `normalizar_expensas` (`expensas_usd`), `crear_indicadores_missing` (`{columna}_informado` int8) |
| **Estado** | ✔ Implementado |
| **Dependencias** | `pandas`, `requests` |
| **FX API** | `https://api.argentinadatos.com/v1/cotizaciones/dolares/{market}/{date}`, `FX_MARKET = "blue"` (opciones: oficial, blue, bolsa, contadoconliqui, mayorista, etc.) |
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

Decisiones de diseño:
- **Sin fuga de información:** no se reutiliza `construir_features` (codifica e imputa sobre todo el dataset). Se ajustan sobre train los ordenes ordinales (`crear_orden_mediana`) y la imputación por mediana (`crear_imputador`), y se reaplican a val/test con `aplicar_preprocesamiento`. Categorías no vistas en train -> `CODIGO_DESCONOCIDO (-1)`.
- **Split 80/10/10 reproducible** vía `dividir_train_val_test` (`random_state=42`).
- **Baseline (mediana)** como referencia mínima con `DummyRegressor(strategy="median")`.
- **XGBoost** con parámetros por defecto en `PARAMS_XGBOOST_DEFAULT` (300 árboles, depth 4, lr 0.05, regularización) y `random_state` para reproducibilidad.
- **Métricas sobre el target logarítmico:** RMSE log (≈ error relativo), RMSE USD (deshaciendo el log) y R².

Resultados sobre el dataset real (fase 5):

| Modelo | RMSE log (val) | RMSE USD (val) | R² (val) | RMSE log (test) | R² (test) |
|---|---|---|---|---|---|
| baseline (mediana) | 0.6423 | $185.135 | -0.0025 | — | — |
| XGBoost | 0.2718 | $112.963 | 0.8205 | 0.3040 | 0.7830 |

XGBoost reduce el RMSE log en ~58 % respecto del baseline; error relativo
mediano en test: 2,5 %.

#### 9.4d `tracking/` (implementado en la fase 6)

| Archivo | Contenido |
|---|---|
| `experimentos.py` | `configurar_tracking` (URI + experimento, creándolo si no existe), `registrar_resultado` (abre la corrida: loguea params, métricas, artefacto JSON `resumen_entrenamiento.json`, modelo XGBoost con firma vía `infer_signature` y lo versiona en el Model Registry; devuelve `(run_id, version)`), `finalizar_corrida` |

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

Resultado sobre el dataset real (fase 6): experimento
`prediccion_precios_propiedades`, corrida con las métricas de la sección 9.4c
y modelo `modelo_precio_propiedades` en el Model Registry (versión 1, con
firma de entrada/salida para servir el modelo).

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
| `curate.py` | ✔ implementado | Orquestar la curación (usa `src/real_estate/curation`) |
| `features.py` | ✔ implementado | Orquestar el feature engineering (usa `src/real_estate/features`) |
| `train.py` | ✔ implementado | Orquestar el entrenamiento + tracking MLflow (usa `src/real_estate/models` y `src/real_estate/tracking`; CLI `--input`, `--random-state` y `--no-tracking`) |
| `evaluate.py` | ✘ pendiente | Evaluación de modelos |
| `explain.py` | ✘ pendiente | Explicabilidad (SHAP) |

**Relación con MakeFile:** `make scrape`, `make curate`, `make features` y
`make train` ya existen (aceptan `ARGS="..."`). A medida que existan los
demás scripts, se agregan targets `make evaluate`, `make explain`.

### 9.6 `notebooks/` (01..06 ✔)

| Notebook | Tema | Estado |
|---|---|---|
| `01_eda_estructura_y_calidad.ipynb` | Estructura (2005×32), datos faltantes, cobertura, identificadores y cardinalidad | ✔ ejecutado |
| `02_eda_precio_y_caracteristicas.ipynb` | Distribución del target, outliers, precio vs características, correlaciones | ✔ ejecutado |
| `03_feature_engineering.ipynb` | Construcción de la matriz de features (1.999×16), verificación ordinal, split 80/10/10 | ✔ ejecutado |
| `04_model_analysis.ipynb` | Pipeline train/val/test sin fuga, preprocesamiento ajustado solo en train, baseline vs XGBoost, comparación en val, modelo final en test, error relativo (mediana 2,5 %) e importancia de features | ✔ ejecutado |
| `05_shap_analysis.ipynb` | SHAP sobre validación: valores y base, propiedad aditiva verificada (error máx. 8.6e-06), importancia global, beeswarm/barras, interpretación en USD vía `exp()`, figuras en `reports/figures/` | ✔ ejecutado |
| `06_model_evaluation.ipynb` | Evaluación profunda sobre test: métricas detalladas (baseline vs XGBoost en val, XGBoost en test), tabla de residuos y resumen de errores, histograma de error relativo y real vs. predicho, error por segmento (tipo/barrio/ambientes), sesgo por rango de precio, figuras en `reports/figures/` | ✔ ejecutado |

Nota: los notebooks se ejecutan headless con
`python -m jupyter nbconvert --to notebook --execute --inplace` (backend Agg).

### 9.7 `tests/`

| Ruta | Contenido | Estado |
|---|---|---|
| `tests/unit/test_cleaning.py` | `limpiar_numero` (14 formatos reales de Argenprop), `limpiar_columnas_numericas`, `limpiar_expensas`, `preparar_fecha` | ✔ |
| `tests/unit/test_scraper.py` | `parsear_listing` (tarjeta completa y mínima, swap Capital Federal, fallback de moneda por `idmoneda`), helpers de URL/tipo/ambientes, ciclo CSV (header/append/ids) | ✔ |
| `tests/unit/test_transformations.py` | `obtener_tipo_cambio` (mock de `requests`: dict/lista/sin venta/error de red/retroceso de día), `construir_tabla_tipo_cambio`, `normalizar_moneda` (USD/ARS/moneda desconocida/columnas faltantes), `normalizar_expensas`, `crear_indicadores_missing` | ✔ |
| `tests/unit/test_features.py` | `seleccionar_columnas`, `crear_target_log` (filtro de inválidos y de artefactos < 1.000 USD), `crear_orden_mediana`, `codificar_ordinal`, imputación, `construir_features` end-to-end y `dividir_train_val_test` (tamaños, disjunción, reproducibilidad) — 22 tests | ✔ |
| `tests/unit/test_models.py` | `ajustar_preprocesamiento` (aprende ordenes e imputador, ignora columnas ausentes), `aplicar_preprocesamiento` (codifica/imputa, categoría solo en val -> `CODIGO_DESCONOCIDO`, no modifica el original), `separar_features_target`, `calcular_metricas`, `entrenar_baseline` (mediana), `entrenar_xgboost` (shape, reproducibilidad, params) y `entrenar_y_evaluar` end-to-end (supera al baseline) — 15 tests | ✔ |
| `tests/unit/test_tracking.py` | `configurar_tracking` (crea experimento en store local, respeta env `MLFLOW_TRACKING_URI`), `registrar_resultado` (devuelve run_id + versión y cierra la corrida; loguea params, métricas, artefacto `resumen_entrenamiento.json`; modelo con firma en el Model Registry), versionado v1/v2 y `finalizar_corrida` — 7 tests | ✔ |
| `tests/unit/test_explainability.py` | `calcular_shap` (forma, base finita, nombres; propiedad aditiva base + Σ ≈ predicción), `importancia_global` (todas las features, no negativa, descendente), `grafico_resumen`/`grafico_barras` (devuelven `Figure` con ejes), `guardar_figuras` (escribe PNG no vacíos) — 5 tests | ✔ |
| `tests/unit/test_evaluacion.py` | `metricas_detalladas` (predicción perfecta → errores 0 y R² 1; consistencia con `calcular_metricas`), `tabla_residuos` (columnas, relaciones internas, exactitud), `resumen_errores` (claves, coherencia con la tabla, sesgo cero), `metricas_por_segmento` (n por grupo, NaN aparte), `bias_por_rango_precio` (bandas balanceadas y ordenadas), gráficos (devuelven `Figure`, PNG no vacíos) — 14 tests | ✔ |
| `tests/integration/test_pipeline.py` | `curar_csv` end-to-end sobre CSV sintético (columnas del scraper), con tipo de cambio mockeado: conversión USD/ARS, indicadores `*_informado`, conversión de tipos textuales, `FileNotFoundError` | ✔ |

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

### 9.10 Docker (pendiente)

`Dockerfile` + `docker-compose.yml` para contenerizar el proyecto / el servicio
de predicción (deployment).

### 9.11 CI/CD

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

### 9.12 Datos

| Ruta | Contenido | Estado |
|---|---|---|
| `data/raw/propiedades_argenprop.csv` | Dataset crudo, 2.005 registros, 20 columnas | ✔ |
| `data/interim/` | Datos intermedios (p. ej., entre curación y features) | 🏗 vacía |
| `data/processed/propiedades_argenprop_curado.csv` | Dataset curado, 2.005 filas, 32 columnas (listo para EDA/features) | ✔ |
| `data/processed/propiedades_argenprop_features.csv` | Matriz de features, 1.999 filas × 16 columnas, 0 faltantes (lista para modelar) | ✔ |
| `data/external/` | Datos externos (p. ej., tipo de cambio) | 🏗 vacía |

### 9.13 Otros directorios

| Ruta | Estado | Nota |
|---|---|---|
| `models/` | 🏗 vacía | Artefactos de modelos (futuro) |
| `reports/figures/` | ✔ en uso | Figuras SHAP del notebook 05 (gitignored; se regeneran con el notebook) |
| `reports/metrics/` | 🏗 vacía | Métricas de evaluación |
| `mlruns/` | ✔ en uso | Store local de MLflow (experimentos, corridas y repositorio de modelos; gitignored) |
| `docs/architecture.md` | ✔ | Este documento (mapa vivo) |

---

## 10. Estado actual — matriz OK / pendiente

### ✅ Hecho

- [x] Idea y problema definidos
- [x] Fuente de datos real seleccionada (Argenprop)
- [x] Scraper desarrollado y refactorizado (`src/real_estate/ingestion/scraper.py` + `scripts/scrape.py`)
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
- [x] MLflow tracking (`src/real_estate/tracking`)
- [x] SHAP analysis (`src/real_estate/explainability`, notebook 05)
- [x] Model evaluation (`src/real_estate/evaluacion`, notebook 06)
- [x] GitHub Actions (`ci.yml` + `dvc.yml`)
- [ ] Dockerfile
- [ ] docker-compose
- [ ] Deployment / API

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
modelo_precio_propiedades (versión con firma)
        ↓
notebooks/05_shap_analysis.ipynb ──→ src/real_estate/explainability/shap_analysis.py
                                      (TreeExplainer sobre val → valores/base → figuras)
        ↓
notebooks/06_model_evaluation.ipynb ──→ src/real_estate/evaluacion/analisis.py
                                         (residuos / error por segmento / sesgo por rango, sobre test)
        ↓
reports/figures/ (shap_*.png, evaluacion_*.png)
        ↓
Docker / Deployment
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
explain.py ──→ src/real_estate/explainability ──→ shap
evaluate.py ──→ src/real_estate/evaluacion ──→ scikit-learn / matplotlib
```

---

## 13. Diferencias entre la estructura conceptual y el estado real

> Registro explícito para no perder de vista la transición.

| Concepto | Realidad | Acción |
|---|---|---|
| `Makefile` | El archivo se llama `MakeFile` | Renombrar o aceptar el nombre actual |
| `scripts/scrape.py` | ✔ Migrado: `src/real_estate/ingestion/scraper.py` + `scripts/scrape.py`. La raíz quedó limpia | ✔ |
| `scripts/curate.py` | ✔ Implementado: `src/real_estate/curation/` (cleaning, transformations, validation, pipeline) + `scripts/curate.py` | ✔ |
| `scripts/` con 5 scripts | `scrape.py`, `curate.py`, `features.py` y `train.py` existen; faltan `evaluate.py`, `explain.py` | Parcial |
| `configs/config.yaml` | Carpeta creada, sin archivo | Pendiente |
| `dvc.yaml` / `dvc.lock` | ✔ Implementado: etapas `curar` y `features` con hashes md5. Remote por defecto `local` → `dvcstore/` | ✔ |
| `Dockerfile` / `docker-compose.yml` | No existen | Pendiente |
| `.github/workflows/ci.yml` | ✔ Implementado: lint (Ruff), type check (Mypy) y tests (Pytest + cobertura) en Python 3.11/3.12 | ✔ |
| `.github/workflows/dvc.yml` | ✔ Implementado: valida etapas (`stage list`), estado (`status`) y `pull` best-effort | ✔ |
| `data/raw/` | Contiene `propiedades_argenprop.csv` (2.005 registros) | ✔ |
| `data/processed/` | Contiene `propiedades_argenprop_curado.csv` (32 columnas) y `propiedades_argenprop_features.csv` (16 columnas) | ✔ |
| `tests/` | ✔ Implementado: unit (cleaning, scraper, transformations, features, models, tracking, explainability, evaluacion) + integration (pipeline) — 135 tests | ✔ |
| `src/real_estate/models` | ✔ Implementado en la fase 5: `entrenamiento.py` (baseline, XGBoost, evaluación sin fuga) | ✔ |
| `src/real_estate/explainability` | ✔ Implementado en la fase 7: `shap_analysis.py` (valores SHAP, base, importancia global, figuras) | ✔ |
| `src/real_estate/evaluacion` | ✔ Implementado en la fase 8: `analisis.py` (métricas detalladas, residuos, error por segmento, sesgo por rango, figuras) | ✔ |
| `notebooks/` | ✔ 01..06 ejecutados (estructura/calidad, precio/características, feature engineering, model analysis, shap analysis, model evaluation) | ✔ |
| `reports/figures/` | ✔ En uso: figuras SHAP (05) y de evaluación (06) (gitignored) | ✔ |
| `models/` | Vacía (esqueleto) | Pendiente |
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
