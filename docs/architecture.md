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
- **Linting** (Ruff)
- **Type checking** (Mypy en modo `strict`)
- **CI** (GitHub Actions)
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
- **DVC** todavía no figura como dependencia Python en `pyproject.toml` porque la
  capa de versionado de datos aún no está implementada. Se agregará cuando exista
  `dvc.yaml` / `dvc.lock`.
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
| Data versioning | DVC | Versionado de datos | ✘ pendiente (aún no es dependencia) |
| Testing | Pytest, pytest-cov | Tests | ✔ dependencia dev |
| Code quality | Ruff, Mypy | Lint + type check | ✔ dependencia dev |
| Git hooks | pre-commit | Gates antes del commit | ✔ dependencia dev |
| CI/CD | GitHub Actions | Integración continua | ✘ pendiente (workflows vacíos) |
| Containerization | Docker, Docker Compose | Contenedores | ✘ pendiente |
| Configuración | Pydantic Settings, python-dotenv | Configuración y env vars | ✔ dependencia |
| Jupyter | Jupyter, IPython kernel | Notebooks de análisis | ✔ dependencia dev |

**Exclusiones explícitas:** Optuna (no se usará). DVC como dependencia aún no
(se agrega cuando se implemente la capa).

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
│   ├── raw/    propiedades_argenprop.csv   ✔ (2.005 registros)
│   ├── interim/                             🏗 vacía
│   ├── processed/ propiedades_argenprop_curado.csv   ✔ (32 columnas)
│   └── external/                            🏗 vacía
│
├── dvc.yaml                        ✘
├── dvc.lock                        ✘
│
├── notebooks/                      🏗 vacía (se planifican 01..05)
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
│       ├── features/              🏗 vacío
│       ├── models/                🏗 vacío
│       ├── explainability/        🏗 vacío
│       ├── tracking/              🏗 vacío
│       └── utils/                 🏗 vacío
│
├── scripts/
│   ├── scrape.py                   ✔ (entry point de adquisición)
│   └── curate.py                   ✔ (entry point de curación)
│   (se planifican train/evaluate/explain)
│
├── tests/
│   ├── unit/                       🏗 vacía
│   └── integration/                🏗 vacía
│
├── models/                         🏗 vacía
│
├── reports/
│   ├── figures/                    🏗 vacía
│   └── metrics/                    🏗 vacía
│
├── mlruns/                         🏗 vacía
│
├── docs/
│   └── architecture.md             ✔ (este documento)
│
└── .github/
    └── workflows/
        ├── ci.yml                  ✘
        └── dvc.yml                 ✘
```

> **Nota:** el directorio **no es todavía un repositorio Git** (`git init` pendiente).

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
| **Dependencias dev** | pytest, pytest-cov, ruff, mypy, pandas-stubs, pre-commit, jupyter, ipykernel |
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
| **Targets actuales** | `install`, `install-dev`, `format`, `lint`, `typecheck`, `test`, `coverage`, `check`, `clean` (además de `help`) |
| **Targets futuros** | `scrape`, `curate`, `train`, `evaluate`, `explain`, `dvc`, `docker-...` — solo cuando los componentes existan realmente |
| **Dependencias** | `pyproject.toml` (comandos pip/pytest/ruff/mypy) |
| **Usado por** | Desarrolladores, CI (futuro) |
| **Nota** | El esquema conceptual lo llama `Makefile`; el archivo real en disco es `MakeFile`. |

#### `pre-commit-config.yaml`

| Atributo | Valor |
|---|---|
| **Propósito** | Gates de calidad antes de cada commit |
| **Responsabilidades** | Ejecutar Ruff (check + format), Mypy, y validaciones generales |
| **Hooks** | `ruff-check` (`--fix`), `ruff-format`, `mypy` (+ `pandas-stubs`, `types-requests`), `check-yaml`, `check-toml`, `end-of-file-fixer`, `trailing-whitespace`, `check-added-large-files` (`--maxkb=1000`) |
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

### 9.4 Paquete `src/real_estate/` — subpaquetes pendientes

Paquete descubrible vía `pyproject.toml` (layout `src/`). Import como
`real_estate.*`. `ingestion/` y `curation/` ya están implementados (secciones
9.2 y 9.3). Los siguientes subpaquetes existen pero están vacíos:

| Subpaquete | Responsabilidad planificada |
|---|---|
| `features/` | Feature engineering (features derivadas, splits train/val/test) |
| `models/` | Baseline, XGBoost, entrenamiento y evaluación |
| `explainability/` | SHAP analysis |
| `tracking/` | Integración con MLflow (experimentos, registry) |
| `utils/` | Utilidades transversales (config con pydantic-settings, logging, etc.) |

### 9.5 `scripts/`

| Script | Estado | Responsabilidad |
|---|---|---|
| `scrape.py` | ✔ implementado | Orquestar la adquisición (usa `src/real_estate/ingestion`) |
| `curate.py` | ✔ implementado | Orquestar la curación (usa `src/real_estate/curation`) |
| `train.py` | ✘ pendiente | Orquestar el entrenamiento (usaría `src/real_estate/models` + `tracking`) |
| `evaluate.py` | ✘ pendiente | Evaluación de modelos |
| `explain.py` | ✘ pendiente | Explicabilidad (SHAP) |

**Relación con MakeFile:** `make scrape` y `make curate` ya existen (aceptan
`ARGS="..."`). A medida que existan los demás scripts, se agregan targets
`make train`, `make evaluate`, `make explain`.

### 9.6 `notebooks/` (pendiente)

| Notebook planificado | Tema |
|---|---|
| `01_data_exploration.ipynb` | Exploración inicial |
| `02_data_quality.ipynb` | Análisis de calidad de datos |
| `03_feature_engineering.ipynb` | Ingeniería de features |
| `04_model_analysis.ipynb` | Análisis de modelos |
| `05_shap_analysis.ipynb` | SHAP / explicabilidad |

### 9.7 `tests/` (pendiente)

| Carpeta | Contenido planificado |
|---|---|
| `tests/unit/` | Tests unitarios (parsers, curación, features, etc.) |
| `tests/integration/` | Tests de integración (pipelines, scripts) |

### 9.8 `configs/config.yaml` (pendiente)

Configuración centralizada del proyecto (parámetros de scraper, curación,
entrenamiento, MLflow). Se cargaría vía pydantic-settings.

### 9.9 DVC (pendiente)

`dvc.yaml` + `dvc.lock` para versionar datos (raw → curation → processed).
Workflow de DVC en `.github/workflows/dvc.yml` (pendiente).

### 9.10 Docker (pendiente)

`Dockerfile` + `docker-compose.yml` para contenerizar el proyecto / el servicio
de predicción (deployment).

### 9.11 CI/CD (pendiente)

`ci.yml`: ejecutar Pytest / Ruff / Mypy en GitHub Actions; `dvc.yml`: reproducir
el pipeline de datos. Workflows creados como carpeta vacía.

### 9.12 Datos

| Ruta | Contenido | Estado |
|---|---|---|
| `data/raw/propiedades_argenprop.csv` | Dataset crudo, 2.005 registros, 20 columnas | ✔ |
| `data/interim/` | Datos intermedios (p. ej., entre curación y features) | 🏗 vacía |
| `data/processed/propiedades_argenprop_curado.csv` | Dataset curado, 2.005 filas, 32 columnas (listo para EDA/features) | ✔ |
| `data/external/` | Datos externos (p. ej., tipo de cambio) | 🏗 vacía |

### 9.13 Otros directorios

| Ruta | Estado | Nota |
|---|---|---|
| `models/` | 🏗 vacía | Artefactos de modelos (futuro) |
| `reports/figures/` | 🏗 vacía | Figuras para reportes |
| `reports/metrics/` | 🏗 vacía | Métricas de evaluación |
| `mlruns/` | 🏗 vacía | Ruta local de MLflow (futuro) |
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

- [ ] Tests del código del proyecto (`tests/`) — hoy solo verificación manual
- [ ] DVC (`dvc.yaml`, `dvc.lock`, dependencia)
- [ ] EDA estructurado (notebooks 01/02)
- [ ] Feature Engineering (`src/real_estate/features`, notebook 03)
- [ ] Pipeline Train / Validation / Test
- [ ] Baseline
- [ ] XGBoost training pipeline
- [ ] MLflow tracking (`src/real_estate/tracking`)
- [ ] Model evaluation
- [ ] SHAP analysis (`src/real_estate/explainability`, notebook 05)
- [ ] GitHub Actions (`ci.yml`, `dvc.yml`)
- [ ] Dockerfile
- [ ] docker-compose
- [ ] Deployment / API
- [ ] `git init` del repositorio

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
data/raw/propiedades_argenprop.csv
        ↓
scripts/curate.py ──→ src/real_estate/curation/pipeline.py
                      ├── cleaning.py
                      ├── transformations.py
                      └── validation.py
        ↓
data/processed/propiedades_argenprop_curado.csv
        ↓
features/          (pendiente)
        ↓
models/ (Baseline → XGBoost)
        ↓
tracking/ (MLflow)
        ↓
explainability/ (SHAP)  +  models/ (artefacto) → Docker / Deployment
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
train.py  ──→ src/real_estate/models ──→ scikit-learn / xgboost
             └──→ src/real_estate/tracking ──→ mlflow
explain.py ──→ src/real_estate/explainability ──→ shap
```

---

## 13. Diferencias entre la estructura conceptual y el estado real

> Registro explícito para no perder de vista la transición.

| Concepto | Realidad | Acción |
|---|---|---|
| `Makefile` | El archivo se llama `MakeFile` | Renombrar o aceptar el nombre actual |
| `scripts/scrape.py` | ✔ Migrado: `src/real_estate/ingestion/scraper.py` + `scripts/scrape.py`. La raíz quedó limpia | ✔ |
| `scripts/curate.py` | ✔ Implementado: `src/real_estate/curation/` (cleaning, transformations, validation, pipeline) + `scripts/curate.py` | ✔ |
| `scripts/` con 5 scripts | `scrape.py` y `curate.py` existen; faltan `train.py`, `evaluate.py`, `explain.py` | Parcial |
| `configs/config.yaml` | Carpeta creada, sin archivo | Pendiente |
| `dvc.yaml` / `dvc.lock` | No existen | Pendiente |
| `Dockerfile` / `docker-compose.yml` | No existen | Pendiente |
| `.github/workflows/{ci,dvc}.yml` | Carpeta creada, vacía | Pendiente |
| `data/raw/` | Contiene `propiedades_argenprop.csv` (2.005 registros) | ✔ |
| `data/processed/` | Contiene `propiedades_argenprop_curado.csv` (32 columnas) | ✔ |
| `notebooks/`, `tests/`, `reports/`, `models/`, `mlruns/` | Vacías (esqueleto) | Pendiente |
| Repositorio Git | No inicializado | Pendiente |
| `README.md` | ✔ Actualizado: documenta scrape y curate vía `scripts/`; ya no menciona `scraper_argenprop2.py` | ✔ |

> **Nota de entorno (resuelta):** el ambiente local tiene Python 3.14.7 +
> numpy 2.5.2, cuyos stubs usan `type` statements (PEP 695, solo Python ≥ 3.12).
> `pyproject.toml` fija `[tool.mypy] python_version = "3.12"` (dentro del rango
> soportado `>=3.11,<3.14`), lo que hace que `mypy` plano, el hook de pre-commit
> y el CI futuro funcionen de forma consistente. El hook de mypy en pre-commit
> además declara `pandas-stubs` y `types-requests` como dependencias extra.

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
