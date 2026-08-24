# Data Pipeline — Ingestión, Curación, Features y DVC

> Documenta el flujo de datos desde el scraping de Argenprop hasta la matriz
> de features lista para modelar.

---

## Flujo general

```text
Argenprop
    ↓
Scraper (scripts/scrape.py)
    ↓
data/raw/propiedades_argenprop.csv
    ↓
DVC (dvc.yaml → etapa curar)
    ↓
Data Curation (scripts/curate.py)
    ↓
data/processed/propiedades_argenprop_curado.csv
    ↓
DVC (dvc.yaml → etapa features)
    ↓
Feature Engineering (scripts/features.py)
    ↓
data/processed/propiedades_argenprop_features.csv
    ↓
Modelos
```

---

## 1. Ingestión — Scraping de Argenprop

**Módulo:** `src/real_estate/ingestion/scraper.py`
**Script:** `scripts/scrape.py`
**Target Makefile:** `make scrape`

### Fuente de datos

Argenprop publica avisos de venta de propiedades en CABA. El scraper obtiene
los datos del listado (sin visitar la página de detalle de cada aviso).

### Limitaciones del sitio

- **Cap de 100 páginas** por búsqueda (≈ 2.000 avisos). HTTP 202 vacío.
- **Segmentación:** se supera buscando por barrio (54 barrios de CABA) y/o tipo
  de propiedad. Cada segmento tiene su propio paginado y cap.
- **Throttle anti-bot:** HTTP 202 en páginas tempranas = bloqueo transitorio.
  Backoff exponencial + pausa larga.

### Campos crudos (20 columnas)

| Columna | Tipo | Notas |
|---|---|---|
| `id` | numérico | ID interno del aviso (confiable) |
| `link` | str | URL del aviso |
| `titulo` | str | Título del aviso |
| `descripcion` | str | Copete/descripción |
| `tipo_propiedad` | str | departamento, casa, ph, etc. |
| `barrio` | str | Ubicación |
| `precio` | numérico | Viene numérico (atributo `data`) |
| `moneda` | str | `USD` o `ARS` |
| `expensas` | texto | `"&plus; $2.200.000 expensas"` |
| `superficie_cubierta` | texto | `"300 m² cubie."` |
| `antiguedad` | texto | `"17 años"`, `"A estrenar"` |
| `ambientes` | numérico | Extraído de la URL |
| ... | ... | Ver `architecture.md` para la lista completa |

> **Nota:** `banos` y `cocheras` quedan mayormente vacíos por limitación del
> listado (no es un bug del scraper).

### Segmentación

El scraper soporta búsqueda por:
- `--tipo` (departamentos, casas, ph...)
- `--barrios` (slugs separados por coma)
- `--todos-los-barrios` (54 segmentos de CABA)
- `--max-paginas` (límite por segmento)

### Progreso y reanudación

- JSON de progreso (`--progreso`): `{"pagina": ultima_ok, "completo": bool}`
- Si se corta, al volver a correr reanuda desde la última página procesada.
- Segmentos completos se saltan (a menos que `--revision-periodica` esté activo).

---

## 2. Data Curation

**Módulos:** `src/real_estate/curation/` (cleaning, transformations, validation, pipeline)
**Script:** `scripts/curate.py`
**Target Makefile:** `make curate`

### Responsabilidades

1. **Limpieza de tipos** — texto crudo → numérico (separadores de miles, `m²`, `años`)
2. **Normalización de moneda** — todo a USD usando el tipo de cambio de la fecha de scrape
3. **Indicadores missing** — columnas `{columna}_informado` (int8) por cada columna afectada
4. **Validación** — reglas de coherencia (precio > 0, superficie > 0, ambientes >= 1)

### Tipo de cambio

- **Fuente primaria:** `data/external/tipo_cambio_blue.csv` (histórico, 5.702 fechas)
- **Fallback:** API de ArgentinaDatos (`/v1/cotizaciones/dolares/blue/{fecha}`)
- Descarga completa con `scripts/download_tipo_cambio.py`

### Ejemplo de transformación

| Crudo | → | Procesado |
|---|---|---|
| `"300 m² cubie."` | → | `300.0` |
| `"&plus; $2.200.000 expensas"` | → | `2200000.0` |
| `"17 años"` | → | `17.0` |
| moneda ARS | → | conversión a USD |

### Output

`data/processed/propiedades_argenprop_curado.csv` (2.005 filas × 32 columnas)

---

## 3. Feature Engineering

**Módulos:** `src/real_estate/features/` (transformations, pipeline)
**Script:** `scripts/features.py`
**Target Makefile:** `make features`

### Transformaciones

1. **Selección de columnas** — descarta cobertura < 3%, texto libre, identificadores
2. **Target logarítmico** — `log_precio_usd = ln(precio_usd)` (aproximadamente normal)
3. **Codificación ordinal** — `barrio` (44 categorías) y `tipo_propiedad` (7) por mediana de precio
4. **Imputación por mediana** — para features numéricas con NaN

### API de sin fuga

```python
# Ajustes se aprenden solo sobre train
ajustes = crear_orden_mediana(train)
imputador = crear_imputador(train)

# Se reaplican a val/test (sin fuga)
train_proc = codificar_ordinal(train, ajustes)
test_proc = codificar_ordinal(test, ajustes)
```

### Split

- 80% train / 10% val / 10% test
- Reproducible vía `random_state=42`
- `dividir_train_val_test()`

### Output

`data/processed/propiedades_argenprop_features.csv` (1.999 filas × 16 columnas, 0 faltantes)

---

## 4. DVC (Versionado de Datos)

**Configuración:** `dvc.yaml` + `dvc.lock`
**Remote:** local por defecto → `dvcstore/` (gitignored)

### Etapas del pipeline

| Etapa | Comando | Dependencias | Outputs |
|---|---|---|---|
| `curar` | `python scripts/curate.py` | scripts, curation, raw CSV | curado CSV |
| `features` | `python scripts/features.py` | scripts, features, curado CSV | features CSV |

### Comandos

```bash
dvc repro      # reproduce las etapas (solo si algo cambió)
dvc push       # sube datos al remote
dvc pull       # baja datos del remote
dvc status     # compara workspace vs remote
```

---

## 5. Datos

| Ruta | Contenido | Estado |
|---|---|---|
| `data/raw/propiedades_argenprop.csv` | Dataset crudo, 2.005 registros | ✔ |
| `data/processed/..._curado.csv` | Dataset curado, 32 columnas | ✔ |
| `data/processed/..._features.csv` | Matriz de features, 16 columnas | ✔ |
| `data/external/tipo_cambio_blue.csv` | Histórico dólar blue, 5.702 fechas | ✔ |
| `data/interim/` | Datos intermedios | vacío |
