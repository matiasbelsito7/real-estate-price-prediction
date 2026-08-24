# Serving — Bundle, API, Docker y Persistencia

> Documenta cómo se expone el modelo como servicio HTTP y cómo persisten
> las oportunidades detectadas.

---

## Flujo general

```text
make export-model
    ↓
models/modelo_precio_propiedades/ (bundle)
    ↓
make serve (FastAPI + Frontend)
    ↓
/  →  Frontend (HTML/CSS/JS)
/health  /predict  /oportunidades  /oportunidades/{id}/explain
    ↓
make docker-build → make docker-up
    ↓
Contenedor Docker (non-root, healthcheck)
```

---

## 1. Bundle de Serving

**Módulos:** `src/real_estate/serving/persistencia.py`, `src/real_estate/persistencia/bundle.py`
**Script:** `scripts/exportar_modelo.py`
**Target Makefile:** `make export-model`

### Contenido del bundle

| Archivo | Contenido |
|---|---|
| `modelo_xgboost.json` | XGBoost guardado con `save_model` nativo |
| `preprocesamiento.json` | Ordenes ordinales + imputador por mediana |
| `features.json` | Orden exacto de las 14 features |
| `metadata.json` | Métricas en test, tamaños de split, fecha de exportación |
| `checksum.json` | Hashes SHA-256 de los 4 archivos anteriores |

### Integridad (checksum SHA-256)

Al guardar el bundle, se genera `checksum.json` con los hashes de cada archivo.
Al cargar, se verifica la integridad antes de reconstruir el modelo. Si hay
mismatch, se lanza `ValueError`.

### Por qué bundle propio y no Model Registry

MLflow loguea solo el XGBoost (sin firma de preprocesamiento); el
`Preprocesamiento` (codificación ordinal + imputación) no se persiste por MLflow.

---

## 2. API FastAPI

**Módulos:** `src/real_estate/api/` (app.py, schemas.py, config.py)
**Target Makefile:** `make serve`

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Frontend (redirige a `/frontend/index.html`) |
| GET | `/health` | Estado, modelo, versión, métricas, estado de DB |
| POST | `/predict` | Predicción de precio (14 features) |
| GET | `/oportunidades` | Lista de oportunidades (paginado, filtros) |
| GET | `/oportunidades/{id}` | Detalle de una oportunidad |
| GET | `/oportunidades/{id}/explain` | Explicabilidad SHAP de una publicación |

### Rate limiting

- SlowAPI con `key_func=get_remote_address`
- 60 requests/min por IP en `/predict`
- Libre en `/health`

### Validación de entrada

- `tipo_propiedad` debe ser uno de los 7 conocidos (departamento, casa, ph, terreno, local, oficina, cochera)
- `barrio` se acepta como string (puede ser nuevo; se loguea warning si no está en el set de entrenamiento)
- Campos numéricos con `ge=0`
- Indicadores `*_informado` opcionales (0/1)

### Contrato de entrada

14 features:
- 6 numéricas imputables: `superficie_cubierta`, `ambientes`, `dormitorios`, `banos`, `antiguedad`, `expensas_usd`
- 6 indicadores `*_informado`
- 2 ordinales: `barrio_ordinal`, `tipo_propiedad_ordinal`

---

## 2.5 Frontend

**Directorio:** `frontend/`
**Target Makefile:** `make serve` (incluido en FastAPI)

### Funcionalidades

- **Tabla de oportunidades** con clasificación, precios y diferencia porcentual
- **Filtros** por clasificación y barrio
- **Paginación** y ordenamiento por columnas
- **Modal de explicabilidad SHAP** por publicación
- **Auto-refresh** cada 30 segundos
- **Diseño responsive** (mobile-friendly)

### Tecnologías

- HTML5, CSS3 (variables CSS), JavaScript vanilla
- Sin dependencias externas (Framework-free)
- Fetch API para comunicación con FastAPI

### Rutas

| Ruta | Descripción |
|---|---|
| `GET /` | Redirige al frontend |
| `GET /frontend/index.html` | Página principal |
| `GET /frontend/style.css` | Estilos |
| `GET /frontend/app.js` | Lógica JavaScript |

### Explicabilidad SHAP

El botón "Explicar" en cada fila abre un modal con:
- Precio predicho por el modelo
- Resumen legible de contribuciones
- Barras de contribución por feature (verde = aumenta precio, rojo = reduce)
- Features ordenadas por importancia absoluta

---

## 3. Docker

**Archivos:** `Dockerfile`, `docker-compose.yml`, `requirements-api.txt`

### Dockerfile multi-stage

1. **build** (`python:3.12-slim`): crea venv con dependencias de serving
2. **runtime** (`python:3.12-slim`): copia venv, bundle, usuario non-root

### Seguridad

- Usuario `appuser` (UID 1000) — no ejecuta como root
- `HEALTHCHECK` sobre `/health` cada 30s

### docker-compose.yml

| Servicio | Imagen | Descripción |
|---|---|---|
| `api` | build local | FastAPI con bundle montado como volumen |
| `postgres` | postgres:16-alpine | Base de oportunidades del ETL |

### Uso

```bash
make export-model    # genera el bundle
make docker-build    # construye la imagen
make docker-up       # levanta api + postgres
make docker-logs     # sigue los logs
```

---

## 4. Persistencia PostgreSQL (ETL periódico)

**Módulos:** `src/real_estate/persistencia/` (config, db, esquema, repositorio)
**Script:** `scripts/etl_oportunidades.py`
**Target Makefile:** `make etl`

### Tabla `oportunidades`

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | String(64) PK | ID del aviso |
| `titulo` | Text | Título del aviso |
| `link` | Text | URL del aviso |
| `barrio` | String | Barrio |
| `tipo_propiedad` | String | Tipo de propiedad |
| `precio_usd` | Float | Precio publicado en USD |
| `precio_predicho_usd` | Float | Precio predicho por el modelo |
| `ratio_precio` | Float | `predicho / publicado` |
| `diferencia_usd` | Float | Diferencia en USD |
| `diferencia_porcentual` | Float | Diferencia en % |
| `clasificacion` | String | buena_compra / precio_justo / mala_compra |
| `modelo_version` | String | Versión del modelo usado |
| `fecha_prediccion` | Date | Fecha de la predicción |
| `actualizado_en` | Timestamp | Última actualización |
| `superficie_cubierta` | Float | Superficie cubierta (m²) — para SHAP |
| `ambientes` | Float | Cantidad de ambientes — para SHAP |
| `dormitorios` | Float | Cantidad de dormitorios — para SHAP |
| `banos` | Float | Cantidad de baños — para SHAP |
| `antiguedad` | Float | Antigüedad (años) — para SHAP |
| `expensas_usd` | Float | Expensas mensuales USD — para SHAP |

### Upsert multi-dialecto

El `ON CONFLICT DO UPDATE` se construye con el `insert()` del dialecto activo
(PostgreSQL o SQLite), elegido en runtime.

### Flujo del ETL

```text
scrape (revision_periodica, CABA)
    ↓
data/raw/propiedades_nuevas.csv
    ↓
dedup contra la base (ids_procesados)
    ↓
predicción + clasificación por propiedad
    ↓
upsert en PostgreSQL
    ↓
reports/oportunidades_nuevas.csv
```

---

## 5. CD del Champion

**Script:** `scripts/publicar_champion.py`
**Workflow:** `.github/workflows/cd_champion.yml`

### Flujo

1. `make export-model` → `make publicar-champion`
2. Fingerprint determinístico en `models/champion_actual.json`
3. Push del fingerprint a `main` dispara el CD
4. CD: restaura bundle → smoke test → build imagen → push a GHCR

### Fingerprint

- Determinístico (sin marcas de tiempo)
- Si el champion no cambió, el archivo no cambia y no se re-dispara el CD
