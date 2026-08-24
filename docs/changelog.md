# Changelog — Mejoras recientes

> Registro de las implementaciones realizadas en las sesiones recientes.
> Cada entrada documenta qué se hizo, por qué y en qué archivos.

---

## [Unreleased] — 2026-08-24

### Added

#### 13. Frontend con explicabilidad SHAP y auto-refresh

**Problema:** no había interfaz visual para explorar las oportunidades de compra
ni para entender por qué el modelo predijo un precio específico.

**Solución:**
- Creado `frontend/index.html` — tabla de oportunidades con filtros y modal
- Creado `frontend/style.css` — diseño responsive con variables CSS
- Creado `frontend/app.js` — lógica: fetch API, renderizado, auto-refresh 30s
- Nuevo endpoint `GET /oportunidades/{id}/explain` — SHAP values por publicación
- Raw features (superficie, ambientes, etc.) almacenadas en tabla `oportunidades`
- Esquemas Pydantic: `ExplicacionFeature`, `ExplicacionPublicacion`
- Tests: 4 tests para endpoint explain

#### 14. Comandos Makefile para frontend y ETL

**Problema:** faltaban targets para el scraping de CABA completo y para
servir el frontend.

**Solución:**
- `make etl-scrape` — ETL con scraping de todos los barrios de CABA
- `make serve` — FastAPI + Frontend en `0.0.0.0:8000`
- `make serve-api` — Solo FastAPI (sin frontend)

---

## [Unreleased] — 2026-08-23

### Added

#### 1. Logging estructurado (reemplaza `print()`)

**Problema:** el proyecto usaba `print()` para logging, sin niveles ni formato
consistente.

**Solución:**
- Creado `src/real_estate/utils/logging.py` con `configurar_logging()` centralizada
- Creado `src/real_estate/utils/__init__.py`
- Reemplazados todos los `print()` por `logger.info/warning/error` en:
  - `src/real_estate/curation/` (cleaning, transformations, validation, pipeline)
  - `src/real_estate/features/` (transformations, pipeline)
  - `src/real_estate/models/` (entrenamiento, tuning, modelos_lineales)
  - `src/real_estate/serving/` (evaluar, clasificacion)
  - `src/real_estate/ingestion/scraper.py`
  - `src/real_estate/persistencia/etl_oportunidades.py`
  - Todos los scripts (14 archivos)

#### 2. Configuración centralizada (pydantic-settings + YAML)

**Problema:** parámetros hardcodeados dispersos en múltiples módulos
(`FX_MARKET`, `PRECIO_MINIMO_USD`, `PARAMS_XGBOOST_DEFAULT`, etc.).

**Solución:**
- Creado `configs/config.yaml` con todas las secciones del proyecto
- Creado `src/real_estate/utils/config.py` con sub-modelos pydantic y
  `ConfiguracionProyecto` que carga desde YAML + env vars (prefijo `RE_`)
- Agregada dependencia `pyyaml` a `pyproject.toml` y `requirements-api.txt`
- Agregado `types-PyYAML` como dependencia dev

#### 3. Usuario non-root en Dockerfile

**Problema:** la imagen Docker ejecutaba la app como root.

**Solución:**
- Agregado `useradd --create-home appuser` (UID 1000) en la etapa runtime
- `USER appuser` antes de `EXPOSE`

#### 4. Health check con estado de DB

**Problema:** `/health` no reportaba el estado de la base de datos.

**Solución:**
- Endpoint `/health` ahora incluye `"base_datos": "ok" | "no_configurado" | "no_disponible"`
- Consulta `SELECT 1` perezosa solo si el engine está configurado

#### 5. Rate limiting en API (SlowAPI)

**Problema:** la API no tenía protección contra abuso.

**Solución:**
- Agregado `slowapi` como dependencia (pyproject.toml + requirements-api.txt)
- 60 requests/min por IP en `/predict`
- Libre en `/health`

#### 6. Scripts evaluate.py y explain.py

**Problema:** faltaban los entry points CLI para evaluación profunda y
explicabilidad SHAP.

**Solución:**
- Creado `scripts/evaluate.py` (198 lines) — evaluación con métricas detalladas,
  residuos, análisis por segmento y sesgo por rango
- Creado `scripts/explain.py` (142 lines) — SHAP values, importancia global,
  figuras beeswarm y barras
- Agregados targets `make evaluate` y `make explain` al Makefile

#### 7. Checksum SHA-256 en bundle de serving

**Problema:** no había verificación de integridad del bundle de serving.

**Solución:**
- `guardar_bundle()` genera `checksum.json` con hashes SHA-256 de los 4 archivos
- `cargar_bundle()` verifica integridad antes de reconstruir el modelo
- Implementado en ambos módulos: `persistencia/bundle.py` y `serving/persistencia.py`
- Tests: `test_checksum_es_archivo_json_valido`, `test_checksum_verifica_integridad`

#### 8. Validación de tipo_propiedad en API

**Problema:** la API aceptaba cualquier string como `tipo_propiedad`.

**Solución:**
- Validación contra los 7 tipos conocidos: departamento, casa, ph, terreno,
  local, oficina, cochera
- `barrio` se acepta como string libre (puede ser nuevo; se loguea warning)

#### 9. Fix repositorio.py None guard

**Problema:** `ids_procesados()` podía intentar convertir `None` a string.

**Solución:**
- Agregado `if fila[0] is not None` en el set comprehension

#### 10. Tests de integración adicionales

**Solución:**
- Tests de checksum en `test_serving.py` (valid JSON + tamper detection)
- Tests de validación de API en `test_api.py` (tipo_propiedad desconocido 422,
  barrio desconocido aceptado)

#### 11. Makefile rename + targets

**Solución:**
- Renombrado `MakeFile` → `Makefile`
- Agregados targets `evaluate` y `explain`
- Actualizado help con las nuevas opciones

#### 12. Documentación refactorizada

**Problema:** toda la documentación estaba concentrada en `architecture.md`
(1400+ líneas) y `roadmap.md`, sin responsabilidades claras.

**Solución:** documentación dividida en 8 archivos:
- `overview.md` — resumen y principios del proyecto
- `architecture.md` — registro de componentes (refactorizado)
- `data-pipeline.md` — ingestión, curación, features, DVC
- `models.md` — entrenamiento, tuning, evaluación, explicabilidad
- `serving.md` — bundle, API, Docker, persistencia
- `ci-cd.md` — workflows y calidad de código
- `roadmap.md` — checklist limpio de fases
- `changelog.md` — este archivo
