# Roadmap — Predicción de precio + detección de oportunidades de compra

> **Documento de planificación.** Este archivo describe, fase por fase, cómo
> llevar el proyecto a su objetivo final: predecir el precio de propiedades
> con el modelo entrenado, y clasificar las **nuevas publicaciones** de
> Argenprop como **buena/mala compra** comparando el valor predicho contra el
> valor publicado. Cada fase se implementa siguiendo el flujo de calidad del
> proyecto (actualizar `docs/architecture.md`, pre-commit, commit y push).

---

## 1. Objetivo

El predictor ya entrena modelos que estiman el precio de una propiedad
(XGBoost, fase 5, trackeado en MLflow). Lo que falta es la **capa de uso**:

1. **Scrapear automáticamente** las publicaciones nuevas de Argenprop en un
   **dataset aparte** (no van al dataset de entrenamiento).
2. **Predecir el precio** de cada nueva publicación con el mejor modelo.
3. **Clasificar** cada publicación como **buena compra** (el modelo valora más
   que lo publicado) o **mala compra** (el modelo valora menos), con un
   **score relativo** que ordene las oportunidades.
4. **Experimentar** con más modelos (regresión lineal Lasso y Ridge) y
   explorar los hiperparámetros de XGBoost. Todo trackeado con MLflow.

---

## 2. Contexto sobre el que se apoya

| Pieza | Dónde vive | Estado |
|---|---|---|
| Scraper con dedup por `id`, backoff 202 y progreso | `src/real_estate/ingestion/scraper.py` + `scripts/scrape.py` | ✔ |
| Curación (USD, indicadores missing) | `src/real_estate/curation/` + `scripts/curate.py` | ✔ |
| Features + split train/val/test | `src/real_estate/features/` + `scripts/features.py` | ✔ |
| Modelos (baseline, XGBoost) | `src/real_estate/models/entrenamiento.py` + `scripts/train.py` | ✔ |
| Tracking MLflow | `src/real_estate/tracking/experimentos.py` | ✔ |
| Serving (bundle + predicción) | `src/real_estate/serving/` + `scripts/exportar_modelo.py` | ✔ |

---

## 3. Decisiones ya tomadas

| Pregunta | Decisión |
|---|---|
| Frecuencia del scraping automático | **Programado cada X días** (default: 7 días / semanal). La cadencia es configurable. |
| Umbral de clasificación | Zona neutra = **1 ± desviación estándar** del ratio `precio_predicho / precio_publicado` (computada por corrida sobre el lote de nuevas publicaciones). `ratio > 1 + std` → buena compra; `ratio < 1 - std` → mala compra; dentro de la zona → sin clasificar. |
| Datos intermedios en disco | **No.** Solo se persisten el CSV crudo de nuevas y el CSV final evaluado (con predicción y clasificación). Curado/features se procesan en memoria por corrida. |
| Fecha de publicación real (tarea #19) | **Pendiente.** Cuando se resuelva, se avisa al usuario y se decide si se incorpora al pipeline (afectaría antigüedad real y precio histórico). |
| Exclusiones | Optuna queda excluido (regla de `docs/architecture.md`); el tuning de XGBoost usa GridSearchCV/RandomizedSearchCV. |

---

## 4. Fase 1 — Dataset separado de nuevas publicaciones

**Objetivo:** scrapear las publicaciones nuevas en un dataset propio, sin tocar
el dataset de entrenamiento.

- **Script:** `scripts/scrape_nuevas.py`.
- **Output:** `data/raw/propiedades_nuevas.csv` (dedup interno por `id` contra
  este mismo archivo, no contra el de entrenamiento).
- **Progreso:** `data/raw/progreso_scrape_nuevas.json`.
- **Reutiliza** toda la maquinaria de `scraper.py` (HEADERS, `parsear_listing`,
  backoff 202, `cargar_ids_existentes`). Se agrega el parámetro
  `revision_periodica` a `scrapear()` para que una corrida programada
  re-escanee segmentos ya marcados como completos (las publicaciones nuevas
  aparecen después de la última corrida; el dedup evita duplicar).
- **Programación:** se agenda con un cron cada X días (default 7). La primera
  corrida captura el estado actual del mercado (baseline); las siguientes
  capturan solo lo nuevo.
- **Nota operativa:** el sitio bloquea (HTTP 202) si el ritmo es alto; el
  scraper ya maneja el backoff. Si una corrida se interrumpe, se reanuda por
  progreso. El alcance por corrida se acota con `--max-paginas` si se desea.

**Entregables:** script + tests unitarios + entrada en `docs/architecture.md`.

---

## 5. Fase 2 — Predicción de precio sobre las nuevas publicaciones

**Objetivo:** aplicar curación + features + modelo entrenado al dataset nuevo.

- **Script:** `scripts/evaluar_nuevas.py` (core en `src/real_estate/serving/evaluar.py`).
  1. Curación del dataset de nuevas en memoria (misma pipeline: limpieza,
     normalización de moneda a USD con el tipo de cambio de la fecha de
     scrape, expensas, indicadores de missing) — `curar_dataset`.
  2. Carga del bundle de serving exportado (`cargar_bundle`) y predicción de
     `precio_predicho_usd` (replica las features del entrenamiento vía
     `ModeloPrediccion`).
- **Output:** `data/processed/propiedades_nuevas_evaluadas.csv` con
  `precio_predicho_usd` y `fecha_prediccion` (además del precio publicado en USD).
- **Implementado ✔** (commit `bb6faab` es fase 1; fase 2 en este cambio):
  tests de integración `tests/integration/test_evaluar_nuevas.py` (6 tests)
  + entrada en `docs/architecture.md`.

---

## 6. Fase 3 — Clasificación buena/mala compra + score relativo

**Objetivo:** comparar el valor predicho contra el publicado, en absoluto y en
relativo.

- Por publicación: `ratio = precio_predicho_usd / precio_publicado_usd`.
- Se computa la **desviación estándar** del ratio sobre el lote de la corrida.
- Clasificación con zona neutra `1 ± std`:

| Condición | Clasificación |
|---|---|
| `ratio > 1 + std` | **Buena compra** |
| `ratio < 1 - std` | **Mala compra** |
| `1 - std <= ratio <= 1 + std` | **Sin clasificar** |

- El **score relativo** es el propio `ratio` (ordenado descendente = mejores
  oportunidades primero). Ejemplo del usuario:
  - Modelo predice 100.000 USD, publicado en 80.000 → ratio 1.25.
  - Modelo predice 60.000 USD, publicado en 40.000 → ratio 1.50.
  - **Ambas son buenas compras**, pero la segunda es relativamente mejor
    (ratio mayor).
- **Outputs:** `reports/ofertas.csv` + ranking por ratio + notebook
  `07_deteccion_oportunidades.ipynb` con la distribución de ratios.

---

## 7. Fase 4 — Modelos lineales: Lasso y Ridge

**Objetivo:** comparar contra XGBoost, mismo pipeline y misma validación.

- Nuevo módulo `src/real_estate/models/modelos_lineales.py` (o extensión de
  `entrenamiento.py`).
- Misma estructura train/val/test y mismas features que XGBoost (comparación
  justa). Requiere que las features numéricas estén imputadas y escaladas
  (los lineales son sensibles a la escala; los árboles no).
- Un run de MLflow por modelo con métricas idénticas (RMSE log, RMSE USD, R²).

---

## 8. Fase 5 — Tuning de hiperparámetros de XGBoost

**Objetivo:** explorar el espacio de hiperparámetros y elegir un campeón.

- `GridSearchCV`/`RandomizedSearchCV` sobre el conjunto de validación
  (Optuna queda excluido por regla del proyecto).
- Espacio: `n_estimators`, `max_depth`, `learning_rate`, `subsample`,
  `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`.
- Un run de MLflow por trial; al final se compara y el mejor se registra como
  campeón.

---

## 9. Fase 6 — MLflow integral

**Objetivo:** que todo el flujo quede trackeado y comparable.

- Por cada modelo/trial: parámetros, métricas y artefactos (importancia de
  features, curvas, SHAP).
- Comparación de runs para decidir el modelo de producción (champion).
- Reutiliza `tracking/experimentos.py`; se agrega el registro del champion al
  exportar el modelo de serving.

---

## 10. Orden de implementación

1. **Fase 1** (dataset separado + programación) → base de todo. **✔ COMPLETADA.**
2. **Fases 2 + 3** (predicción + clasificación) → el valor central. **Fase 2 ✔
   COMPLETADA** (script + tests + docs); **fase 3 EN CURSO.**
3. **Fases 4 + 5** (modelos lineales + tuning) → en paralelo, ambos dependen
   solo del pipeline existente.
4. **Fase 6** (MLflow) → transversal, se va haciendo en cada fase.

## 11. Pendientes abiertos

- [ ] **Tarea #19:** fecha de publicación real desde la página de detalle.
      Queda abierta; al resolverse se avisa al usuario y se evalúa incorporarla.
- [ ] Definir el valor exacto de "cada X días" para el cron (default 7).
- [ ] Decidir si el scraping programado re-escaneará el segmento completo o un
      número acotado de páginas por corrida.
