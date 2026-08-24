# Models — Entrenamiento, Tuning, Evaluación y Explicabilidad

> Documenta el ciclo de vida de los modelos: desde el baseline hasta la
> explicabilidad con SHAP.

---

## Flujo general

```text
features CSV
    ↓
Train / Validation / Test (80/10/10)
    ↓
Baseline (mediana)
    ↓
XGBoost (default)
    ↓
Lasso / Ridge (roadmap fase 4)
    ↓
Tuning de XGBoost (roadmap fase 5)
    ↓
MLflow tracking
    ↓
Comparación y champion (roadmap fase 6)
    ↓
Evaluación profunda
    ↓
SHAP (explicabilidad)
    ↓
Bundle de serving
```

---

## 1. Entrenamiento

**Módulo:** `src/real_estate/models/entrenamiento.py`
**Script:** `scripts/train.py`
**Target Makefile:** `make train`

### Pipeline de entrenamiento

```python
resultado = entrenar_y_evaluar(train, val, test, random_state=42)
```

1. Ajusta preprocesamiento sobre **solo train** (sin fuga)
2. Entrena baseline (DummyRegressor median)
3. Entrena XGBoost (300 árboles, depth 4, lr 0.05)
4. Evalúa en val y test

### Métricas

| Métrica | Descripción |
|---|---|
| RMSE log | Error en espacio logarítmico (≈ error relativo) |
| RMSE USD | Error en dólares (deshace el log con `exp()`) |
| R² | Coeficiente de determinación |

### Resultados sobre el dataset real

| Modelo | RMSE log (val) | RMSE USD (val) | R² (val) | RMSE log (test) | R² (test) |
|---|---|---|---|---|---|
| baseline (mediana) | 0.6423 | $185.135 | -0.0025 | — | — |
| XGBoost (default) | 0.2718 | $112.963 | 0.8205 | 0.3040 | 0.7830 |

---

## 2. Modelos Lineales (Lasso / Ridge)

**Módulo:** `src/real_estate/models/modelos_lineales.py`
**Script:** `scripts/train_lineales.py`
**Target Makefile:** `make train-lineales`

### Diseño

- `StandardScaler` dentro del pipeline (los lineales son sensibles a la escala)
- Mismo preprocesamiento y features que XGBoost (comparación justa)
- Una corrida MLflow por modelo, sin Model Registry

### Resultados

| Modelo | RMSE log (test) | R² (test) |
|---|---|---|
| Ridge (alpha=1.0) | 0.3372 | 0.7330 |
| XGBoost (default) | 0.3040 | 0.7830 |

**Conclusión:** Ridge es el mejor lineal pero XGBoost sigue siendo el campeón.

---

## 3. Tuning de Hiperparámetros

**Módulo:** `src/real_estate/models/tuning.py`
**Script:** `scripts/train_tuning.py`
**Target Makefile:** `make tuning`

### Espacio de búsqueda

9 hiperparámetros: `n_estimators`, `max_depth`, `learning_rate`, `subsample`,
`colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`.

### Métodos

- `grid` — GridSearchCV sobre `GRID_REDUCIDO` (8 combinaciones)
- `random` — RandomizedSearchCV sobre el espacio completo

### Resultados (`--metodo grid`)

CV RMSE log 0.2886, pero test RMSE log 0.3020 vs 0.3040 del default.
**Conclusión:** el default ya era casi óptimo para este dataset.

---

## 4. Experiment Tracking (MLflow)

**Módulos:** `src/real_estate/tracking/` (experimentos.py, comparacion.py)

### Qué se trackea

- **Parámetros:** hiperparámetros del modelo, random_state, tamaños de split
- **Métricas:** RMSE log, RMSE USD, R² (val y test)
- **Artefactos:** `resumen_entrenamiento.json`, `feature_importances.json`
- **Modelo:** XGBoost con firma de entrada/salida en Model Registry

### Champion

`scripts/comparar_runs.py` (`make compare`) lista las corridas con
`xgboost_test_rmse_log` de mejor a peor y elige el champion.

---

## 5. Evaluación Profunda

**Módulo:** `src/real_estate/evaluacion/analisis.py`
**Script:** `scripts/evaluate.py`
**Target Makefile:** `make evaluate`

### Análisis

1. **Métricas detalladas** — RMSE log, RMSE/MAE/MedAE/MAPE en USD, R²
2. **Residuos** — precio real/predicho, error relativo, sesgo medio
3. **Por segmento** — error agrupado por tipo_propiedad, barrio, ambientes
4. **Sesgo por rango** — sobreestima lo barato, subestima lo caro

### Resultados (test)

- RMSE log 0.3040, RMSE USD $87.244, R² 0.7830
- Sesgo medio +5.02% (sobreestima levemente)
- Error relativo absoluto mediano 17.73%

---

## 6. Explicabilidad (SHAP)

**Módulo:** `src/real_estate/explainability/shap_analysis.py`
**Script:** `scripts/explain.py`
**Target Makefile:** `make explain`

### Análisis

- **TreeExplainer** sobre el modelo XGBoost ya entrenado
- **Propiedad aditiva:** `base + Σ valores ≈ predict(X)` (error máx. 8.6e-06)
- **Importancia global:** media |SHAP| por feature

### Top-3 features (por media |SHAP|)

1. `superficie_cubierta` (0.30)
2. `barrio_ordinal` (0.14)
3. `expensas_usd` (0.07)

### Figuras

- `shap_resumen.png` (beeswarm)
- `shap_importancia.png` (barras)

---

## 7. Scripts

| Script | Target | Descripción |
|---|---|---|
| `scripts/train.py` | `make train` | Entrenamiento + tracking MLflow |
| `scripts/train_lineales.py` | `make train-lineales` | Lasso/Ridge + tracking |
| `scripts/train_tuning.py` | `make tuning` | Tuning XGBoost + tracking |
| `scripts/comparar_runs.py` | `make compare` | Elegir champion |
| `scripts/evaluate.py` | `make evaluate` | Evaluación profunda |
| `scripts/explain.py` | `make explain` | Explicabilidad SHAP |
