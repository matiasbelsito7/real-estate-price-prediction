"""
Tuning de hiperparámetros de XGBoost (Fase 5 del roadmap).

Explora el espacio de hiperparámetros de XGBoost (`n_estimators`,
`max_depth`, `learning_rate`, `subsample`, `colsample_bytree`,
`min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`) con
`GridSearchCV` o `RandomizedSearchCV` (Optuna queda excluido por regla de
`docs/architecture.md`), sobre el mismo pipeline train/val/test sin fuga:

1. Preprocesamiento ajustado solo sobre train y reaplicado a val/test
   (mismas features y mismo target logarítmico que el resto del pipeline).
2. La búsqueda se hace con validación cruzada interna sobre train
   (scoring RMSE log negativo), sin tocar val/test.
3. El mejor candidato (refit sobre train) se evalúa sobre val y test.

`tunear_xgboost` devuelve `ResultadoTuning` con las métricas del XGBoost
actual (default) contra las del tunedo, el mejor RMSE log de CV, la tabla de
trials ordenada por ranking y el modelo ajustado. El tracking con MLflow
queda en `real_estate.tracking.experimentos.registrar_tuning` (un run por
trial, anidado bajo un run resumen). No se versiona nada en el Model
Registry: el champion se elige y registra recién en la fase 6.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor

from real_estate.features.transformations import TARGET_LOG
from real_estate.models.entrenamiento import (
    PARAMS_XGBOOST_DEFAULT,
    Preprocesamiento,
    ajustar_preprocesamiento,
    aplicar_preprocesamiento,
    calcular_metricas,
    entrenar_xgboost,
    mostrar_metricas,
    separar_features_target,
)

logger = logging.getLogger(__name__)

#: Iteraciones por defecto de RandomizedSearchCV (trials muestreados).
N_ITER_DEFAULT = 30

#: Folds por defecto de la validación cruzada interna.
CV_DEFAULT = 3

#: Espacio de búsqueda para RandomizedSearchCV (los 9 hiperparámetros del
#: roadmap). El resto de los parámetros hereda `PARAMS_XGBOOST_DEFAULT`.
ESPACIO_HIPERPARAMETROS: dict[str, list[object]] = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 7],
    "gamma": [0.0, 0.1, 0.5],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [0.1, 1.0, 10.0],
}

#: Grid reducido para GridSearchCV (combina los valores más probables para
#: acotar el coste: 2 * 2 * 2 * 1 * 1 * 1 * 1 * 1 * 1 = 8 combinaciones).
GRID_REDUCIDO: dict[str, list[object]] = {
    "n_estimators": [100, 300],
    "max_depth": [3, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "min_child_weight": [3],
    "gamma": [0.0],
    "reg_alpha": [0.0],
    "reg_lambda": [1.0],
}


@dataclass
class ResultadoTuning:
    """Métricas y búsqueda del pipeline de la fase 5.

    - `metricas_default_val`: XGBoost con `PARAMS_XGBOOST_DEFAULT` sobre val
      (referencia del campeón actual antes de tunear).
    - `metricas_tunedo_val` / `metricas_tunedo_test`: mejor candidato de la
      búsqueda (refit sobre train) evaluado sobre val y test.
    - `mejor_puntaje_cv`: RMSE log de la validación cruzada del mejor trial
      (positivo; el scoring interno es `neg_root_mean_squared_error`).
    - `cv_resultados`: tabla de trials ordenada por ranking (columna
      `rank_test_score`, `cv_rmse_log`, `cv_rmse_log_std` y una columna
      `param_<nombre>` por hiperparámetro).
    """

    metricas_default_val: dict[str, float]
    metricas_tunedo_val: dict[str, float]
    metricas_tunedo_test: dict[str, float]
    mejor_params: dict[str, object]
    mejor_puntaje_cv: float
    cv_resultados: pd.DataFrame
    modelo_tunedo: XGBRegressor
    ajustes: Preprocesamiento
    metodo: str
    cv: int
    n_iter: int | None
    n_trials: int


def extraer_cv_resultados(busqueda: GridSearchCV | RandomizedSearchCV) -> pd.DataFrame:
    """Tabla limpia de trials desde `cv_results_` (ranking y RMSE log de CV).

    `mean_test_score` viene negativo por el scoring `neg_root_mean_squared_error`;
    se conserva tal cual (la conversión a RMSE positivo la hace `tunear_xgboost`).
    """

    df = pd.DataFrame(busqueda.cv_results_)

    columnas_param = sorted(columna for columna in df.columns if columna.startswith("param_"))

    return (
        df[["rank_test_score", "mean_test_score", "std_test_score"] + columnas_param]
        .sort_values("rank_test_score")
        .reset_index(drop=True)
    )


def tunear_xgboost(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    metodo: str = "random",
    espacio: dict[str, list[object]] | None = None,
    n_iter: int = N_ITER_DEFAULT,
    cv: int = CV_DEFAULT,
    n_jobs: int = -1,
    random_state: int = 42,
) -> ResultadoTuning:
    """
    Búsqueda de hiperparámetros de XGBoost con CV interna (sin fuga).

    1. Ajusta el preprocesamiento sobre train y lo reaplica a val/test.
    2. Entrena el XGBoost default (`PARAMS_XGBOOST_DEFAULT`) como referencia
       sobre val.
    3. Busca con `GridSearchCV` (exhaustivo) o `RandomizedSearchCV`
       (muestreo de `n_iter` trials) con CV interna sobre train.
    4. Refit del mejor candidato sobre train; evalúa sobre val y test.

    El estimador de la búsqueda parte de `PARAMS_XGBOOST_DEFAULT` (los
    hiperparámetros no incluidos en el espacio heredan el default actual) y
    usa `n_jobs=1` a nivel de XGBoost para no sobrescribir threads cuando el
    propio `GridSearchCV`/`RandomizedSearchCV` paraleliza con `n_jobs`.
    """

    if metodo not in ("grid", "random"):
        raise ValueError(f"método de búsqueda inválido: '{metodo}' (usar 'grid' o 'random')")

    logger.info("=" * 70)
    logger.info("TUNING DE HIPERPARÁMETROS DE XGBOOST")
    logger.info("=" * 70)

    ajustes = ajustar_preprocesamiento(train)

    logger.info(
        "Preprocesamiento ajustado solo sobre train (sin fuga): "
        "%d categóricas con orden ordinal, "
        "%d numéricas con imputación por mediana.",
        len(ajustes.ordenes),
        len(ajustes.imputador),
    )

    train_proc = aplicar_preprocesamiento(train, ajustes)
    val_proc = aplicar_preprocesamiento(val, ajustes)
    test_proc = aplicar_preprocesamiento(test, ajustes)

    x_train, y_train = separar_features_target(train_proc)
    x_val, y_val = separar_features_target(val_proc)
    x_test, y_test = separar_features_target(test_proc)

    logger.info(
        "Split: train %s | val %s | test %s\nFeatures: %d | Target: %s",
        f"{len(x_train):,}",
        f"{len(x_val):,}",
        f"{len(x_test):,}",
        x_train.shape[1],
        TARGET_LOG,
    )

    # Referencia: XGBoost con los parámetros actuales (sin tunear).
    modelo_default = entrenar_xgboost(x_train, y_train, random_state=random_state)
    metricas_default_val = calcular_metricas(y_val, modelo_default.predict(x_val))

    espacio = ESPACIO_HIPERPARAMETROS if espacio is None else espacio

    n_candidatos = int(np.prod([len(valores) for valores in espacio.values()]))
    n_trials = n_candidatos if metodo == "grid" else n_iter

    estimador = XGBRegressor(**PARAMS_XGBOOST_DEFAULT, random_state=random_state, n_jobs=1)

    if metodo == "grid":
        busqueda: GridSearchCV | RandomizedSearchCV = GridSearchCV(
            estimator=estimador,
            param_grid=espacio,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=n_jobs,
            refit=True,
            return_train_score=False,
        )
        descripcion = f"GridSearchCV | {n_candidatos} combinaciones"
    else:
        busqueda = RandomizedSearchCV(
            estimator=estimador,
            param_distributions=espacio,
            n_iter=n_iter,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=n_jobs,
            refit=True,
            return_train_score=False,
            random_state=random_state,
        )
        descripcion = f"RandomizedSearchCV | {n_iter} trials muestreados"

    logger.info("Búsqueda: %s | CV: %d folds | scoring: RMSE log", descripcion, cv)

    busqueda.fit(x_train, y_train)

    mejor_params = dict(busqueda.best_params_)
    mejor_puntaje_cv = -float(busqueda.best_score_)
    modelo_tunedo = busqueda.best_estimator_

    metricas_tunedo_val = calcular_metricas(y_val, modelo_tunedo.predict(x_val))
    metricas_tunedo_test = calcular_metricas(y_test, modelo_tunedo.predict(x_test))

    cv_resultados = extraer_cv_resultados(busqueda)
    cv_resultados["cv_rmse_log"] = -cv_resultados["mean_test_score"]
    cv_resultados["cv_rmse_log_std"] = cv_resultados["std_test_score"]
    cv_resultados = cv_resultados.drop(columns=["mean_test_score", "std_test_score"])

    logger.info("Métricas sobre VALIDACIÓN:")
    mostrar_metricas("default", metricas_default_val)
    mostrar_metricas("tunedo", metricas_tunedo_val)

    logger.info("Mejor params: %s", mejor_params)
    logger.info("Mejor RMSE log de CV: %.4f", mejor_puntaje_cv)

    logger.info("Métricas del mejor candidato sobre TEST:")
    mostrar_metricas("tunedo", metricas_tunedo_test)

    return ResultadoTuning(
        metricas_default_val=metricas_default_val,
        metricas_tunedo_val=metricas_tunedo_val,
        metricas_tunedo_test=metricas_tunedo_test,
        mejor_params=mejor_params,
        mejor_puntaje_cv=mejor_puntaje_cv,
        cv_resultados=cv_resultados,
        modelo_tunedo=modelo_tunedo,
        ajustes=ajustes,
        metodo=metodo,
        cv=cv,
        n_iter=n_iter if metodo == "random" else None,
        n_trials=n_trials,
    )
