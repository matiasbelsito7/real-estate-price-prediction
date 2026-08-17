"""
Tracking de experimentos con MLflow (Fase 6).

Registra parámetros, métricas, un artefacto JSON de resumen y el modelo
XGBoost (con firma) en un experimento de MLflow, y versiona el modelo en el
Model Registry. Para los modelos lineales (fase 4) registra una corrida por
modelo (Lasso y Ridge) sin versionar nada en el Model Registry. Para el
tuning de XGBoost (fase 5) registra un run resumen más un run anidado por
trial, sin versionar nada (el champion se elige en la fase 6).

API pública:

1. `configurar_tracking` — define el tracking URI y el experimento.
2. `registrar_resultado` — abre una corrida, registra params/métricas/
   artefactos, loguea el modelo y lo versiona. Devuelve `(run_id, version)`.
3. `registrar_lineales` — una corrida por modelo lineal con params, métricas
   (val y, para el mejor, test) y pipeline con firma. Devuelve
   `[(nombre_modelo, run_id), ...]`.
4. `registrar_tuning` — un run resumen (métricas default vs tunedo, mejor
   modelo con firma y artefactos) más un run anidado por trial de la
   búsqueda. Devuelve `(run_id_resumen, [(trial, run_id), ...])`.
5. `finalizar_corrida` — cierra la corrida activa si la hubiera.

El módulo de modelos (`real_estate.models.entrenamiento`) se mantiene puro:
el tracking se inyecta desde acá sin acoplarlo al entrenamiento.
"""

from __future__ import annotations

import os

import mlflow
import pandas as pd

from real_estate.models.entrenamiento import (
    ResultadoEntrenamiento,
    aplicar_preprocesamiento,
    separar_features_target,
)
from real_estate.models.modelos_lineales import ResultadoLineales
from real_estate.models.tuning import ResultadoTuning

#: Nombre del experimento por defecto (agrupa todas las corridas del proyecto).
EXPERIMENTO_DEFAULT = "prediccion_precios_propiedades"

#: Nombre con el que se registra/versiona el modelo en el Model Registry.
MODELO_DEFAULT = "modelo_precio_propiedades"

#: Tracking URI local por defecto (store de archivos en el repo, gitignored).
TRACKING_URI_DEFAULT = "mlruns"


def configurar_tracking(
    tracking_uri: str | None = None,
    experimento: str = EXPERIMENTO_DEFAULT,
) -> str:
    """
    Configura el tracking URI y el experimento (creándolo si no existe).

    El URI se resuelve en este orden: argumento explícito, variable de
    entorno `MLFLOW_TRACKING_URI`, o el store local por defecto (`mlruns/`).
    Devuelve el nombre del experimento configurado.

    Nota: el store de archivos de MLflow 3.x requiere el opt-out
    `MLFLOW_ALLOW_FILE_STORE`; se habilita por defecto para que el store
    local funcione sin configuración adicional (no afecta a otros backends).
    """

    # Permite el store de archivos local (default) sin servidor.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or TRACKING_URI_DEFAULT

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experimento)

    return experimento


def _metricas_prefijadas(prefijo: str, metricas: dict[str, float]) -> dict[str, float]:
    """Prefija las métricas para evitar colisiones de nombre en MLflow."""

    return {f"{prefijo}_{clave}": valor for clave, valor in metricas.items()}


def registrar_resultado(
    resultado: ResultadoEntrenamiento,
    train: pd.DataFrame,
    random_state: int = 42,
    dataset_info: str | None = None,
    split_sizes: dict[str, int] | None = None,
) -> tuple[str, str]:
    """
    Registra en MLflow la corrida completa y devuelve `(run_id, version)`.

    Params: tipo de modelo, semilla, tamaño del split y todos los hiper-
    parámetros del XGBoost (`get_params()`).

    Métricas: RMSE log / RMSE USD / R² del baseline y de XGBoost sobre val,
    y de XGBoost sobre test (prefijadas para no colisionar).

    Artefactos: un JSON de resumen (`resumen_entrenamiento.json`), la
    importancia de features del XGBoost (`feature_importances.json`, ordenada
    de mayor a menor peso) y el modelo XGBoost logueado con su firma
    (`infer_signature`), que se registra en el Model Registry para versionarlo.
    """

    with mlflow.start_run() as corrida:
        run_id = str(corrida.info.run_id)

        # Reconstruye X_train / y_train aplicando el preprocesamiento que
        # aprendió el pipeline (sin duplicar lógica ni tocar el módulo puro).
        train_proc = aplicar_preprocesamiento(train, resultado.ajustes)
        x_train, y_train = separar_features_target(train_proc)

        # ---- Params -------------------------------------------------------
        parametros: dict[str, object] = {
            "tipo_modelo": "xgboost",
            "random_state": str(random_state),
            "n_features": str(x_train.shape[1]),
            "n_train": str(x_train.shape[0]),
        }
        if dataset_info is not None:
            parametros["dataset_info"] = dataset_info
        if split_sizes is not None:
            for nombre, tamano in split_sizes.items():
                parametros[f"n_{nombre}"] = str(tamano)
        parametros.update(
            {
                f"xgboost_{clave}": str(valor)
                for clave, valor in resultado.modelo_xgboost.get_params().items()
            }
        )
        mlflow.log_params(parametros)

        # ---- Métricas -----------------------------------------------------
        mlflow.log_metrics(_metricas_prefijadas("baseline_val", resultado.metricas_baseline_val))
        mlflow.log_metrics(_metricas_prefijadas("xgboost_val", resultado.metricas_xgboost_val))
        mlflow.log_metrics(_metricas_prefijadas("xgboost_test", resultado.metricas_xgboost_test))

        # ---- Artefacto de resumen -----------------------------------------
        resumen = {
            "tipo_modelo": "xgboost",
            "random_state": random_state,
            "dataset_info": dataset_info,
            "metricas_baseline_val": resultado.metricas_baseline_val,
            "metricas_xgboost_val": resultado.metricas_xgboost_val,
            "metricas_xgboost_test": resultado.metricas_xgboost_test,
            "parametros_xgboost": resultado.modelo_xgboost.get_params(),
        }
        mlflow.log_dict(resumen, "resumen_entrenamiento.json")

        # ---- Importancia de features (Fase 6) -----------------------------
        # Mapea `feature_importances_` (normalizadas, suman 1) a los nombres
        # de las features del pipeline, ordenadas de mayor a menor peso.
        importancia = {
            str(columna): float(valor)
            for columna, valor in zip(
                x_train.columns,
                resultado.modelo_xgboost.feature_importances_,
                strict=True,
            )
        }
        importancia = dict(sorted(importancia.items(), key=lambda item: item[1], reverse=True))
        mlflow.log_dict(importancia, "feature_importances.json")

        # ---- Modelo con firma + versionado --------------------------------
        firma = mlflow.models.infer_signature(x_train, y_train)
        model_info = mlflow.xgboost.log_model(
            xgb_model=resultado.modelo_xgboost,
            name="modelo",
            signature=firma,
        )
        model_uri = getattr(model_info, "model_uri", f"runs:/{run_id}/modelo")
        version = mlflow.register_model(model_uri, MODELO_DEFAULT)

        return run_id, str(version.version)


def registrar_lineales(
    resultado: ResultadoLineales,
    train: pd.DataFrame,
    random_state: int = 42,
    dataset_info: str | None = None,
    split_sizes: dict[str, int] | None = None,
) -> list[tuple[str, str]]:
    """
    Registra en MLflow una corrida por modelo lineal y devuelve `[(nombre, run_id)]`.

    Cada corrida loguea sus parámetros (tipo, alpha, semilla, tamaño del
    split), sus métricas sobre val (`val_*`) y, para el mejor modelo, las de
    test (`test_*`), además de un JSON de resumen y el pipeline logueado con
    su firma (`infer_signature`).

    Los modelos lineales no se versionan en el Model Registry: el champion
    se elige y registra recién en la fase 6.
    """

    train_proc = aplicar_preprocesamiento(train, resultado.ajustes)
    x_train, y_train = separar_features_target(train_proc)
    firma = mlflow.models.infer_signature(x_train, y_train)

    runs: list[tuple[str, str]] = []

    for nombre, modelo, metricas_val in (
        ("lasso", resultado.modelo_lasso, resultado.metricas_lasso_val),
        ("ridge", resultado.modelo_ridge, resultado.metricas_ridge_val),
    ):
        with mlflow.start_run() as corrida:
            run_id = str(corrida.info.run_id)

            alpha = float(modelo.named_steps["modelo"].alpha)

            # ---- Params ---------------------------------------------------
            parametros: dict[str, object] = {
                "tipo_modelo": nombre,
                "alpha": str(alpha),
                "random_state": str(random_state),
                "n_features": str(x_train.shape[1]),
                "n_train": str(x_train.shape[0]),
            }
            if dataset_info is not None:
                parametros["dataset_info"] = dataset_info
            if split_sizes is not None:
                for nombre_split, tamano in split_sizes.items():
                    parametros[f"n_{nombre_split}"] = str(tamano)
            mlflow.log_params(parametros)

            # ---- Métricas -------------------------------------------------
            mlflow.log_metrics(_metricas_prefijadas("val", metricas_val))

            es_mejor = nombre == resultado.mejor
            if es_mejor:
                mlflow.log_metrics(_metricas_prefijadas("test", resultado.metricas_mejor_test))

            # ---- Artefacto de resumen -------------------------------------
            resumen = {
                "tipo_modelo": nombre,
                "alpha": alpha,
                "random_state": random_state,
                "dataset_info": dataset_info,
                "metricas_val": metricas_val,
                "metricas_test": resultado.metricas_mejor_test if es_mejor else None,
                "mejor_modelo": resultado.mejor,
            }
            mlflow.log_dict(resumen, "resumen_lineal.json")

            # ---- Modelo con firma (sin Model Registry) ---------------------
            mlflow.sklearn.log_model(
                sk_model=modelo,
                artifact_path="modelo",
                signature=firma,
            )

            runs.append((nombre, run_id))

    return runs


def registrar_tuning(
    resultado: ResultadoTuning,
    train: pd.DataFrame,
    random_state: int = 42,
    dataset_info: str | None = None,
    split_sizes: dict[str, int] | None = None,
) -> tuple[str, list[tuple[int, str]]]:
    """
    Registra en MLflow la búsqueda de hiperparámetros y devuelve
    `(run_id_resumen, [(trial, run_id_trial), ...])`.

    El run resumen loguea los parámetros de la búsqueda (método, folds,
    trials, mejor configuración), las métricas del XGBoost default vs el
    tunedo (val y test), el mejor modelo con firma y tres artefactos:
    `resumen_tuning.json`, `mejor_params.json` y `cv_resultados.json` (la
    tabla de trials vía `log_table`).

    Además crea un run anidado por trial (un run de MLflow por trial, según
    el roadmap) con los parámetros del trial y sus métricas de CV (RMSE log,
    desviación y ranking).

    No se versiona nada en el Model Registry: el champion se elige y registra
    recién en la fase 6.
    """

    train_proc = aplicar_preprocesamiento(train, resultado.ajustes)
    x_train, y_train = separar_features_target(train_proc)
    firma = mlflow.models.infer_signature(x_train, y_train)

    with mlflow.start_run() as corrida:
        run_id_resumen = str(corrida.info.run_id)

        # ---- Params -------------------------------------------------------
        parametros: dict[str, object] = {
            "tipo_modelo": "xgboost",
            "metodo_tuning": resultado.metodo,
            "cv_folds": str(resultado.cv),
            "n_trials": str(resultado.n_trials),
            "random_state": str(random_state),
            "n_features": str(x_train.shape[1]),
            "n_train": str(x_train.shape[0]),
        }
        if resultado.n_iter is not None:
            parametros["n_iter"] = str(resultado.n_iter)
        if dataset_info is not None:
            parametros["dataset_info"] = dataset_info
        if split_sizes is not None:
            for nombre, tamano in split_sizes.items():
                parametros[f"n_{nombre}"] = str(tamano)
        parametros.update(
            {f"tuned_{clave}": str(valor) for clave, valor in resultado.mejor_params.items()}
        )
        mlflow.log_params(parametros)

        # ---- Métricas -----------------------------------------------------
        mlflow.log_metrics(_metricas_prefijadas("default_val", resultado.metricas_default_val))
        mlflow.log_metrics(_metricas_prefijadas("tunedo_val", resultado.metricas_tunedo_val))
        mlflow.log_metrics(_metricas_prefijadas("tunedo_test", resultado.metricas_tunedo_test))
        mlflow.log_metric("mejor_puntaje_cv_rmse_log", resultado.mejor_puntaje_cv)

        # ---- Artefactos ---------------------------------------------------
        resumen = {
            "metodo": resultado.metodo,
            "cv_folds": resultado.cv,
            "n_iter": resultado.n_iter,
            "n_trials": resultado.n_trials,
            "mejor_params": resultado.mejor_params,
            "mejor_puntaje_cv_rmse_log": resultado.mejor_puntaje_cv,
            "metricas_default_val": resultado.metricas_default_val,
            "metricas_tunedo_val": resultado.metricas_tunedo_val,
            "metricas_tunedo_test": resultado.metricas_tunedo_test,
        }
        mlflow.log_dict(resumen, "resumen_tuning.json")
        mlflow.log_dict({"mejor_params": resultado.mejor_params}, "mejor_params.json")
        mlflow.log_table(resultado.cv_resultados, artifact_file="cv_resultados.json")

        # ---- Mejor modelo con firma (sin Model Registry) ------------------
        mlflow.xgboost.log_model(
            xgb_model=resultado.modelo_tunedo,
            name="modelo_tunedo",
            signature=firma,
        )

        # ---- Un run anidado por trial -------------------------------------
        runs_trials: list[tuple[int, str]] = []
        # `enumerate` da un índice entero (los trials se ordenan por ranking);
        # `iterrows` por sí solo expondría el índice como `Hashable` para mypy.
        for indice, (_, fila) in enumerate(resultado.cv_resultados.iterrows()):
            with mlflow.start_run(nested=True) as trial:
                trial_id = str(trial.info.run_id)

                params_trial = {
                    clave.removeprefix("param_"): valor
                    for clave, valor in fila.items()
                    if isinstance(clave, str) and clave.startswith("param_")
                }
                mlflow.log_params({clave: str(valor) for clave, valor in params_trial.items()})
                mlflow.log_metric("cv_rmse_log", float(fila["cv_rmse_log"]))
                mlflow.log_metric("cv_rmse_log_std", float(fila["cv_rmse_log_std"]))
                mlflow.log_metric("cv_rank", int(fila["rank_test_score"]))

                runs_trials.append((indice, trial_id))

    return run_id_resumen, runs_trials


def finalizar_corrida() -> None:
    """Cierra la corrida activa si la hubiera (no-op si no hay ninguna)."""

    if mlflow.active_run() is not None:
        mlflow.end_run()
