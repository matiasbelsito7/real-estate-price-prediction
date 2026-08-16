"""
Tracking de experimentos con MLflow (Fase 6).

Registra parámetros, métricas, un artefacto JSON de resumen y el modelo
XGBoost (con firma) en un experimento de MLflow, y versiona el modelo en el
Model Registry.

API pública:

1. `configurar_tracking` — define el tracking URI y el experimento.
2. `registrar_resultado` — abre una corrida, registra params/métricas/
   artefactos, loguea el modelo y lo versiona. Devuelve `(run_id, version)`.
3. `finalizar_corrida` — cierra la corrida activa si la hubiera.

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

    Artefactos: un JSON de resumen (`resumen_entrenamiento.json`) y el modelo
    XGBoost logueado con su firma (`infer_signature`), que se registra en el
    Model Registry para versionarlo.
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


def finalizar_corrida() -> None:
    """Cierra la corrida activa si la hubiera (no-op si no hay ninguna)."""

    if mlflow.active_run() is not None:
        mlflow.end_run()
