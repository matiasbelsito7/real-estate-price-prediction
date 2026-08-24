"""
Modelos lineales: Lasso y Ridge (Fase 4 del roadmap).

Compara regresores lineales contra XGBoost con el mismo pipeline y la misma
validación (train / val / test sin fuga):

1. Mismo preprocesamiento que XGBoost (`ajustar_preprocesamiento` aprendido
   solo sobre train, reaplicado a val y test).
2. Mismas features y mismo target logarítmico `log_precio_usd`.
3. Escalado de las features numéricas (los lineales son sensibles a la
   escala; los árboles no): `StandardScaler` dentro del pipeline, ajustado
   solo sobre train para no filtrar información de val/test.

`entrenar_y_evaluar_lineales` entrena Lasso y Ridge sobre train, compara
sobre val y evalúa en test al mejor de ambos. El tracking con MLflow queda
en `real_estate.tracking.experimentos.registrar_lineales` (una corrida por
modelo, sin versionar en el Model Registry; el champion se elige en la
fase 6).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypeAlias

import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from real_estate.features.transformations import TARGET_LOG
from real_estate.models.entrenamiento import (
    Preprocesamiento,
    ajustar_preprocesamiento,
    aplicar_preprocesamiento,
    calcular_metricas,
    mostrar_metricas,
    separar_features_target,
)

logger = logging.getLogger(__name__)

#: Regularización por defecto de Lasso y Ridge.
ALPHA_DEFAULT = 1.0

#: Iteraciones máximas para garantizar la convergencia del solucionador.
MAX_ITER_DEFAULT = 10_000

#: Pipeline escalado + regresor lineal (la escala se ajusta solo sobre train).
#: sklearn 1.9 no expone anotaciones para `Pipeline` (mypy lo ve como `Any`);
#: el alias mantiene la semántica del tipo sin tipado fino sobre los pasos.
PipelineLineal: TypeAlias = Pipeline


@dataclass
class ResultadoLineales:
    """Métricas y modelos del pipeline de la fase 4 (Lasso y Ridge)."""

    metricas_lasso_val: dict[str, float]
    metricas_ridge_val: dict[str, float]
    metricas_mejor_test: dict[str, float]
    modelo_lasso: PipelineLineal
    modelo_ridge: PipelineLineal
    ajustes: Preprocesamiento
    mejor: str


def crear_pipeline_lineal(estimador: Lasso | Ridge) -> PipelineLineal:
    """Pipeline con escalado estándar + regresor lineal.

    El `StandardScaler` se ajusta junto con el modelo sobre el train (el
    pipeline completo hace `fit` de una vez), por lo que no hay fuga hacia
    val/test.
    """

    return Pipeline([("scaler", StandardScaler()), ("modelo", estimador)])


def entrenar_lasso(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = ALPHA_DEFAULT,
    random_state: int = 42,
) -> PipelineLineal:
    """Entrena un Lasso escalado sobre train y devuelve el pipeline ajustado."""

    modelo = crear_pipeline_lineal(
        Lasso(alpha=alpha, max_iter=MAX_ITER_DEFAULT, random_state=random_state)
    )

    modelo.fit(x_train, y_train)

    return modelo


def entrenar_ridge(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = ALPHA_DEFAULT,
    random_state: int = 42,
) -> PipelineLineal:
    """Entrena un Ridge escalado sobre train y devuelve el pipeline ajustado."""

    modelo = crear_pipeline_lineal(
        Ridge(alpha=alpha, max_iter=MAX_ITER_DEFAULT, random_state=random_state)
    )

    modelo.fit(x_train, y_train)

    return modelo


def entrenar_y_evaluar_lineales(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    alpha_lasso: float = ALPHA_DEFAULT,
    alpha_ridge: float = ALPHA_DEFAULT,
    random_state: int = 42,
) -> ResultadoLineales:
    """
    Pipeline completo de la fase 4 (train / val / test, sin fuga).

    1. Ajusta el preprocesamiento sobre train (mismas features que XGBoost).
    2. Lo reaplica a train, val y test por separado.
    3. Separa features del target en cada split.
    4. Entrena Lasso y Ridge sobre train (con escalado dentro del pipeline).
    5. Evalúa ambos sobre val; evalúa en test al mejor de ambos.

    Devuelve `ResultadoLineales` con las métricas y los pipelines ajustados.
    """

    logger.info("=" * 70)
    logger.info("MODELOS LINEALES (Lasso y Ridge)")
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
        "Split: train %s | val %s | test %s\nFeatures: %d | Target: %s | Escalado: StandardScaler",
        f"{len(x_train):,}",
        f"{len(x_val):,}",
        f"{len(x_test):,}",
        x_train.shape[1],
        TARGET_LOG,
    )

    modelo_lasso = entrenar_lasso(
        x_train,
        y_train,
        alpha=alpha_lasso,
        random_state=random_state,
    )
    modelo_ridge = entrenar_ridge(
        x_train,
        y_train,
        alpha=alpha_ridge,
        random_state=random_state,
    )

    metricas_lasso_val = calcular_metricas(y_val, modelo_lasso.predict(x_val))
    metricas_ridge_val = calcular_metricas(y_val, modelo_ridge.predict(x_val))

    logger.info("Métricas sobre VALIDACIÓN:")
    mostrar_metricas("lasso", metricas_lasso_val)
    mostrar_metricas("ridge", metricas_ridge_val)

    mejor = "lasso" if metricas_lasso_val["rmse_log"] <= metricas_ridge_val["rmse_log"] else "ridge"
    modelo_mejor = modelo_lasso if mejor == "lasso" else modelo_ridge
    metricas_mejor_test = calcular_metricas(y_test, modelo_mejor.predict(x_test))

    logger.info("Mejor modelo en val: %s", mejor)
    logger.info("Métricas sobre TEST (modelo final):")
    mostrar_metricas(mejor, metricas_mejor_test)

    return ResultadoLineales(
        metricas_lasso_val=metricas_lasso_val,
        metricas_ridge_val=metricas_ridge_val,
        metricas_mejor_test=metricas_mejor_test,
        modelo_lasso=modelo_lasso,
        modelo_ridge=modelo_ridge,
        ajustes=ajustes,
        mejor=mejor,
    )
