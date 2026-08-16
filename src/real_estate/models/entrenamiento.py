"""
Entrenamiento y evaluación de modelos (Fase 5).

Pipeline train / val / test sin fuga de información:

1. `ajustar_preprocesamiento` aprende **solo sobre train** los parámetros:
   el orden ordinal de las categóricas (mediana de precio) y la mediana de
   imputación de las numéricas.
2. `aplicar_preprocesamiento` reaplica ese ajuste sobre cualquier split
   (train, val o test) con la API `crear_*` / `aplicar_*` de `features`.
3. `entrenar_y_evaluar` entrena el baseline (mediana) y XGBoost sobre train,
   compara sobre val y evalúa en test al mejor modelo.

Métricas sobre el target logarítmico `log_precio_usd`: RMSE en log (equivale
a error relativo), RMSE en USD (deshaciendo el log) y R². El tracking con
MLflow queda para la fase 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from xgboost import XGBRegressor

from real_estate.features.transformations import (
    COLUMNAS_CATEGORICAS,
    TARGET_LOG,
    TARGET_PRECIO,
    aplicar_imputacion,
    codificar_ordinal,
    crear_imputador,
    crear_orden_mediana,
)

# Parámetros por defecto de XGBoost (razonables para datos tabulares con
# ~1.600 filas de train; se pueden sobrescribir vía `entrenar_xgboost`).
PARAMS_XGBOOST_DEFAULT: dict[str, object] = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_lambda": 1.0,
}


@dataclass(frozen=True)
class Preprocesamiento:
    """Parámetros aprendidos sobre train: ordenes ordinales e imputador.

    - `ordenes`: categoría -> ranking por mediana de precio (una entrada por
      columna categórica presente en el train).
    - `imputador`: columna -> mediana de imputación (sin columnas sin
      valores válidos).
    """

    ordenes: dict[str, list[str]]
    imputador: dict[str, float]


@dataclass
class ResultadoEntrenamiento:
    """Métricas y modelos del pipeline de la fase 5."""

    metricas_baseline_val: dict[str, float]
    metricas_xgboost_val: dict[str, float]
    metricas_xgboost_test: dict[str, float]
    modelo_baseline: DummyRegressor
    modelo_xgboost: XGBRegressor
    ajustes: Preprocesamiento


def ajustar_preprocesamiento(train: pd.DataFrame) -> Preprocesamiento:
    """
    Ajusta sobre train los parámetros sin fuga: ordenes ordinales e imputador.

    Solo las columnas presentes en `train` generan ajuste (p. ej. si el
    dataset ya viene codificado, no hay categóricas que re-ajustar).
    """

    ordenes = {
        columna: crear_orden_mediana(train, columna)
        for columna in COLUMNAS_CATEGORICAS
        if columna in train.columns
    }

    imputador = crear_imputador(train)

    return Preprocesamiento(ordenes=ordenes, imputador=imputador)


def aplicar_preprocesamiento(
    df: pd.DataFrame,
    ajustes: Preprocesamiento,
) -> pd.DataFrame:
    """
    Reaplica sobre `df` (train, val o test) lo aprendido en train.

    Las categóricas se reemplazan por `{columna}_ordinal` (categorías nuevas
    o NaN -> `CODIGO_DESCONOCIDO`) y las numéricas se rellenan con la mediana
    ajustada. No modifica el DataFrame recibido.
    """

    df = df.copy()

    for columna, orden in ajustes.ordenes.items():
        if columna in df.columns:
            df = codificar_ordinal(df, columna, orden)

    df = aplicar_imputacion(df, ajustes.imputador)

    return df


def separar_features_target(
    df: pd.DataFrame,
    target: str = TARGET_LOG,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa las features (todo excepto `precio_usd` y `log_precio_usd`) del target.

    El target por defecto es `log_precio_usd`; `precio_usd` se excluye de las
    features para no filtrar el valor objetivo (solo se usa para reportar el
    RMSE en USD).
    """

    columnas_features = [
        columna for columna in df.columns if columna not in (TARGET_PRECIO, TARGET_LOG)
    ]

    return df[columnas_features], df[target]


def calcular_metricas(
    y_real: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """
    Métricas sobre el target logarítmico: RMSE en log, RMSE en USD y R².

    El RMSE en log se interpreta como error relativo (p. ej. 0.30 en log
    ~ 30 %); el RMSE en USD deshace el log para dar la magnitud absoluta.
    """

    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "rmse_log": float(root_mean_squared_error(y_real, y_pred)),
        "rmse_usd": float(root_mean_squared_error(np.exp(y_real), np.exp(y_pred))),
        "r2": float(r2_score(y_real, y_pred)),
    }


def mostrar_metricas(nombre: str, metricas: dict[str, float]) -> None:
    """Imprime las métricas con formato (RMSE log, RMSE USD y R²)."""

    print(
        f"  {nombre:12s} | RMSE log: {metricas['rmse_log']:.4f} "
        f"| RMSE USD: {metricas['rmse_usd']:>12,.0f} "
        f"| R²: {metricas['r2']:.4f}"
    )


def entrenar_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> DummyRegressor:
    """
    Baseline: predice siempre la mediana de `y_train`.

    Sirve de referencia mínima: si XGBoost no supera a la mediana, el
    problema no tiene señal aprovechable.
    """

    modelo = DummyRegressor(strategy="median")

    modelo.fit(X_train, y_train)

    return modelo


def entrenar_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, object] | None = None,
    random_state: int = 42,
) -> XGBRegressor:
    """Entrena un XGBRegressor con los parámetros dados (o los por defecto)."""

    params = PARAMS_XGBOOST_DEFAULT if params is None else params

    modelo = XGBRegressor(**params, random_state=random_state, n_jobs=-1)

    modelo.fit(X_train, y_train)

    return modelo


def entrenar_y_evaluar(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    params_xgb: dict[str, object] | None = None,
    random_state: int = 42,
) -> ResultadoEntrenamiento:
    """
    Pipeline completo de la fase 5 (train / val / test, sin fuga).

    1. Ajusta el preprocesamiento sobre train.
    2. Lo reaplica a train, val y test por separado.
    3. Separa features del target en cada split.
    4. Entrena baseline (mediana) y XGBoost sobre train.
    5. Evalúa ambos sobre val; evalúa XGBoost en test (modelo final).

    Devuelve `ResultadoEntrenamiento` con las métricas y los modelos
    ajustados (para análisis posterior, p. ej. SHAP en la fase 7).
    """

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO Y EVALUACIÓN")
    print("=" * 70)

    ajustes = ajustar_preprocesamiento(train)

    print(
        "\nPreprocesamiento ajustado solo sobre train (sin fuga): "
        f"{len(ajustes.ordenes)} categóricas con orden ordinal, "
        f"{len(ajustes.imputador)} numéricas con imputación por mediana."
    )

    train_proc = aplicar_preprocesamiento(train, ajustes)
    val_proc = aplicar_preprocesamiento(val, ajustes)
    test_proc = aplicar_preprocesamiento(test, ajustes)

    X_train, y_train = separar_features_target(train_proc)
    X_val, y_val = separar_features_target(val_proc)
    X_test, y_test = separar_features_target(test_proc)

    print(
        f"\nSplit: train {len(X_train):,} | val {len(X_val):,} | test {len(X_test):,}"
        f"\nFeatures: {X_train.shape[1]} | Target: {TARGET_LOG}"
    )

    modelo_baseline = entrenar_baseline(X_train, y_train)
    modelo_xgboost = entrenar_xgboost(
        X_train,
        y_train,
        params=params_xgb,
        random_state=random_state,
    )

    metricas_baseline_val = calcular_metricas(y_val, modelo_baseline.predict(X_val))
    metricas_xgboost_val = calcular_metricas(y_val, modelo_xgboost.predict(X_val))
    metricas_xgboost_test = calcular_metricas(y_test, modelo_xgboost.predict(X_test))

    print("\nMétricas sobre VALIDACIÓN:")
    mostrar_metricas("baseline", metricas_baseline_val)
    mostrar_metricas("xgboost", metricas_xgboost_val)

    mejor = "xgboost" if metricas_xgboost_val["rmse_log"] < metricas_baseline_val["rmse_log"] else "baseline"

    print(f"\nMejor modelo en val: {mejor}")

    if mejor == "xgboost":
        print("\nMétricas de XGBoost sobre TEST (modelo final):")
        mostrar_metricas("xgboost", metricas_xgboost_test)

    return ResultadoEntrenamiento(
        metricas_baseline_val=metricas_baseline_val,
        metricas_xgboost_val=metricas_xgboost_val,
        metricas_xgboost_test=metricas_xgboost_test,
        modelo_baseline=modelo_baseline,
        modelo_xgboost=modelo_xgboost,
        ajustes=ajustes,
    )
