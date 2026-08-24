"""
Orquestación de la etapa de Feature Engineering.

Aplica las transformaciones en orden sobre el dataset curado:
1. Selección de columnas (descarte de sin señal).
2. Target logarítmico y filtro de precios inválidos.
3. Codificación ordinal de categóricas (ajustada sobre los datos recibidos).
4. Imputación por mediana de las numéricas con faltantes.

`construir_features` es la función de alto nivel usada por
`scripts/features.py` y por el notebook 03.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from real_estate.features.transformations import (
    COLUMNAS_CATEGORICAS,
    aplicar_imputacion,
    codificar_ordinal,
    crear_imputador,
    crear_orden_mediana,
    crear_target_log,
    seleccionar_columnas,
)

logger = logging.getLogger(__name__)


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la matriz de features a partir del dataset curado.

    Ajusta el orden ordinal y las medianas de imputación sobre el propio
    `df` recibido. Para val/test se debe reutilizar el orden y el imputador
    ajustados sobre el entrenamiento (ver `dividir_train_val_test` y el
    notebook 03).
    """

    df = df.copy()

    logger.info("=" * 70)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 70)

    logger.info("Filas iniciales: %s | Columnas: %d", f"{df.shape[0]:,}", df.shape[1])

    # 1. Selección de columnas
    df = seleccionar_columnas(df)

    logger.info("Tras seleccionar columnas: %d columnas", df.shape[1])

    # 2. Target logarítmico (filtra precios inválidos)
    df = crear_target_log(df)

    # 3. Codificación ordinal de categóricas por mediana de precio
    for columna in COLUMNAS_CATEGORICAS:
        if columna in df.columns:
            orden = crear_orden_mediana(df, columna)
            df = codificar_ordinal(df, columna, orden)
            logger.info("%-30s -> %d categorías codificadas", columna, len(orden))

    # 4. Imputación por mediana
    imputador = crear_imputador(df)

    df = aplicar_imputacion(df, imputador)

    logger.info("Imputadas por mediana: %s", sorted(imputador))

    logger.info("Filas finales: %s | Columnas: %d", f"{df.shape[0]:,}", df.shape[1])
    logger.info("Faltantes restantes: %d", int(df.isna().sum().sum()))

    return df


def dividir_train_val_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide la matriz de features en train / val / test.

    Proporciones por defecto: 80 % train, 10 % val, 10 % test
    (el 20 % restante se parte a la mitad). El split es reproducible
    mediante `random_state`.
    """

    train, temporal = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    val, test = train_test_split(
        temporal,
        test_size=1 - val_size,
        random_state=random_state,
    )

    return train, val, test


def mostrar_features(df: pd.DataFrame) -> None:
    """Muestra un resumen de la matriz de features construida."""

    logger.info("=" * 70)
    logger.info("MATRIZ DE FEATURES")
    logger.info("=" * 70)

    logger.info("Filas: %s | Columnas: %d", f"{df.shape[0]:,}", df.shape[1])

    logger.info("Primeras filas:\n%s", df.head().to_string())
    logger.info("Tipos de datos:\n%s", df.dtypes)
    logger.info("Valores faltantes:\n%s", df.isna().sum())
