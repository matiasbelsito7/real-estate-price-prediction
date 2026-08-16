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


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la matriz de features a partir del dataset curado.

    Ajusta el orden ordinal y las medianas de imputación sobre el propio
    `df` recibido. Para val/test se debe reutilizar el orden y el imputador
    ajustados sobre el entrenamiento (ver `dividir_train_val_test` y el
    notebook 03).
    """

    df = df.copy()

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    print(f"\nFilas iniciales: {df.shape[0]:,} | Columnas: {df.shape[1]}")

    # 1. Selección de columnas
    df = seleccionar_columnas(df)

    print(f"Tras seleccionar columnas: {df.shape[1]} columnas")

    # 2. Target logarítmico (filtra precios inválidos)
    df = crear_target_log(df)

    # 3. Codificación ordinal de categóricas por mediana de precio
    for columna in COLUMNAS_CATEGORICAS:
        if columna in df.columns:
            orden = crear_orden_mediana(df, columna)
            df = codificar_ordinal(df, columna, orden)
            print(f"{columna:30s} -> {len(orden)} categorías codificadas")

    # 4. Imputación por mediana
    imputador = crear_imputador(df)

    df = aplicar_imputacion(df, imputador)

    print(f"Imputadas por mediana: {sorted(imputador)}")

    print(f"\nFilas finales: {df.shape[0]:,} | Columnas: {df.shape[1]}")
    print(f"Faltantes restantes: {int(df.isna().sum().sum())}")

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

    print("\n" + "=" * 70)
    print("MATRIZ DE FEATURES")
    print("=" * 70)

    print(f"\nFilas: {df.shape[0]:,} | Columnas: {df.shape[1]}")

    print("\nPrimeras filas:")
    print(df.head().to_string())

    print("\nTipos de datos:")
    print(df.dtypes)

    print("\nValores faltantes:")
    print(df.isna().sum())
