"""
Orquestación de la etapa de Data Curation.

Aplica las etapas en orden sobre un DataFrame:
0. Visualización del dataset crudo.
1. Limpieza de tipos (cleaning).
2. Normalización de moneda a USD (transformations).
3. Manejo de valores faltantes (transformations).
4. Validación de coherencia (validation).
5. Exportación del dataset curado.

`curar_csv` es la función de alto nivel usada por `scripts/curate.py`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from real_estate.curation.cleaning import (
    limpiar_columnas_numericas,
    limpiar_expensas,
    preparar_fecha,
)
from real_estate.curation.transformations import (
    crear_indicadores_missing,
    normalizar_expensas,
    normalizar_moneda,
)
from real_estate.curation.validation import validar


def mostrar_dataset(df: pd.DataFrame) -> None:
    """Muestra información general del dataset original."""

    print("\n" + "=" * 70)
    print("DATASET ORIGINAL")
    print("=" * 70)

    print(f"\nFilas:    {df.shape[0]:,}")
    print(f"Columnas: {df.shape[1]}")

    print("\nColumnas:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nPrimeras filas:")
    print(df.head())

    print("\nTipos de datos:")
    print(df.dtypes)

    print("\nValores faltantes:")
    missing = df.isna().sum()
    missing_percentage = df.isna().mean() * 100

    missing_table = pd.DataFrame(
        {
            "missing": missing,
            "percentage": missing_percentage.round(2),
        }
    )

    print(missing_table)


def curar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica las etapas de curación sobre el DataFrame.

    No modifica el DataFrame recibido: opera (y devuelve) una copia.
    """

    df = df.copy()

    # 1. Limpieza de tipos
    df = preparar_fecha(df)
    df = limpiar_columnas_numericas(df)
    df = limpiar_expensas(df)

    # 2. Normalizar moneda
    df = normalizar_moneda(df)
    df = normalizar_expensas(df)

    # 3. Missing values
    df = crear_indicadores_missing(df)

    # 4. Validación (reporte, no modifica datos)
    validar(df)

    return df


def mostrar_dataset_curado(df: pd.DataFrame) -> None:
    """Muestra información general del dataset curado."""

    print("\n" + "=" * 70)
    print("DATASET CURADO")
    print("=" * 70)

    print(f"\nFilas:    {df.shape[0]:,}")
    print(f"Columnas: {df.shape[1]}")

    print("\nPrimeras filas:")
    print(df.head().to_string())

    print("\nTipos de datos:")
    print(df.dtypes)

    print("\nValores faltantes:")
    missing = pd.DataFrame(
        {
            "missing": df.isna().sum(),
            "percentage": (df.isna().mean() * 100).round(2),
        }
    )

    print(missing)


def curar_csv(input_file: str | Path, output_file: str | Path) -> None:
    """Lee el CSV crudo, lo cura y guarda el resultado."""

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

    print(f"Cargando dataset: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)

    mostrar_dataset(df)

    df = curar_dataset(df)

    mostrar_dataset_curado(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("CURADO FINALIZADO")
    print("=" * 70)

    print(f"\nDataset curado guardado en:\n{output_path}")
