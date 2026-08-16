#!/usr/bin/env python3
"""
Entry point de la etapa de Feature Engineering.

Uso:
    python scripts/features.py
    python scripts/features.py --input data/processed/propiedades_argenprop_curado.csv --output data/processed/propiedades_argenprop_features.csv

Etapas:
    1. Selección de columnas (descarte de sin señal).
    2. Target logarítmico y filtro de precios inválidos.
    3. Codificación ordinal de categóricas por mediana de precio.
    4. Imputación por mediana.
    5. Exportación de la matriz de features.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Permite importar el paquete `real_estate` (layout src/) sin instalarlo.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from real_estate.features.pipeline import construir_features, mostrar_features  # noqa: E402


def construir_features_csv(input_file: str | Path, output_file: str | Path) -> None:
    """Lee el CSV curado, construye la matriz de features y la guarda."""

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

    print(f"Cargando dataset: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)

    df = construir_features(df)

    mostrar_features(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING FINALIZADO")
    print("=" * 70)

    print(f"\nMatriz de features guardada en:\n{output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature Engineering: construye la matriz de features a partir del dataset curado"
    )
    parser.add_argument(
        "--input",
        default="data/processed/propiedades_argenprop_curado.csv",
        help="CSV curado de entrada (default: data/processed/propiedades_argenprop_curado.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/propiedades_argenprop_features.csv",
        help="CSV de salida con la matriz de features (default: data/processed/propiedades_argenprop_features.csv)",
    )
    args = parser.parse_args()

    construir_features_csv(input_file=args.input, output_file=args.output)


if __name__ == "__main__":
    main()
