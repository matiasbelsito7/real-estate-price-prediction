#!/usr/bin/env python3
"""
Entry point de la etapa de Modelado (Fase 5).

Uso:
    python scripts/train.py
    python scripts/train.py --input data/processed/propiedades_argenprop_curado.csv

Etapas:
    1. Carga del dataset curado, selección de columnas y target logarítmico.
    2. Split reproducible train / val / test (80 / 10 / 10).
    3. Preprocesamiento ajustado solo sobre train (sin fuga) y reaplicado a
       val / test.
    4. Baseline (mediana) y XGBoost entrenados sobre train.
    5. Evaluación sobre val (comparación) y sobre test (modelo final).

Nota: el guardado de artefactos del modelo y el tracking con MLflow llegan en
la fase 6.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Permite importar el paquete `real_estate` (layout src/) sin instalarlo.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from real_estate.features.pipeline import dividir_train_val_test  # noqa: E402
from real_estate.features.transformations import (  # noqa: E402
    crear_target_log,
    seleccionar_columnas,
)
from real_estate.models.entrenamiento import entrenar_y_evaluar  # noqa: E402


def entrenar_pipeline(input_file: str | Path, random_state: int = 42) -> None:
    """Corre el pipeline de modelado completo y reporta las métricas."""

    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

    print(f"Cargando dataset: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)

    df = crear_target_log(seleccionar_columnas(df))

    train, val, test = dividir_train_val_test(df, random_state=random_state)

    entrenar_y_evaluar(train, val, test, random_state=random_state)

    print("\n" + "=" * 70)
    print("MODELADO FINALIZADO (fase 5)")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Modelado: pipeline train/val/test sin fuga, baseline (mediana) y "
            "XGBoost con evaluación sobre el dataset curado"
        )
    )
    parser.add_argument(
        "--input",
        default="data/processed/propiedades_argenprop_curado.csv",
        help=("CSV curado de entrada (default: data/processed/propiedades_argenprop_curado.csv)"),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla para el split y el entrenamiento (default: 42)",
    )
    args = parser.parse_args()

    entrenar_pipeline(input_file=args.input, random_state=args.random_state)


if __name__ == "__main__":
    main()
