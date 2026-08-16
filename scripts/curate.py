#!/usr/bin/env python3
"""
Entry point de la etapa de Data Curation.

Uso:
    python scripts/curate.py
    python scripts/curate.py --input data/raw/propiedades_argenprop.csv --output data/processed/propiedades_argenprop_curado.csv

Etapas:
    1. Limpieza y conversión de tipos.
    2. Normalización de moneda a USD (mercado blue, por fecha de scraping).
    3. Manejo de valores faltantes (indicadores de informado).
    4. Validación de coherencia (reporte).
    5. Exportación del dataset curado.
"""

import argparse
import sys
from pathlib import Path

# Permite importar el paquete `real_estate` (layout src/) sin instalarlo.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from real_estate.curation.pipeline import curar_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data Curation: limpia, normaliza a USD y valida el dataset crudo de Argenprop"
    )
    parser.add_argument(
        "--input",
        default="data/raw/propiedades_argenprop.csv",
        help="CSV crudo de entrada (default: data/raw/propiedades_argenprop.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/propiedades_argenprop_curado.csv",
        help="CSV curado de salida (default: data/processed/propiedades_argenprop_curado.csv)",
    )
    args = parser.parse_args()

    curar_csv(input_file=args.input, output_file=args.output)


if __name__ == "__main__":
    main()
