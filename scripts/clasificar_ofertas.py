#!/usr/bin/env python3
"""
Entry point de la clasificación de oportunidades de compra (Fase 3).

Uso:
    python scripts/clasificar_ofertas.py
    python scripts/clasificar_ofertas.py --input data/processed/propiedades_nuevas_evaluadas.csv
    python scripts/clasificar_ofertas.py --output reports/ofertas.csv

Etapas (en `real_estate.serving.clasificacion`):
    1. Lectura del CSV evaluado (salida de `scripts/evaluar_nuevas.py`).
    2. Cálculo del ratio `precio_predicho_usd / precio_usd` y la clasificación
       buena/mala compra con zona neutra `1 ± std` (std del lote).
    3. Ranking por ratio descendente (mejores oportunidades primero).
    4. Exportación de `reports/ofertas.csv`.

Pensado para correrse después de `scripts/evaluar_nuevas.py` (mismo cron).
"""

import argparse
import sys
from pathlib import Path

# Permite importar el paquete `real_estate` (layout src/) sin instalarlo.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from real_estate.serving.clasificacion import (  # noqa: E402
    INPUT_DEFAULT,
    OUTPUT_DEFAULT,
    clasificar_y_exportar,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clasifica las nuevas publicaciones evaluadas como buena/mala "
            "compra (zona neutra 1 ± std) y guarda el ranking de oportunidades"
        )
    )
    parser.add_argument(
        "--input",
        default=INPUT_DEFAULT,
        help=f"CSV evaluado de entrada (default: {INPUT_DEFAULT})",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_DEFAULT,
        help=f"Ranking de ofertas de salida (default: {OUTPUT_DEFAULT})",
    )
    args = parser.parse_args()

    clasificar_y_exportar(input_file=args.input, output_file=args.output)


if __name__ == "__main__":
    main()
