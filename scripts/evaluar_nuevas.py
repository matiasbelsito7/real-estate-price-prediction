#!/usr/bin/env python3
"""
Entry point de la predicción sobre las nuevas publicaciones (Fase 2).

Uso:
    python scripts/evaluar_nuevas.py
    python scripts/evaluar_nuevas.py --input data/raw/propiedades_nuevas.csv
    python scripts/evaluar_nuevas.py --modelo models/modelo_precio_propiedades
    python scripts/evaluar_nuevas.py --output data/processed/propiedades_nuevas_evaluadas.csv

Etapas (en `real_estate.serving.evaluar`):
    1. Curación del dataset de nuevas en memoria (misma pipeline de la fase 3).
    2. Carga del bundle de serving con el modelo XGBoost y el preprocesamiento.
    3. Predicción de `precio_predicho_usd` por publicación.
    4. Exportación del dataset evaluado.

Pensado para correrse después de `scripts/scrape_nuevas.py` (programado con el
mismo cron). No persiste curado/features intermedios: solo el CSV crudo de
entrada y el CSV evaluado de salida.
"""

import argparse
import logging
import sys
from pathlib import Path

# Permite importar el paquete `real_estate` (layout src/) sin instalarlo.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from real_estate.serving.evaluar import (  # noqa: E402
    INPUT_DEFAULT,
    MODELO_DEFAULT,
    OUTPUT_DEFAULT,
    evaluar_nuevas,
)
from real_estate.utils.logging import configurar_logging  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    configurar_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Predice el precio de las nuevas publicaciones de Argenprop "
            "con el modelo exportado (bundle de serving)"
        )
    )
    parser.add_argument(
        "--input",
        default=INPUT_DEFAULT,
        help=f"CSV crudo de nuevas publicaciones (default: {INPUT_DEFAULT})",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_DEFAULT,
        help=f"CSV evaluado de salida (default: {OUTPUT_DEFAULT})",
    )
    parser.add_argument(
        "--modelo",
        default=MODELO_DEFAULT,
        help=f"Directorio del bundle de serving (default: {MODELO_DEFAULT})",
    )
    args = parser.parse_args()

    evaluar_nuevas(
        input_file=args.input,
        output_file=args.output,
        directorio_modelo=args.modelo,
    )


if __name__ == "__main__":
    main()
