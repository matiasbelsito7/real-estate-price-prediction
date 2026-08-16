#!/usr/bin/env python3
"""
Entry point del scraper de propiedades en venta en Capital Federal (Argenprop).

Uso:
    python scripts/scrape.py
    python scripts/scrape.py --max-paginas 5 --output data/raw/prueba.csv
    python scripts/scrape.py --pagina-inicio 1 --delay-min 2 --delay-max 4

Por defecto arranca en la página 1 y sigue hasta que el sitio deje de
devolver avisos. Guarda a medida que avanza en el CSV, así que si se
corta a mitad de camino podés volver a correrlo y sigue donde quedó
(no vuelve a bajar avisos que ya tenés, según su 'id').
"""

import argparse
import sys
from pathlib import Path

# Permite importar el paquete `real_estate` (layout src/) sin instalarlo.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from real_estate.ingestion.scraper import scrapear  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scraper de propiedades en venta en Capital Federal (Argenprop)"
    )
    parser.add_argument(
        "--output",
        default="data/raw/propiedades_argenprop.csv",
        help="Archivo CSV de salida (default: data/raw/propiedades_argenprop.csv)",
    )
    parser.add_argument(
        "--max-paginas", type=int, default=None, help="Límite de páginas a recorrer"
    )
    parser.add_argument("--pagina-inicio", type=int, default=1, help="Página desde la que arrancar")
    parser.add_argument(
        "--delay-min", type=float, default=2.0, help="Delay mínimo entre requests (segundos)"
    )
    parser.add_argument(
        "--delay-max", type=float, default=4.0, help="Delay máximo entre requests (segundos)"
    )
    parser.add_argument(
        "--html-debug",
        default=None,
        help="Si se pasa, guarda ahí el HTML crudo de la última página pedida (para diagnosticar cambios del sitio)",
    )
    args = parser.parse_args()

    try:
        scrapear(
            output_path=args.output,
            max_paginas=args.max_paginas,
            pagina_inicio=args.pagina_inicio,
            delay_min=args.delay_min,
            delay_max=args.delay_max,
            html_debug=args.html_debug,
        )
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario. Lo scrapeado hasta ahora ya está guardado en el CSV.")
        sys.exit(0)


if __name__ == "__main__":
    main()
