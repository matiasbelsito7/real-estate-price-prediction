#!/usr/bin/env python3
"""
Scraper de NUEVAS publicaciones de Argenprop a un dataset separado.

Objetivo: capturar las publicaciones que van apareciendo en Argenprop en un
dataset propio (`data/raw/propiedades_nuevas.csv`), SIN tocar el dataset de
entrenamiento. El dedup por 'id' se hace contra este mismo dataset.

Uso:
    python scripts/scrape_nuevas.py
    python scripts/scrape_nuevas.py --max-paginas 10
    python scripts/scrape_nuevas.py --barrios palermo,recoleta
    python scripts/scrape_nuevas.py --todos-los-barrios

Se diseñó para correrse programado (cada X días). A diferencia de scrape.py,
esta corrida SIEMPRE re-escaneea los segmentos aunque el progreso los tenga
como completos (`revision_periodica=True`): las publicaciones nuevas aparecen
después de la última corrida, y el dedup evita guardar duplicados. Si una
corrida queda interrumpida, reanuda desde la última página procesada.
"""

import argparse
import logging
import sys
from pathlib import Path

# Permite importar el paquete `real_estate` (layout src/) sin instalarlo.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from real_estate.ingestion.scraper import (  # noqa: E402
    BARRIOS_CABA,
    construir_url_segmento,
    scrapear,
)
from real_estate.utils.logging import configurar_logging  # noqa: E402

logger = logging.getLogger(__name__)

OUTPUT_DEFAULT = "data/raw/propiedades_nuevas.csv"
PROGRESO_DEFAULT = "data/raw/progreso_scrape_nuevas.json"


def main() -> None:
    configurar_logging()
    parser = argparse.ArgumentParser(
        description="Scraper de nuevas publicaciones de Argenprop (dataset separado)"
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_DEFAULT,
        help=f"Archivo CSV de salida (default: {OUTPUT_DEFAULT})",
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
    parser.add_argument(
        "--tipo",
        default=None,
        help="Tipo de propiedad (p. ej. departamentos, casas, ph). Restringe el segmento a ese tipo.",
    )
    parser.add_argument(
        "--barrios",
        default=None,
        help="Slugs de barrios separados por coma (p. ej. palermo,recoleta,caballito). Un segmento por barrio.",
    )
    parser.add_argument(
        "--todos-los-barrios",
        action="store_true",
        help="Recorre los 54 barrios de Capital Federal (un segmento por barrio).",
    )
    parser.add_argument(
        "--progreso",
        default=PROGRESO_DEFAULT,
        help=f"Archivo JSON con el progreso por segmento (default: {PROGRESO_DEFAULT})",
    )
    parser.add_argument(
        "--reintentos-202",
        type=int,
        default=5,
        help="Reintentos con backoff ante un HTTP 202 (bloqueo del sitio) antes de pausar o cortar",
    )
    parser.add_argument(
        "--backoff-202",
        type=float,
        default=15.0,
        help="Backoff inicial (segundos) para reintentar un 202; se duplica en cada intento (máx 120s)",
    )
    parser.add_argument(
        "--pausa-bloqueo",
        type=float,
        default=300.0,
        help="Pausa larga (segundos) cuando el bloqueo 202 persiste, antes de una nueva tanda de reintentos",
    )
    args = parser.parse_args()

    if args.todos_los_barrios:
        barrios = BARRIOS_CABA
    elif args.barrios:
        barrios = [b.strip() for b in args.barrios.split(",") if b.strip()]
    else:
        barrios = []

    if barrios:
        segmentos = [(b, construir_url_segmento(tipo=args.tipo, barrio=b)) for b in barrios]
    else:
        segmentos = [
            ("global", construir_url_segmento(tipo=args.tipo)),
        ]

    logger.info(
        f"Recorro {len(segmentos)} segmento(s) en modo revision periódica "
        "(re-escaneea aunque el progreso diga completo): "
        + ", ".join(f"'{nombre}'" for nombre, _ in segmentos)
    )

    try:
        for nombre_segmento, base_url in segmentos:
            logger.info(f"\n========== Segmento: {nombre_segmento} ==========")
            scrapear(
                output_path=args.output,
                max_paginas=args.max_paginas,
                pagina_inicio=args.pagina_inicio,
                delay_min=args.delay_min,
                delay_max=args.delay_max,
                html_debug=args.html_debug,
                base_url=base_url,
                nombre_segmento=nombre_segmento,
                archivo_progreso=args.progreso,
                reintentos_202=args.reintentos_202,
                backoff_202_inicial=args.backoff_202,
                pausa_bloqueo=args.pausa_bloqueo,
                revision_periodica=True,
            )
    except KeyboardInterrupt:
        logger.info(
            "\nInterrumpido por el usuario. Lo scrapeado hasta ahora ya está guardado en el CSV."
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
