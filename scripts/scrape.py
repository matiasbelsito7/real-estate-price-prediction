#!/usr/bin/env python3
"""
Entry point del scraper de propiedades en venta en Capital Federal (Argenprop).

Uso:
    python scripts/scrape.py
    python scripts/scrape.py --max-paginas 5 --output data/raw/prueba.csv
    python scripts/scrape.py --pagina-inicio 1 --delay-min 2 --delay-max 4
    python scripts/scrape.py --todos-los-barrios --tipo departamentos
    python scripts/scrape.py --barrios palermo,recoleta,caballito

El sitio corta toda búsqueda en la página 100 (HTTP 202 vacío): son
2000 avisos por búsqueda, aunque el widget de paginación muestre miles
de páginas. Para acumular más (por ejemplo 10.000+) hay que recorrer
varios segmentos: `--barrios` y/o `--tipo` dividen la búsqueda y cada
segmento tiene su propio paginado y su propio cap de 100 páginas.

Por defecto arranca en la página 1 y sigue hasta que el sitio deje de
devolver avisos. Guarda a medida que avanza en el CSV, así que si se
corta a mitad de camino podés volver a correrlo y sigue donde quedó
(no vuelve a bajar avisos que ya tenés, según su 'id'). El progreso por
segmento se guarda en `--progreso` (JSON), así que una corrida
interrumpida reanuda desde la última página procesada.
"""

import argparse
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
        default="data/raw/progreso_scrape.json",
        help="Archivo JSON con el progreso por segmento (default: data/raw/progreso_scrape.json)",
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

    print(
        f"Recorro {len(segmentos)} segmento(s): "
        + ", ".join(f"'{nombre}'" for nombre, _ in segmentos)
    )

    try:
        for nombre_segmento, base_url in segmentos:
            print(f"\n========== Segmento: {nombre_segmento} ==========")
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
            )
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario. Lo scrapeado hasta ahora ya está guardado en el CSV.")
        sys.exit(0)


if __name__ == "__main__":
    main()
