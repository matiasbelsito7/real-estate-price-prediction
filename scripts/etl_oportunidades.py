#!/usr/bin/env python3
"""
Entry point del ETL periódico de oportunidades (Fase 12).

Uso:
    python scripts/etl_oportunidades.py
    python scripts/etl_oportunidades.py --input data/raw/propiedades_nuevas.csv
    python scripts/etl_oportunidades.py --modelo models/modelo_precio_propiedades
    python scripts/etl_oportunidades.py --todos-los-barrios   # scrapea CABA antes del ETL

Etapas (la lógica vive en `real_estate.serving.etl_oportunidades`):
    1. Scraping (opcional) de NUEVAS publicaciones de Capital Federal.
    2. Dedup contra los ids ya persistidos en PostgreSQL (solo nuevas).
    3. Curación + predicción con el bundle del champion.
    4. Clasificación por propiedad (buena_compra / precio_justo / mala_compra).
    5. Persistencia en PostgreSQL y exportación del CSV de oportunidades.

La base se configura con variables de entorno (POSTGRES_*) o `.env`. En CI la
provee el servicio `postgres` del workflow de GitHub Actions.
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
from real_estate.persistencia.config import ConfiguracionPostgres  # noqa: E402
from real_estate.persistencia.db import crear_engine  # noqa: E402
from real_estate.persistencia.esquema import crear_tablas  # noqa: E402
from real_estate.serving.etl_oportunidades import (  # noqa: E402
    INPUT_DEFAULT,
    OUTPUT_DEFAULT,
    ejecutar_etl,
)
from real_estate.serving.evaluar import MODELO_DEFAULT  # noqa: E402

PROGRESO_DEFAULT = "data/raw/progreso_scrape_nuevas.json"


def _construir_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ETL periódico de oportunidades: scrapea (opcional) nuevas "
            "publicaciones de CABA, predice con el champion y persiste en PostgreSQL"
        )
    )

    scrape = parser.add_argument_group("scraping (opcional, desencadena el scrapeo)")
    scrape.add_argument(
        "--todos-los-barrios",
        action="store_true",
        help="Recorre los 54 barrios de Capital Federal (un segmento por barrio).",
    )
    scrape.add_argument(
        "--barrios",
        default=None,
        help="Slugs de barrios separados por coma (p. ej. palermo,recoleta).",
    )
    scrape.add_argument(
        "--max-paginas", type=int, default=None, help="Límite de páginas a recorrer"
    )
    scrape.add_argument("--pagina-inicio", type=int, default=1, help="Página desde la que arrancar")
    scrape.add_argument(
        "--tipo",
        default=None,
        help="Tipo de propiedad (p. ej. departamentos, casas, ph).",
    )
    scrape.add_argument(
        "--html-debug",
        default=None,
        help="Si se pasa, guarda ahí el HTML crudo de la última página pedida.",
    )
    scrape.add_argument(
        "--delay-min", type=float, default=2.0, help="Delay mínimo entre requests (segundos)"
    )
    scrape.add_argument(
        "--delay-max", type=float, default=4.0, help="Delay máximo entre requests (segundos)"
    )
    scrape.add_argument(
        "--progreso",
        default=PROGRESO_DEFAULT,
        help=f"Archivo JSON con el progreso por segmento (default: {PROGRESO_DEFAULT})",
    )

    etl = parser.add_argument_group("etl")
    etl.add_argument(
        "--input",
        default=INPUT_DEFAULT,
        help=f"CSV crudo de nuevas publicaciones (default: {INPUT_DEFAULT})",
    )
    etl.add_argument(
        "--output",
        default=OUTPUT_DEFAULT,
        help=f"CSV de oportunidades nuevas de salida (default: {OUTPUT_DEFAULT})",
    )
    etl.add_argument(
        "--modelo",
        default=MODELO_DEFAULT,
        help=f"Directorio del bundle de serving (default: {MODELO_DEFAULT})",
    )

    return parser.parse_args()


def _scrapear(args: argparse.Namespace) -> None:
    """Recorre los segmentos de CABA en modo revisión periódica (dedup por id
    contra el propio CSV). Es el mismo comportamiento que `scrape_nuevas.py`."""

    if args.todos_los_barrios:
        barrios = BARRIOS_CABA
    elif args.barrios:
        barrios = [b.strip() for b in args.barrios.split(",") if b.strip()]
    else:
        barrios = []

    if barrios:
        segmentos = [(b, construir_url_segmento(tipo=args.tipo, barrio=b)) for b in barrios]
    else:
        segmentos = [("global", construir_url_segmento(tipo=args.tipo))]

    print(
        f"Recorro {len(segmentos)} segmento(s) en modo revisión periódica "
        "(re-escaneea aunque el progreso diga completo): "
        + ", ".join(f"'{nombre}'" for nombre, _ in segmentos)
    )

    for nombre_segmento, base_url in segmentos:
        print(f"\n========== Segmento: {nombre_segmento} ==========")
        scrapear(
            output_path=args.input,
            max_paginas=args.max_paginas,
            pagina_inicio=args.pagina_inicio,
            delay_min=args.delay_min,
            delay_max=args.delay_max,
            html_debug=args.html_debug,
            base_url=base_url,
            nombre_segmento=nombre_segmento,
            archivo_progreso=args.progreso,
            revision_periodica=True,
        )


def main() -> None:
    args = _construir_argumentos()

    if args.todos_los_barrios or args.barrios:
        _scrapear(args)

    config = ConfiguracionPostgres()
    engine = crear_engine(config)
    crear_tablas(engine)

    ejecutar_etl(
        engine=engine,
        input_file=args.input,
        output_file=args.output,
        directorio_modelo=args.modelo,
    )


if __name__ == "__main__":
    main()
