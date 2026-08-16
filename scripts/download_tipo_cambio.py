#!/usr/bin/env python3
"""
Descarga el histórico del dólar (mercado blue) desde ArgentinaDatos y lo
guarda como dataset versionado en `data/external/tipo_cambio_blue.csv`.

Uso:
    python scripts/download_tipo_cambio.py
    python scripts/download_tipo_cambio.py --output data/external/tipo_cambio_blue.csv

La serie arranca en 2011 y llega al día hábil más reciente. El dataset se
usa en `real_estate.curation.transformations` para traducir precios en ARS
a USD según la fecha de publicación/scraping, sin depender de la API en
cada corrida de curación.
"""

import argparse
import csv
import sys
from pathlib import Path

import requests

# Serie completa del dólar blue (compra/venta por día hábil).
URL_HISTORICO = "https://api.argentinadatos.com/v1/cotizaciones/dolares/blue"

COLUMNAS = ["fecha", "compra", "venta"]

OUTPUT_DEFAULT = "data/external/tipo_cambio_blue.csv"


def descargar_historico(url: str, timeout: int = 30) -> list[dict[str, object]]:
    """Descarga la serie completa y valida que tenga el formato esperado."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    datos = response.json()

    if not isinstance(datos, list):
        raise ValueError(f"La API no devolvió una lista: {type(datos)}")

    return [dict(d) for d in datos]


def guardar_historico(datos: list[dict[str, object]], ruta: str) -> None:
    """Guarda la serie en CSV (fecha, compra, venta)."""
    salida = Path(ruta)
    salida.parent.mkdir(parents=True, exist_ok=True)

    with open(salida, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS)
        writer.writeheader()

        for d in datos:
            writer.writerow(
                {
                    "fecha": d.get("fecha", ""),
                    "compra": d.get("compra", ""),
                    "venta": d.get("venta", ""),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga el histórico del dólar blue y lo guarda como CSV"
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_DEFAULT,
        help=f"Archivo CSV de salida (default: {OUTPUT_DEFAULT})",
    )
    args = parser.parse_args()

    print(f"Descargando histórico del dólar blue desde {URL_HISTORICO} ...")

    try:
        datos = descargar_historico(URL_HISTORICO)
    except requests.RequestException as e:
        print(f"Error de red: {e}", file=sys.stderr)
        sys.exit(1)

    if not datos:
        print("La API devolvió una serie vacía.", file=sys.stderr)
        sys.exit(1)

    guardar_historico(datos, args.output)

    primera = datos[0].get("fecha")
    ultima = datos[-1].get("fecha")
    print(f"Guardados {len(datos):,} registros en {args.output}")
    print(f"Rango: {primera} -> {ultima}")

    # Sanity check de que las fechas estén ordenadas y sin duplicados.
    fechas = [d.get("fecha") for d in datos]
    if len(fechas) != len(set(fechas)):
        print("Aviso: la serie contiene fechas duplicadas.", file=sys.stderr)


if __name__ == "__main__":
    main()
