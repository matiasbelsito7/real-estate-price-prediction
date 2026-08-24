"""
Transformaciones del dataset (Data Curation).

Normalización de moneda a USD con tipo de cambio histórico (mercado blue),
usando el dataset versionado `data/external/tipo_cambio_blue.csv` como
fuente primaria y la API de ArgentinaDatos como fallback para fechas no
cubiertas. Además crea indicadores de valores informados (missing indicators).
"""

from __future__ import annotations

import csv
import logging
import os
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Mercado utilizado para convertir ARS -> USD.
# Opciones disponibles en ArgentinaDatos:
# oficial, blue, bolsa, contadoconliqui, mayorista, etc.
FX_MARKET = "blue"

FX_API_URL = "https://api.argentinadatos.com/v1/cotizaciones/dolares/{market}/{date}"

# Dataset histórico versionado del dólar blue (descargado con
# `scripts/download_tipo_cambio.py` y trackeado con DVC). Se usa como
# fuente primaria del tipo de cambio para no depender de la API en cada
# corrida de curación.
RUTA_TIPO_CAMBIO_HISTORICO = "data/external/tipo_cambio_blue.csv"

# Columnas donde queremos conservar información sobre
# si el dato fue informado o no.
MISSING_INDICATOR_COLUMNS = [
    "expensas",
    "superficie_cubierta",
    "superficie_semicubierta",
    "superficie_total",
    "ambientes",
    "dormitorios",
    "banos",
    "cocheras",
    "antiguedad",
]


def obtener_tipo_cambio(
    fecha: str,
    market: str = FX_MARKET,
    max_intentos: int = 7,
) -> float | None:
    """
    Obtiene el tipo de cambio histórico ARS/USD.

    Si la fecha cae en fin de semana o feriado,
    retrocede días hasta encontrar una cotización.

    Devuelve ARS por 1 USD.
    """

    fecha_actual = pd.Timestamp(fecha)

    for _ in range(max_intentos):
        fecha_str = fecha_actual.strftime("%Y/%m/%d")

        url = FX_API_URL.format(market=market, date=fecha_str)

        try:
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if isinstance(data, dict):
                    data = [data]

                if data:
                    # Usamos "venta":
                    # cuánto ARS cuesta comprar 1 USD.
                    venta = data[0].get("venta")

                    if venta is not None:
                        return float(venta)

        except (requests.RequestException, ValueError, TypeError):
            pass

        # Retrocedemos un día
        fecha_actual -= pd.Timedelta(days=1)

        time.sleep(0.2)

    return None


def cargar_tipo_cambio_historico(ruta: str) -> dict[str, float]:
    """
    Carga el dataset histórico de tipo de cambio (CSV versionado).

    Devuelve {fecha: venta} para cada fecha del CSV. Si el archivo no
    existe, devuelve un dict vacío (la conversión cae a la API).
    """

    if not os.path.exists(ruta):
        logger.warning("No existe '%s'; se usará la API por fecha", ruta)
        return {}

    tabla: dict[str, float] = {}

    with open(ruta, encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            fecha = fila.get("fecha")
            venta = fila.get("venta")

            if fecha and venta:
                tabla[fecha] = float(venta)

    logger.info("Tipo de cambio histórico cargado: %s fechas (%s)", f"{len(tabla):,}", ruta)

    return tabla


def construir_tabla_tipo_cambio(
    fechas: pd.Series,
    ruta_historico: str | None = None,
) -> dict[str, float]:
    """
    Obtiene el tipo de cambio una sola vez por fecha.

    Esto evita realizar una request por cada propiedad. Si se pasa
    `ruta_historico` (CSV versionado del dólar blue), se usa como fuente
    primaria; las fechas no cubiertas caen a la API.
    """

    fechas_unicas = sorted(fechas.dropna().dt.strftime("%Y-%m-%d").unique())

    logger.info("=" * 70)
    logger.info("2. OBTENIENDO TIPOS DE CAMBIO")
    logger.info("=" * 70)

    logger.info("Fechas únicas encontradas: %d", len(fechas_unicas))

    tabla_local = cargar_tipo_cambio_historico(ruta_historico) if ruta_historico else {}

    tipos_cambio: dict[str, float] = {}

    for fecha in fechas_unicas:
        tasa = tabla_local.get(fecha)

        if tasa is not None:
            logger.info("  %s -> histórico: 1 USD = %s ARS", fecha, f"{tasa:,.2f}")

        else:
            logger.info("  %s -> consultando %s...", fecha, FX_MARKET)

            tasa = obtener_tipo_cambio(fecha, FX_MARKET)

        if tasa is not None:
            tipos_cambio[fecha] = tasa

        else:
            logger.error("No se encontró cotización para %s", fecha)

    return tipos_cambio


def normalizar_moneda(
    df: pd.DataFrame,
    ruta_tipo_cambio: str | None = RUTA_TIPO_CAMBIO_HISTORICO,
) -> pd.DataFrame:
    """
    Crea precio_usd.

    Si la publicación ya está en USD:
        precio_usd = precio

    Si está en ARS:
        precio_usd = precio / tipo_cambio

    El tipo de cambio sale del dataset histórico versionado
    (`ruta_tipo_cambio`) y, para fechas no cubiertas, de la API.
    No modifica precio ni moneda originales.
    """

    logger.info("=" * 70)
    logger.info("3. NORMALIZACIÓN DE MONEDA")
    logger.info("=" * 70)

    if "precio" not in df.columns:
        raise ValueError("El dataset no tiene la columna 'precio'.")

    if "moneda" not in df.columns:
        raise ValueError("El dataset no tiene la columna 'moneda'.")

    if "fecha_scrape" not in df.columns:
        raise ValueError("El dataset no tiene 'fecha_scrape'.")

    fechas = df["fecha_scrape"].dt.strftime("%Y-%m-%d")

    tipos_cambio = construir_tabla_tipo_cambio(df["fecha_scrape"], ruta_historico=ruta_tipo_cambio)

    df["tipo_cambio_ars_usd"] = fechas.map(tipos_cambio)

    # Inicialmente dejamos todo como NaN
    df["precio_usd"] = pd.NA

    moneda = df["moneda"].astype("string").str.upper().str.strip()

    # --------------------------------------------
    # USD
    # --------------------------------------------

    mask_usd = moneda.eq("USD")

    df.loc[mask_usd, "precio_usd"] = df.loc[mask_usd, "precio"]

    # --------------------------------------------
    # ARS
    # --------------------------------------------

    mask_ars = moneda.eq("ARS")

    df.loc[mask_ars, "precio_usd"] = (
        df.loc[mask_ars, "precio"] / df.loc[mask_ars, "tipo_cambio_ars_usd"]
    )

    df["precio_usd"] = pd.to_numeric(df["precio_usd"], errors="coerce")

    logger.info("Propiedades en USD: %s", f"{mask_usd.sum():,}")
    logger.info("Propiedades en ARS: %s", f"{mask_ars.sum():,}")
    logger.info("Precio USD calculado: %s", f"{df['precio_usd'].notna().sum():,}")

    return df


def normalizar_expensas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea expensas_usd utilizando el mismo tipo de cambio
    de la fecha de scraping.

    Las expensas del dataset se consideran ARS cuando
    están informadas.
    """

    if "expensas" not in df.columns:
        return df

    if "tipo_cambio_ars_usd" not in df.columns:
        return df

    df["expensas_usd"] = df["expensas"] / df["tipo_cambio_ars_usd"]

    return df


def crear_indicadores_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables binarias que indican si un dato
    fue informado por el aviso.

    Ejemplo:
        banos = NaN  ->  banos_informado = 0
        banos = 2    ->  banos_informado = 1
    """

    logger.info("=" * 70)
    logger.info("4. MANEJO DE VALORES FALTANTES")
    logger.info("=" * 70)

    for column in MISSING_INDICATOR_COLUMNS:
        if column not in df.columns:
            continue

        indicator = f"{column}_informado"

        df[indicator] = df[column].notna().astype("int8")

        missing = df[column].isna().sum()

        logger.info("%-30s faltantes: %s", column, f"{missing:,}")

    return df
