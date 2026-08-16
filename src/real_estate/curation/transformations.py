"""
Transformaciones del dataset (Data Curation).

Normalización de moneda a USD con tipo de cambio histórico (mercado blue
de ArgentinaDatos, consultado por fecha de scraping) y creación de
indicadores de valores informados (missing indicators).
"""

from __future__ import annotations

import time

import pandas as pd
import requests

# Mercado utilizado para convertir ARS -> USD.
# Opciones disponibles en ArgentinaDatos:
# oficial, blue, bolsa, contadoconliqui, mayorista, etc.
FX_MARKET = "blue"

FX_API_URL = "https://api.argentinadatos.com/v1/cotizaciones/dolares/{market}/{date}"

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


def construir_tabla_tipo_cambio(fechas: pd.Series) -> dict[str, float]:
    """
    Obtiene el tipo de cambio una sola vez por fecha.

    Esto evita realizar una request por cada propiedad.
    """

    fechas_unicas = sorted(fechas.dropna().dt.strftime("%Y-%m-%d").unique())

    print("\n" + "=" * 70)
    print("2. OBTENIENDO TIPOS DE CAMBIO")
    print("=" * 70)

    print(f"Fechas únicas encontradas: {len(fechas_unicas)}")

    tipos_cambio: dict[str, float] = {}

    for fecha in fechas_unicas:
        print(f"  {fecha} -> consultando {FX_MARKET}...")

        tasa = obtener_tipo_cambio(fecha, FX_MARKET)

        if tasa is not None:
            tipos_cambio[fecha] = tasa

            print(f"      1 USD = {tasa:,.2f} ARS")

        else:
            print("      ERROR: no se encontró cotización")

    return tipos_cambio


def normalizar_moneda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea precio_usd.

    Si la publicación ya está en USD:
        precio_usd = precio

    Si está en ARS:
        precio_usd = precio / tipo_cambio

    No modifica precio ni moneda originales.
    """

    print("\n" + "=" * 70)
    print("3. NORMALIZACIÓN DE MONEDA")
    print("=" * 70)

    if "precio" not in df.columns:
        raise ValueError("El dataset no tiene la columna 'precio'.")

    if "moneda" not in df.columns:
        raise ValueError("El dataset no tiene la columna 'moneda'.")

    if "fecha_scrape" not in df.columns:
        raise ValueError("El dataset no tiene 'fecha_scrape'.")

    fechas = df["fecha_scrape"].dt.strftime("%Y-%m-%d")

    tipos_cambio = construir_tabla_tipo_cambio(df["fecha_scrape"])

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

    print(f"\nPropiedades en USD: {mask_usd.sum():,}")
    print(f"Propiedades en ARS: {mask_ars.sum():,}")
    print(f"Precio USD calculado: {df['precio_usd'].notna().sum():,}")

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

    print("\n" + "=" * 70)
    print("4. MANEJO DE VALORES FALTANTES")
    print("=" * 70)

    for column in MISSING_INDICATOR_COLUMNS:
        if column not in df.columns:
            continue

        indicator = f"{column}_informado"

        df[indicator] = df[column].notna().astype("int8")

        missing = df[column].isna().sum()

        print(f"{column:30s} faltantes: {missing:,}")

    return df
