"""
Limpieza y conversión de tipos (Data Curation).

Convierte los campos que vienen como texto del dataset crudo a valores
numéricos, contemplando formatos reales: separadores de miles, símbolos
de moneda, "m²", texto adicional, entidades HTML, etc.

Ejemplos:
    "300 m² cubie."        -> 300.0
    "90 m²"                -> 90.0
    "2 dorm."              -> 2.0
    "17 años"              -> 17.0
    "$250.000"             -> 250000.0
    "&plus; $2.200.000
     expensas"             -> 2200000.0
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# Columnas que deberían ser numéricas
NUMERIC_COLUMNS = [
    "precio",
    "superficie_cubierta",
    "superficie_semicubierta",
    "superficie_total",
    "ambientes",
    "dormitorios",
    "banos",
    "cocheras",
    "antiguedad",
]


def limpiar_numero(valor: Any) -> Any:
    """
    Convierte strings inmobiliarios a números.

    También maneja:
        NaN
        None
        strings vacíos
    """

    if pd.isna(valor):
        return pd.NA

    texto = str(valor).strip()

    if not texto:
        return pd.NA

    # Elimina entidades HTML que puedan haber quedado
    texto = texto.replace("&plus;", "")
    texto = texto.replace("&nbsp;", " ")

    # Elimina símbolos de moneda y saltos de línea que suelen traer los
    # textos crudos del listado, p. ej. "&plus; $330.000\nexpensas".
    texto = texto.replace("$", "")
    texto = texto.replace("\n", "").replace("\r", "")

    # Elimina espacios
    texto = texto.replace(" ", "")

    # Toma solo el bloque numérico inicial (dígitos y separadores de miles
    # o decimales). El texto que sigue al número en la tarjeta (p. ej.
    # "expensas", "dorm.", "m² cubie.") rompía el chequeo de separador de
    # miles, dejando valores 1000x más chicos.
    coincidencia = re.match(r"\d[\d.,]*", texto)
    if coincidencia:
        texto = coincidencia.group()

    # --------------------------------------------------------
    # Caso típico argentino:
    #
    # 2.200.000 -> 2200000
    # 1.500,50  -> 1500.50
    #
    # --------------------------------------------------------

    # Si tiene punto y coma:
    if "." in texto and "," in texto:
        # Ejemplo: 1.250,50
        # puntos = separadores de miles, coma = decimal
        texto = texto.replace(".", "")
        texto = texto.replace(",", ".")

    elif "." in texto:
        partes = texto.split(".")

        # Si después del último punto hay exactamente tres dígitos,
        # asumimos separador de miles:
        # 2.200.000 -> 2200000
        # 250.000   -> 250000
        if all(len(parte) == 3 for parte in partes[1:]):
            texto = texto.replace(".", "")

    elif "," in texto:
        partes = texto.split(",")

        # 1500,50 -> 1500.50
        if len(partes[-1]) == 2:
            texto = texto.replace(",", ".")

    # --------------------------------------------------------
    # Extraemos el primer número
    # --------------------------------------------------------

    match = re.search(r"\d+(?:\.\d+)?", texto)

    if not match:
        return pd.NA

    try:
        return float(match.group())

    except ValueError:
        return pd.NA


def limpiar_columnas_numericas(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte las columnas numéricas a tipos numéricos."""

    print("\n" + "=" * 70)
    print("1. LIMPIEZA DE TIPOS")
    print("=" * 70)

    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            continue

        print(f"Convirtiendo: {column}")

        df[column] = df[column].apply(limpiar_numero)
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def limpiar_expensas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte expensas de texto a número.

    Ejemplo:
        "&plus; $2.200.000 expensas" -> 2200000
    """

    if "expensas" not in df.columns:
        return df

    print("\nConvirtiendo expensas...")

    df["expensas"] = df["expensas"].apply(limpiar_numero)
    df["expensas"] = pd.to_numeric(df["expensas"], errors="coerce")

    return df


def preparar_fecha(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte fecha_scrape a datetime."""

    if "fecha_scrape" not in df.columns:
        return df

    df["fecha_scrape"] = pd.to_datetime(
        df["fecha_scrape"],
        errors="coerce",
        utc=True,
    )

    return df
