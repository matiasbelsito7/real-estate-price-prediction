"""
Transformaciones del dataset (Feature Engineering).

Convierte el dataset curado en una matriz de features lista para modelar:
1. Descarte de columnas sin señal (EDA 01: cobertura < 3 % o texto/identificadores).
2. Target `log_precio_usd` (EDA 02: el log es aproximadamente normal) y filtro
   de precios inválidos (< `PRECIO_MINIMO_USD`).
3. Codificación ordinal de categóricas por mediana del precio (barrio,
   tipo_propiedad) — apta para árboles, que es el modelo planificado.
4. Imputación por mediana de las numéricas con faltantes.

Todas las funciones que "aprenden" del dataset (ordenes, medianas) exponen
una variante de ajuste (`crear_*`) y otra de aplicación (`aplicar_*`/`codificar_*`)
para poder ajustar sobre el set de entrenamiento y aplicar sobre val/test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columnas sin señal (EDA 01): cobertura < 3 % o texto libre/identificadores.
COLUMNAS_DESCARTAR = [
    # Cobertura casi nula (> 97 % faltante) + sus indicadores casi constantes
    "cocheras",
    "cocheras_informado",
    "superficie_semicubierta",
    "superficie_semicubierta_informado",
    "superficie_total",
    "superficie_total_informado",
    # Sub-barrio: 88.8 % faltante
    "sub_barrio",
    # Texto libre y metadatos
    "link",
    "titulo",
    "descripcion",
    "fecha_scrape",
    # Identificadores
    "id",
    "idtipopropiedad",
    # Redundancias con el dataset curado (ya normalizado a USD)
    "precio",
    "moneda",
    "tipo_cambio_ars_usd",
    "expensas",
]

# Numéricas con faltantes que se imputan con la mediana del entrenamiento.
COLUMNAS_IMPUTAR = [
    "superficie_cubierta",
    "ambientes",
    "dormitorios",
    "banos",
    "antiguedad",
    "expensas_usd",
]

# Indicadores binarios de "dato informado" que se conservan como features.
COLUMNAS_INDICADOR = [
    "superficie_cubierta_informado",
    "ambientes_informado",
    "dormitorios_informado",
    "banos_informado",
    "antiguedad_informado",
    "expensas_informado",
]

# Categóricas codificadas por orden de mediana de precio.
COLUMNAS_CATEGORICAS = ["barrio", "tipo_propiedad"]

TARGET_PRECIO = "precio_usd"
TARGET_LOG = "log_precio_usd"

# Precio mínimo creíble de una propiedad en USD. Los avisos con precios
# menores son artefactos del scraping (el campo "precio" parseó texto basura,
# p. ej. "U$S A B2" -> 1) y distorsionan el target logarítmico
# (log(1) = 0 frente a una mediana de log(precio) ~ 11.9).
PRECIO_MINIMO_USD = 1000

# Código reservado para categorías no vistas en el ajuste (o NaN).
CODIGO_DESCONOCIDO = -1


def seleccionar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conserva solo features y target, descartando las columnas sin señal.

    No modifica el DataFrame recibido: opera sobre una copia.
    """

    return df.drop(columns=COLUMNAS_DESCARTAR, errors="ignore")


def crear_target_log(
    df: pd.DataFrame,
    columna_precio: str = TARGET_PRECIO,
    precio_minimo: float = PRECIO_MINIMO_USD,
) -> pd.DataFrame:
    """
    Crea `log_precio_usd` y descarta precios inválidos (NaN o < `precio_minimo`).

    El logaritmo estabiliza la varianza de la cola larga de propiedades
    caras (EDA 02) y hace que el RMSE se interprete como error relativo.
    """

    df = df.copy()

    if columna_precio not in df.columns:
        raise ValueError(f"El dataset no tiene la columna '{columna_precio}'.")

    previas = len(df)

    df = df.loc[df[columna_precio] >= precio_minimo].copy()

    descartadas = previas - len(df)

    if descartadas:
        print(
            f"Descartadas {descartadas:,} filas con precio inválido (< {precio_minimo:,.0f} USD)."
        )

    df[TARGET_LOG] = np.log(df[columna_precio])

    return df


def crear_orden_mediana(
    df: pd.DataFrame,
    columna: str,
    target: str = TARGET_PRECIO,
) -> list[str]:
    """
    Devuelve las categorías de `columna` ordenadas por la mediana de `target`.

    El orden se ajusta sobre el set de entrenamiento; las categorías que
    aparecen ordenadas dan el ranking ordinal (el código 0 es la más barata).
    """

    if columna not in df.columns:
        raise ValueError(f"El dataset no tiene la columna '{columna}'.")

    if target not in df.columns:
        raise ValueError(f"El dataset no tiene la columna '{target}'.")

    # dropna=True excluye la categoría NaN (se codifica como desconocida).
    orden = (
        df.groupby(columna, dropna=True)[target].median().sort_values(ascending=True).index.tolist()
    )

    return [str(categoria) for categoria in orden]


def codificar_ordinal(
    df: pd.DataFrame,
    columna: str,
    orden: list[str],
) -> pd.DataFrame:
    """
    Reemplaza `columna` por `{columna}_ordinal` usando el ranking de `orden`.

    Las categorías ausentes del `orden` (o NaN) se codifican con
    `CODIGO_DESCONOCIDO`, de modo que el modelo no reciba valores nuevos
    sin mapear ni propague el NaN.
    """

    df = df.copy()

    if columna not in df.columns:
        raise ValueError(f"El dataset no tiene la columna '{columna}'.")

    codigos = {categoria: indice for indice, categoria in enumerate(orden)}

    columna_ordinal = f"{columna}_ordinal"

    df[columna_ordinal] = (
        df[columna].astype("string").map(codigos).fillna(CODIGO_DESCONOCIDO).astype("int32")
    )

    return df.drop(columns=[columna])


def crear_imputador(
    df: pd.DataFrame,
    columnas: list[str] | None = None,
) -> dict[str, float]:
    """
    Calcula la mediana de cada columna a imputar (sobre el entrenamiento).

    Devuelve un dict columna -> mediana; las columnas sin valores válidos
    quedan fuera del dict (y `aplicar_imputacion` las deja como NaN).
    """

    columnas = COLUMNAS_IMPUTAR if columnas is None else columnas

    imputador: dict[str, float] = {}

    for columna in columnas:
        if columna not in df.columns:
            continue

        mediana = df[columna].median()

        if not pd.isna(mediana):
            imputador[columna] = float(mediana)

    return imputador


def aplicar_imputacion(
    df: pd.DataFrame,
    imputador: dict[str, float],
) -> pd.DataFrame:
    """
    Rellena los NaN de cada columna con la mediana correspondiente.
    """

    df = df.copy()

    for columna, mediana in imputador.items():
        if columna in df.columns:
            df[columna] = df[columna].fillna(mediana)

    return df
