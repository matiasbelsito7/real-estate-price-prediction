"""
Clasificación de oportunidades de compra (Fase 3 del roadmap).

`clasificar_oportunidades` agrega a un dataset evaluado (el que produce
`evaluar_nuevas`, con `precio_usd` publicado y `precio_predicho_usd`) dos
columnas:

- `ratio_precio`: `precio_predicho_usd / precio_publicado_usd` (score
  relativo; NaN cuando el precio publicado no es un valor positivo finito).
- `clasificacion`: `buena_compra` / `mala_compra` / `sin_clasificar` según la
  zona neutra `1 ± std`, donde `std` es la desviación estándar del ratio
  computada sobre el lote de la corrida (solo ratios válidos).

`clasificar_por_diferencia` clasifica **por propiedad** (no por lote): suma
`diferencia_usd`, `diferencia_porcentual` y `clasificacion` con valores
`buena_compra` / `precio_justo` / `mala_compra` según un umbral porcentual
fijo (±10 % por defecto). Es la clasificación estable que persiste el ETL
periódico (fase 12) en PostgreSQL y expone la API.

`clasificar_y_exportar` orquesta el flujo completo del ranking CSV: lee el
CSV evaluado, clasifica, ordena por ratio descendente (mejores oportunidades
primero) y guarda `reports/ofertas.csv`. `scripts/clasificar_ofertas.py` es
el entry point de CLI (wrapper fino).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_DEFAULT = "data/processed/propiedades_nuevas_evaluadas.csv"
OUTPUT_DEFAULT = "reports/ofertas.csv"

BUENA_COMPRA = "buena_compra"
MALA_COMPRA = "mala_compra"
PRECIO_JUSTO = "precio_justo"
SIN_CLASIFICAR = "sin_clasificar"

#: Umbral porcentual (±10 %) de la clasificación por propiedad: si la
#: diferencia supera el umbral en cualquier dirección se sale de "precio_justo".
UMBRAL_PRECIO_JUSTO_DEFAULT = 0.10

# Columnas del ranking de ofertas: identificación, precios, score y clasificación.
COLUMNAS_OFERTAS = [
    "id",
    "titulo",
    "link",
    "barrio",
    "tipo_propiedad",
    "precio_usd",
    "precio_predicho_usd",
    "ratio_precio",
    "clasificacion",
    "fecha_prediccion",
]


def _ratio_precio(df: pd.DataFrame) -> pd.Series:
    """Ratio predicho/publicado; NaN si el publicado no es positivo y finito."""

    publicado = pd.to_numeric(df["precio_usd"], errors="coerce")
    predicho = pd.to_numeric(df["precio_predicho_usd"], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = predicho / publicado

    return ratio.replace([np.inf, -np.inf], np.nan)


def clasificar_oportunidades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega `ratio_precio` y `clasificacion` al dataset evaluado.

    La zona neutra es `1 ± std`, con `std` calculada sobre los ratios válidos
    del lote (desviación estándar muestral, ddof=1). Si hay menos de dos
    ratios válidos no hay desviación computable y se usa `std = 0` (la zona
    neutra degenera a `[1, 1]`): se clasifica comparando contra 1. Los ratios
    inválidos (precio publicado 0, faltante o no finito) quedan sin clasificar.
    """

    df = df.copy()

    df["ratio_precio"] = _ratio_precio(df)

    ratio_valido = df["ratio_precio"].dropna()
    std = float(ratio_valido.std())
    if math.isnan(std):
        std = 0.0

    limite_superior = 1 + std
    limite_inferior = 1 - std

    # np.select: las comparaciones con NaN son False -> caen en el default.
    ratio = df["ratio_precio"].to_numpy()
    df["clasificacion"] = np.select(
        [
            ratio > limite_superior,
            ratio < limite_inferior,
        ],
        [BUENA_COMPRA, MALA_COMPRA],
        default=SIN_CLASIFICAR,
    )

    return df


def clasificar_por_diferencia(
    df: pd.DataFrame,
    *,
    umbral: float = UMBRAL_PRECIO_JUSTO_DEFAULT,
) -> pd.DataFrame:
    """
    Clasifica cada propiedad por su diferencia predicho/publicado.

    Agrega `diferencia_usd` y `diferencia_porcentual` (en %) además de
    `clasificacion` con valores `buena_compra` / `precio_justo` /
    `mala_compra` según un umbral porcentual fijo (`umbral`, ±10 % por
    defecto). Las propiedades sin precio publicado válido (0, faltante o no
    finito) quedan con `diferencia_usd`/`diferencia_porcentual` NaN y
    clasificadas como `sin_clasificar`.
    """

    df = df.copy()

    publicado = pd.to_numeric(df["precio_usd"], errors="coerce")
    predicho = pd.to_numeric(df["precio_predicho_usd"], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        diferencia_usd = predicho - publicado
        diferencia_porcentual = ((predicho - publicado) / publicado) * 100

    valido = publicado > 0
    df["diferencia_usd"] = np.where(valido, diferencia_usd, np.nan)
    df["diferencia_porcentual"] = np.where(valido, diferencia_porcentual, np.nan)

    # np.select: las comparaciones con NaN son False -> caen en el default
    # (precio_justo), así que los inválidos se corrigen después.
    porcentual = df["diferencia_porcentual"].to_numpy()
    clasificacion = np.select(
        [
            porcentual > umbral * 100,
            porcentual < -umbral * 100,
        ],
        [BUENA_COMPRA, MALA_COMPRA],
        default=PRECIO_JUSTO,
    )

    df["clasificacion"] = np.where(valido, clasificacion, SIN_CLASIFICAR)

    return df


def clasificar_y_exportar(
    input_file: str | Path = INPUT_DEFAULT,
    output_file: str | Path = OUTPUT_DEFAULT,
) -> Path:
    """
    Lee el CSV evaluado, clasifica cada publicación y guarda el ranking.

    El ranking ordena por `ratio_precio` descendente (mejores oportunidades
    primero); las publicaciones sin ratio válido quedan al final. Devuelve la
    ruta del archivo de salida.
    """

    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

    print(f"Cargando dataset evaluado: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)

    df = clasificar_oportunidades(df)

    print(f"\nPublicaciones clasificadas: {len(df):,}")
    print(df["clasificacion"].value_counts().to_string())

    df = df.sort_values("ratio_precio", ascending=False, na_position="last")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df[COLUMNAS_OFERTAS].to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("RANKING DE OPORTUNIDADES")
    print("=" * 70)

    top = df[COLUMNAS_OFERTAS].head(10)
    print(f"\nTop {len(top)} (ratio descendente):")
    print(top.to_string(index=False))

    print(f"\nRanking guardado en:\n{output_path}")

    return output_path
