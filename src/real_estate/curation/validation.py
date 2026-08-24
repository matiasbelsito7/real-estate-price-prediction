"""
Validación de coherencia del dataset curado (Data Curation).

Detecta valores anómalos o inconsistentes sin modificar el dataset:

- precio > 0
- superficie > 0
- ambientes >= 1

Los valores faltantes no se consideran inválidos: solo se reportan
registros con valores presentes pero sin sentido (p. ej., precio <= 0).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

logger = logging.getLogger(__name__)


def _contar_invalidos(
    df: pd.DataFrame,
    columnas: list[str],
    es_invalido: Callable[[pd.Series], pd.Series],
) -> int:
    """Cuenta registros con valor presente e inválido en alguna columna."""
    mask = pd.Series(False, index=df.index, dtype=bool)
    for col in columnas:
        if col in df.columns:
            serie = df[col]
            mask = mask | (serie.notna() & es_invalido(serie))
    return len(df[mask])


def validar_precio_positivo(df: pd.DataFrame) -> int:
    """Registros con precio (o precio_usd) presente y <= 0."""
    return _contar_invalidos(df, ["precio", "precio_usd"], lambda s: s <= 0)


def validar_superficie_positiva(df: pd.DataFrame) -> int:
    """Registros con superficie presente y <= 0 (cubierta/semicubierta/total)."""
    return _contar_invalidos(
        df,
        ["superficie_cubierta", "superficie_semicubierta", "superficie_total"],
        lambda s: s <= 0,
    )


def validar_ambientes_minimos(df: pd.DataFrame) -> int:
    """Registros con ambientes presente y < 1."""
    return _contar_invalidos(df, ["ambientes"], lambda s: s < 1)


def validar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta las reglas de validación y devuelve un resumen.

    No modifica el DataFrame de entrada: solo reporta.
    """

    total = len(df)

    reglas = [
        ("precio_positivo", "precio > 0", validar_precio_positivo(df)),
        (
            "superficie_positiva",
            "superficie > 0 (cubierta/semicubierta/total)",
            validar_superficie_positiva(df),
        ),
        ("ambientes_minimos", "ambientes >= 1", validar_ambientes_minimos(df)),
    ]

    reporte = pd.DataFrame(
        {
            "regla": [regla[0] for regla in reglas],
            "descripcion": [regla[1] for regla in reglas],
            "registros_invalidos": [regla[2] for regla in reglas],
            "porcentaje": [round(100 * regla[2] / total, 2) if total else 0.0 for regla in reglas],
        }
    )

    logger.info("=" * 70)
    logger.info("VALIDACIÓN")
    logger.info("=" * 70)
    logger.info("\n%s", reporte.to_string(index=False))

    return reporte
