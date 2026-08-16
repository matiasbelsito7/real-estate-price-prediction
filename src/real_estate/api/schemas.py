"""
Schemas Pydantic de la API de predicción (Fase 10).

`PropiedadEntrada` modela el payload que recibe `/predict`. Los indicadores
`*_informado` (6 columnas que marcan si el dato numérico fue informado o se
imputó) se derivan automáticamente de la presencia del valor si no se envían
explícitamente, replicando la lógica de la etapa de features.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Pares columna numérica -> columna indicador "informado".
COLUMNAS_NUMERICAS_INFORMADO: list[tuple[str, str]] = [
    ("superficie_cubierta", "superficie_cubierta_informado"),
    ("ambientes", "ambientes_informado"),
    ("dormitorios", "dormitorios_informado"),
    ("banos", "banos_informado"),
    ("antiguedad", "antiguedad_informado"),
    ("expensas_usd", "expensas_informado"),
]


class PropiedadEntrada(BaseModel):
    """Propiedad a valorar enviada a `/predict`."""

    tipo_propiedad: str = Field(description="Tipo de propiedad (ej.: departamento, casa, ph)")
    barrio: str = Field(description="Barrio de CABA (ej.: Palermo, Caballito)")

    superficie_cubierta: float | None = Field(default=None, ge=0)
    ambientes: float | None = Field(default=None, ge=0)
    dormitorios: float | None = Field(default=None, ge=0)
    banos: float | None = Field(default=None, ge=0)
    antiguedad: float | None = Field(default=None, ge=0)
    expensas_usd: float | None = Field(default=None, ge=0)

    superficie_cubierta_informado: int | None = Field(default=None, ge=0, le=1)
    ambientes_informado: int | None = Field(default=None, ge=0, le=1)
    dormitorios_informado: int | None = Field(default=None, ge=0, le=1)
    banos_informado: int | None = Field(default=None, ge=0, le=1)
    antiguedad_informado: int | None = Field(default=None, ge=0, le=1)
    expensas_informado: int | None = Field(default=None, ge=0, le=1)


class PrediccionSalida(BaseModel):
    """Resultado de la valoración de una propiedad."""

    precio_usd: float = Field(description="Precio estimado en dólares")
    log_precio_usd: float = Field(description="Precio estimado en escala logarítmica")
