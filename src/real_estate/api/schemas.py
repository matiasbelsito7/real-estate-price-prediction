"""
Schemas Pydantic de la API (Fases 10 y 12).

`PropiedadEntrada` modela el payload que recibe `/predict`. Los indicadores
`*_informado` (6 columnas que marcan si el dato numérico fue informado o se
imputó) se derivan automáticamente de la presencia del valor si no se envían
explícitamente, replicando la lógica de la etapa de features.

`Oportunidad` modela una fila de la tabla `oportunidades` (la que persiste el
ETL periódico y exponen `/oportunidades` y `/oportunidades/{id}`).
"""

from __future__ import annotations

from datetime import datetime

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


class Oportunidad(BaseModel):
    """Oportunidad de compra persistida por el ETL periódico (Fase 12)."""

    id: str = Field(description="Id de la publicación en Argenprop")
    titulo: str | None = Field(default=None, description="Título del aviso")
    link: str | None = Field(default=None, description="URL de la publicación")
    barrio: str | None = Field(default=None, description="Barrio de CABA")
    tipo_propiedad: str | None = Field(default=None, description="Tipo de propiedad")
    precio_usd: float | None = Field(default=None, description="Precio publicado en USD")
    precio_predicho_usd: float | None = Field(
        default=None, description="Precio estimado por el modelo en USD"
    )
    ratio_precio: float | None = Field(
        default=None, description="Ratio predicho/publicado del ranking de ofertas"
    )
    diferencia_usd: float | None = Field(
        default=None, description="Diferencia absoluta predicho - publicado (USD)"
    )
    diferencia_porcentual: float | None = Field(
        default=None, description="Diferencia porcentual vs. el precio publicado"
    )
    clasificacion: str | None = Field(
        default=None, description="buena_compra | precio_justo | mala_compra | sin_clasificar"
    )
    modelo_version: str | None = Field(default=None, description="Versión del modelo que predijo")
    fecha_prediccion: str | None = Field(default=None, description="Fecha de la predicción (ISO)")
    actualizado_en: datetime | None = Field(default=None, description="Última actualización (ISO)")
