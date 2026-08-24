"""
Esquema de la tabla `oportunidades` (Fase 12).

Define la tabla con SQLAlchemy Core (no ORM) y `crear_tablas`, que la crea si
no existe (idempotente). Las columnas repiten los nombres de variables ya
implementados en el pipeline (`precio_usd`, `precio_predicho_usd`,
`ratio_precio`, `clasificacion`, ...) para que el mapeo CSV -> DB -> API sea
directo. `id` es la PK (el id de la publicación en Argenprop).

`fecha_prediccion` se guarda como texto ISO (`AAAA-MM-DD`), igual que en el
CSV evaluado; `actualizado_en` es el timestamp de la última escritura.
"""

from __future__ import annotations

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    MetaData,
    String,
    Table,
    func,
)
from sqlalchemy.engine import Engine

TABLA_OPORTUNIDADES = "oportunidades"

_metadata = MetaData()

oportunidades = Table(
    TABLA_OPORTUNIDADES,
    _metadata,
    Column("id", String(64), primary_key=True),
    Column("titulo", String(512)),
    Column("link", String(512)),
    Column("barrio", String(128)),
    Column("tipo_propiedad", String(64)),
    Column("precio_usd", Float),
    Column("precio_predicho_usd", Float),
    Column("ratio_precio", Float),
    Column("diferencia_usd", Float),
    Column("diferencia_porcentual", Float),
    Column("clasificacion", String(32)),
    Column("modelo_version", String(64)),
    Column("fecha_prediccion", String(32)),
    Column("actualizado_en", DateTime(timezone=True), server_default=func.now()),
    # Raw features para explicabilidad SHAP por publicación
    Column("superficie_cubierta", Float),
    Column("ambientes", Float),
    Column("dormitorios", Float),
    Column("banos", Float),
    Column("antiguedad", Float),
    Column("expensas_usd", Float),
)


def crear_tablas(engine: Engine) -> None:
    """Crea la tabla `oportunidades` si no existe (idempotente)."""

    _metadata.create_all(engine)
