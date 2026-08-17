"""
Repositorio de oportunidades sobre PostgreSQL (Fase 12).

Operaciones sobre la tabla `oportunidades` usando SQLAlchemy Core:

- `ids_procesados`: ids ya persistidos (clave del dedup "solo nuevas").
- `upsert_oportunidades`: inserta o actualiza (ON CONFLICT por `id`).
- `listar_oportunidades`: paginado, con filtros opcionales de clasificación y
  barrio, ordenado de mejor oportunidad a peor (diferencia porcentual
  descendente).
- `obtener_oportunidad`: una publicación por su id.

Los valores `NaN`/`inf` de pandas se traducen a `NULL` (no son serializables
a PostgreSQL); las filas sin `id` válido se descartan.
"""

from __future__ import annotations

import math

import pandas as pd
from sqlalchemy import Engine, select
from sqlalchemy.dialects.postgresql import insert as _insert_postgres
from sqlalchemy.dialects.sqlite import insert as _insert_sqlite

from real_estate.persistencia.esquema import oportunidades

#: Columnas de una oportunidad que se persisten (los nombres ya existen en el
#: pipeline y coinciden con las del CSV que exporta el ETL periódico).
COLUMNAS_OPORTUNIDADES = [
    "id",
    "titulo",
    "link",
    "barrio",
    "tipo_propiedad",
    "precio_usd",
    "precio_predicho_usd",
    "ratio_precio",
    "diferencia_usd",
    "diferencia_porcentual",
    "clasificacion",
    "fecha_prediccion",
]


def _valor_db(valor: object) -> object:
    """Traduce un valor de pandas a un valor serializable por SQLAlchemy
    (`NaN`/`inf`/`-inf` -> `None`)."""

    if isinstance(valor, float) and not math.isfinite(valor):
        return None
    return valor


def _filas_para_tabla(df: pd.DataFrame, modelo_version: str) -> list[dict[str, object]]:
    """Convierte el DataFrame evaluado en filas de la tabla (con su versión
    de modelo). Descarta filas sin `id` válido."""

    filas: list[dict[str, object]] = []

    for fila in df.to_dict(orient="records"):
        id_publicacion = fila.get("id")
        if id_publicacion is None or (
            isinstance(id_publicacion, float) and not math.isfinite(id_publicacion)
        ):
            continue

        fila_db = {columna: _valor_db(fila.get(columna)) for columna in COLUMNAS_OPORTUNIDADES}
        fila_db["modelo_version"] = modelo_version
        filas.append(fila_db)

    return filas


def _upsert(engine: Engine, filas: list[dict[str, object]]) -> None:
    """Inserta o actualiza las filas (ON CONFLICT por `id`)."""

    if not filas:
        return

    insert = _insert_sqlite if engine.dialect.name == "sqlite" else _insert_postgres

    stmt = insert(oportunidades).values(filas)
    set_ = {
        col.name: getattr(stmt.excluded, col.name) for col in oportunidades.c if col.name != "id"
    }
    stmt = stmt.on_conflict_do_update(index_elements=[oportunidades.c.id], set_=set_)

    with engine.begin() as conexion:
        conexion.execute(stmt)


def upsert_oportunidades(
    engine: Engine,
    df: pd.DataFrame,
    *,
    modelo_version: str,
) -> int:
    """Persiste las oportunidades del DataFrame y devuelve la cantidad de filas
    insertadas/actualizadas. Idempotente: re-persistir el mismo `id` lo actualiza."""

    filas = _filas_para_tabla(df, modelo_version)
    _upsert(engine, filas)
    return len(filas)


def ids_procesados(engine: Engine) -> set[str]:
    """Ids de publicaciones ya persistidos (para el dedup de "solo nuevas")."""

    stmt = select(oportunidades.c.id)
    with engine.connect() as conexion:
        return {str(fila[0]) for fila in conexion.execute(stmt)}


def listar_oportunidades(
    engine: Engine,
    *,
    clasificacion: str | None = None,
    barrio: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, object]]:
    """Lista las oportunidades, de mejor a peor (diferencia porcentual
    descendente), con filtros opcionales y paginado."""

    stmt = select(oportunidades).order_by(oportunidades.c.diferencia_porcentual.desc().nullslast())

    if clasificacion is not None:
        stmt = stmt.where(oportunidades.c.clasificacion == clasificacion)
    if barrio is not None:
        stmt = stmt.where(oportunidades.c.barrio == barrio)

    stmt = stmt.limit(limit).offset(offset)

    with engine.connect() as conexion:
        filas = conexion.execute(stmt).mappings().all()

    return [dict(fila) for fila in filas]


def obtener_oportunidad(engine: Engine, id_publicacion: str) -> dict[str, object] | None:
    """Devuelve una oportunidad por su id, o `None` si no existe."""

    stmt = select(oportunidades).where(oportunidades.c.id == id_publicacion)

    with engine.connect() as conexion:
        fila = conexion.execute(stmt).mappings().first()

    return dict(fila) if fila is not None else None
