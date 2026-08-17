"""
Persistencia en PostgreSQL de las oportunidades detectadas (Fase 12).

La tabla `oportunidades` guarda el resultado del ETL periódico: por cada
publicación nueva de Argenprop (CABA), el precio publicado en USD, la
predicción del modelo champion, las diferencias (absoluta y porcentual), la
clasificación de oportunidad y la versión del modelo. La API FastAPI lee de
esta tabla; el frontend futuro, a su vez, leerá de la API.
"""

from __future__ import annotations

from real_estate.persistencia.config import ConfiguracionPostgres
from real_estate.persistencia.db import crear_engine
from real_estate.persistencia.esquema import TABLA_OPORTUNIDADES, crear_tablas
from real_estate.persistencia.repositorio import (
    ids_procesados,
    listar_oportunidades,
    obtener_oportunidad,
    upsert_oportunidades,
)

__all__ = [
    "ConfiguracionPostgres",
    "TABLA_OPORTUNIDADES",
    "crear_engine",
    "crear_tablas",
    "ids_procesados",
    "listar_oportunidades",
    "obtener_oportunidad",
    "upsert_oportunidades",
]
