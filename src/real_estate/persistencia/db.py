"""
Conexión a la base de datos (Fase 12).

`crear_engine` construye el `Engine` de SQLAlchemy a partir de una
`ConfiguracionPostgres`. Para PostgreSQL usa `pool_pre_ping` (revalida la
conexión antes de usarla); para SQLite en memoria (`sqlite://`) usa
`StaticPool` para que todos los hilos compartan la misma base (útil en tests).
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import StaticPool

from real_estate.persistencia.config import ConfiguracionPostgres


def crear_engine(config: ConfiguracionPostgres) -> Engine:
    """Crea el `Engine` de SQLAlchemy para el DSN de `config`."""

    dsn = config.dsn_efectivo

    if dsn.startswith("sqlite"):
        return create_engine(
            dsn,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

    return create_engine(dsn, pool_pre_ping=True)
