"""
Configuración de la base de datos PostgreSQL (Fase 12).

`ConfiguracionPostgres` lee la configuración de variables de entorno y de un
`.env` (con pydantic-settings). Variables:

- `POSTGRES_HOST` (default `localhost`)
- `POSTGRES_PORT` (default `5432`)
- `POSTGRES_USER` (default `realestate`)
- `POSTGRES_PASSWORD` (default `realestate`)
- `POSTGRES_DB` (default `real_estate`)
- `DSN`: si se define, tiene prioridad sobre el resto (útil para apuntar a
  otra base o para tests con SQLite en memoria).

El DSN resultante se expone con `dsn_efectivo`.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfiguracionPostgres(BaseSettings):
    """Configuración de conexión a PostgreSQL vía variables de entorno / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "realestate"
    postgres_password: str = "realestate"
    postgres_db: str = "real_estate"
    dsn: str | None = None

    @property
    def dsn_efectivo(self) -> str:
        """DSN de SQLAlchemy: el override `dsn` si está definido, o el armado
        desde las variables `POSTGRES_*` (driver psycopg)."""

        if self.dsn is not None:
            return self.dsn

        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
