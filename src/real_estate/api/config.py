"""
Configuración del servicio de predicción (Fase 10).

`ConfiguracionServicio` lee la configuración de variables de entorno y de un
`.env` (con pydantic-settings). La única variable que necesita el servicio es
`MODELO_DIR`, la ruta al bundle de serving exportado por
`scripts/exportar_modelo.py`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Directorio por defecto del bundle de serving.
MODELO_DIR_DEFAULT = Path("models/modelo_precio_propiedades")


class ConfiguracionServicio(BaseSettings):
    """Configuración del servicio API vía variables de entorno / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    modelo_dir: Path = MODELO_DIR_DEFAULT
