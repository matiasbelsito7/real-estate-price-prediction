"""
Configuración centralizada del proyecto (pydantic-settings).

Carga los parámetros desde ``configs/config.yaml`` y permite sobrescribirlos
con variables de entorno / ``.env``.  Cada sección del YAML se modela como
un sub-modelo de pydantic, y el modelo raíz ``ConfiguracionProyecto``agrega
todos bajo un único抬头.

Uso típico::

    from real_estate.utils.config import ConfiguracionProyecto
    config = ConfiguracionProyecto()
    print(config.scraper.delay_min)
    print(config.models.xgboost.n_estimators)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Ruta al archivo YAML de configuración (relativa a la raíz del proyecto).
CONFIG_YAML = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def _cargar_yaml(ruta: Path) -> dict[str, Any]:
    """Carga un YAML y devuelve su contenido como dict.  Si no existe, devuelve {}."""
    if not ruta.exists():
        return {}
    with open(ruta, encoding="utf-8") as f:
        contenido = yaml.safe_load(f)
    return contenido if isinstance(contenido, dict) else {}


# ------------------------------------------------------------------
# Sub-modelos por sección
# ------------------------------------------------------------------


class ConfigScraper(BaseModel):
    """Parámetros del scraper de Argenprop."""

    delay_min: float = 1.0
    delay_max: float = 3.0
    timeout: int = 20
    reintentos_202: int = 5
    backoff_202_inicial: float = 15.0
    pausa_bloqueo: float = 300.0
    max_paginas: int | None = None


class ConfigCuration(BaseModel):
    """Parámetros de Data Curation."""

    fx_market: str = "blue"
    ruta_tipo_cambio_historico: str = "data/external/tipo_cambio_blue.csv"


class ConfigFeatures(BaseModel):
    """Parámetros de Feature Engineering."""

    precio_minimo_usd: float = 1000
    split_random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.5


class ConfigXGBoost(BaseModel):
    """Hiperparámetros por defecto de XGBoost."""

    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    reg_lambda: float = 1.0
    random_state: int = 42


class ConfigLasso(BaseModel):
    """Hiperparámetros de Lasso."""

    alpha: float = 1.0
    max_iter: int = 10000
    random_state: int = 42


class ConfigRidge(BaseModel):
    """Hiperparámetros de Ridge."""

    alpha: float = 1.0
    max_iter: int = 10000
    random_state: int = 42


class ConfigModels(BaseModel):
    """Configuración de modelos."""

    xgboost: ConfigXGBoost = Field(default_factory=ConfigXGBoost)
    lasso: ConfigLasso = Field(default_factory=ConfigLasso)
    ridge: ConfigRidge = Field(default_factory=ConfigRidge)


class ConfigTuning(BaseModel):
    """Parámetros de tuning de hiperparámetros."""

    n_iter: int = 30
    cv: int = 3
    metodo: str = "random"


class ConfigTracking(BaseModel):
    """Configuración de MLflow tracking."""

    experimento: str = "prediccion_precios_propiedades"
    modelo_registry: str = "modelo_precio_propiedades"
    uri: str = "mlruns"


class ConfigClasificacion(BaseModel):
    """Parámetros de clasificación de oportunidades."""

    umbral_precio_justo: float = 0.10


class ConfigServing(BaseModel):
    """Configuración del servicio de predicción."""

    modelo_dir: str = "models/modelo_precio_propiedades"


class ConfigPostgres(BaseModel):
    """Configuración de PostgreSQL."""

    host: str = "localhost"
    port: int = 5432
    user: str = "realestate"
    password: str = "realestate"
    db: str = "real_estate"


# ------------------------------------------------------------------
# Modelo raíz
# ------------------------------------------------------------------


class ConfiguracionProyecto(BaseSettings):
    """Configuración centralizada del proyecto.

    Carga desde ``configs/config.yaml`` y permite sobrescribir con
    variables de entorno / ``.env``.  Las variables de entorno usan el
    prefijo ``RE_`` (ej. ``RE_SCRAPER__DELAY_MIN=2``).
    """

    model_config = SettingsConfigDict(
        env_prefix="RE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    scraper: ConfigScraper = Field(default_factory=ConfigScraper)
    curation: ConfigCuration = Field(default_factory=ConfigCuration)
    features: ConfigFeatures = Field(default_factory=ConfigFeatures)
    models: ConfigModels = Field(default_factory=ConfigModels)
    tuning: ConfigTuning = Field(default_factory=ConfigTuning)
    tracking: ConfigTracking = Field(default_factory=ConfigTracking)
    clasificacion: ConfigClasificacion = Field(default_factory=ConfigClasificacion)
    serving: ConfigServing = Field(default_factory=ConfigServing)
    postgres: ConfigPostgres = Field(default_factory=ConfigPostgres)

    def __init__(self, **kwargs: Any) -> None:
        # Carga el YAML como valores por defecto (prioridad: env > yaml > defaults).
        yaml_vals = _cargar_yaml(CONFIG_YAML)
        # Fusiona los valores del YAML en kwargs solo si no están ya en env.
        for key, value in yaml_vals.items():
            if key not in kwargs:
                kwargs[key] = value
        super().__init__(**kwargs)
