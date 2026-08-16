"""
Aplicación FastAPI de predicción de precios de propiedades (Fase 10).

Expone dos endpoints:

- `GET /health`: estado del servicio, versión y métricas del modelo.
- `POST /predict`: valoración de una propiedad (JSON) -> precio estimado.

El modelo (bundle de serving) se carga en el *lifespan* de la app desde
`MODELO_DIR` (ver `ConfiguracionServicio`). Si el bundle no existe, la app no
arranca y avisa con un mensaje claro.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI

from real_estate.api.config import ConfiguracionServicio
from real_estate.api.schemas import (
    COLUMNAS_NUMERICAS_INFORMADO,
    PrediccionSalida,
    PropiedadEntrada,
)
from real_estate.serving.modelo import ModeloPrediccion
from real_estate.serving.persistencia import NOMBRE_MODELO, cargar_bundle

VERSION = "0.1.0"


def _entrada_a_dataframe(entrada: PropiedadEntrada) -> pd.DataFrame:
    """Convierte el payload en la fila (DataFrame) de entrada del modelo."""

    fila: dict[str, object] = {
        "tipo_propiedad": entrada.tipo_propiedad,
        "barrio": entrada.barrio,
    }

    for columna, indicador in COLUMNAS_NUMERICAS_INFORMADO:
        valor = getattr(entrada, columna)
        informado = getattr(entrada, indicador)
        fila[columna] = valor
        # Si no se informó el indicador, se deriva de la presencia del valor,
        # igual que en la etapa de features (0 = ausente, 1 = informado).
        fila[indicador] = informado if informado is not None else (1 if valor is not None else 0)

    return pd.DataFrame([fila])


def crear_app(config: ConfiguracionServicio | None = None) -> FastAPI:
    """Crea la app FastAPI con el bundle de modelo cargado en el *lifespan*.

    Si no se pasa `config`, la configuración se lee de variables de entorno /
    `.env`. El bundle debe existir (se genera con
    `python scripts/exportar_modelo.py`); si no, la app no arranca.
    """

    config = config or ConfiguracionServicio()
    ruta_bundle = Path(config.modelo_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not (ruta_bundle / NOMBRE_MODELO).exists():
            raise RuntimeError(
                f"No se encontró el bundle de serving en '{ruta_bundle}'. "
                "Ejecutá primero: python scripts/exportar_modelo.py"
            )
        app.state.modelo = cargar_bundle(ruta_bundle)
        yield

    app = FastAPI(
        title="Real Estate Price Prediction API",
        description="Predicción de precios de propiedades (Argenprop) - Fase 10",
        version=VERSION,
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        modelo: ModeloPrediccion = app.state.modelo
        return {
            "estado": "ok",
            "modelo": "xgboost",
            "version": app.version,
            "metricas_xgboost_test": modelo.metadata.get("metricas_xgboost_test"),
        }

    @app.post("/predict", response_model=PrediccionSalida)
    def predecir(entrada: PropiedadEntrada) -> PrediccionSalida:
        modelo: ModeloPrediccion = app.state.modelo
        df = _entrada_a_dataframe(entrada)

        log_precio_usd = float(modelo.predecir_log(df)[0])
        precio_usd = float(np.exp(log_precio_usd))

        return PrediccionSalida(precio_usd=precio_usd, log_precio_usd=log_precio_usd)

    return app


# Instancia por defecto para `uvicorn real_estate.api.app:app`.
app = crear_app()
