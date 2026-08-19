"""
Aplicación FastAPI de predicción de precios de propiedades (Fases 10 y 12).

Endpoints:

- `GET /health`: estado del servicio, versión y métricas del modelo.
- `POST /predict`: valoración de una propiedad (JSON) -> precio estimado.
- `GET /oportunidades`: listado paginado de las oportunidades que persiste el
  ETL periódico en PostgreSQL, filtrable por clasificación y barrio.
- `GET /oportunidades/{id}`: una oportunidad por su id.

El modelo (bundle de serving) se carga en el *lifespan* de la app desde
`MODELO_DIR` (ver `ConfiguracionServicio`). Si el bundle no existe, la app no
arranca y avisa con un mensaje claro.

La base PostgreSQL es opcional: `crear_app` acepta una `ConfiguracionPostgres`
y, si no se pasa, la lee de variables de entorno / `.env`. El engine se crea
perezosamente (no conecta al arrancar). Si no hay base configurada o es
inalcanzable, los endpoints de oportunidades responden 503; `/health` y
`/predict` no dependen de la base.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy.exc import OperationalError, ProgrammingError

from real_estate.api.config import ConfiguracionServicio
from real_estate.api.schemas import (
    COLUMNAS_NUMERICAS_INFORMADO,
    Oportunidad,
    PrediccionSalida,
    PropiedadEntrada,
)
from real_estate.persistencia import repositorio
from real_estate.persistencia.bundle import NOMBRE_MODELO, cargar_bundle
from real_estate.persistencia.config import ConfiguracionPostgres
from real_estate.persistencia.db import crear_engine
from real_estate.serving.modelo import ModeloPrediccion

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


def _base_no_disponible() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=(
            "Base de datos PostgreSQL no configurada o inalcanzable. "
            "Ejecutá primero el ETL periódico (scripts/etl_oportunidades.py)."
        ),
    )


def crear_app(
    config: ConfiguracionServicio | None = None,
    config_db: ConfiguracionPostgres | None = None,
) -> FastAPI:
    """Crea la app FastAPI con el bundle de modelo cargado en el *lifespan*.

    Si no se pasa `config`, la configuración del modelo se lee de variables de
    entorno / `.env`. El bundle debe existir (se genera con
    `python scripts/exportar_modelo.py`); si no, la app no arranca.

    `config_db` (opcional) configura la base PostgreSQL de oportunidades; si no
    se pasa, se lee de variables de entorno / `.env`. El engine se crea
    perezosamente: si la base no está disponible, los endpoints de
    oportunidades responden 503 sin tumbar la app.
    """

    config = config or ConfiguracionServicio()
    config_db = config_db or ConfiguracionPostgres()
    ruta_bundle = Path(config.modelo_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not (ruta_bundle / NOMBRE_MODELO).exists():
            raise RuntimeError(
                f"No se encontró el bundle de serving en '{ruta_bundle}'. "
                "Ejecutá primero: python scripts/exportar_modelo.py"
            )
        app.state.modelo = cargar_bundle(ruta_bundle)
        app.state.db_engine = crear_engine(config_db)
        yield

    app = FastAPI(
        title="Real Estate Price Prediction API",
        description=(
            "Predicción de precios de propiedades (Argenprop) - "
            "Predicción + oportunidades de compra (Fases 10 y 12)"
        ),
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

    @app.get("/oportunidades", response_model=list[Oportunidad])
    def listar_oportunidades(
        clasificacion: str | None = Query(default=None, description="Filtro por clasificación"),
        barrio: str | None = Query(default=None, description="Filtro por barrio"),
        limit: int = Query(default=50, ge=1, le=100, description="Cantidad máxima de filas"),
        offset: int = Query(default=0, ge=0, description="Desplazamiento para paginar"),
    ) -> list[dict[str, object]]:
        try:
            return repositorio.listar_oportunidades(
                app.state.db_engine,
                clasificacion=clasificacion,
                barrio=barrio,
                limit=limit,
                offset=offset,
            )
        except (OperationalError, ProgrammingError):
            raise _base_no_disponible() from None

    @app.get("/oportunidades/{id_publicacion}", response_model=Oportunidad)
    def obtener_oportunidad(id_publicacion: str) -> dict[str, object]:
        try:
            oportunidad = repositorio.obtener_oportunidad(app.state.db_engine, id_publicacion)
        except (OperationalError, ProgrammingError):
            raise _base_no_disponible() from None
        if oportunidad is None:
            raise HTTPException(
                status_code=404, detail=f"No existe la oportunidad {id_publicacion}"
            )
        return oportunidad

    return app


# Instancia por defecto para `uvicorn real_estate.api.app:app`.
app = crear_app()
