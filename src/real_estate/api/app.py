"""
Aplicación FastAPI de predicción de precios de propiedades (Fases 10, 12 y Frontend).

Endpoints:

- `GET /health`: estado del servicio, versión y métricas del modelo.
- `POST /predict`: valoración de una propiedad (JSON) -> precio estimado.
- `GET /oportunidades`: listado paginado de las oportunidades que persiste el
  ETL periódico en PostgreSQL, filtrable por clasificación y barrio.
- `GET /oportunidades/{id}`: una oportunidad por su id.
- `GET /oportunidades/{id}/explain`: explicabilidad SHAP de una publicación.

El modelo (bundle de serving) se carga en el *lifespan* de la app desde
`MODELO_DIR` (ver `ConfiguracionServicio`). Si el bundle no existe, la app no
arranca y avisa con un mensaje claro.

La base PostgreSQL es opcional: `crear_app` acepta una `ConfiguracionPostgres`
y, si no se pasa, la lee de variables de entorno / `.env`. El engine se crea
perezosamente (no conecta al arrancar). Si no hay base configurada o es
inalcanzable, los endpoints de oportunidades responden 503; `/health` y
`/predict` no dependen de la base.

El frontend se sirve como archivos estáticos desde `/frontend/`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, ProgrammingError
from starlette.requests import Request

from real_estate.api.config import ConfiguracionServicio
from real_estate.api.schemas import (
    COLUMNAS_NUMERICAS_INFORMADO,
    ExplicacionFeature,
    ExplicacionPublicacion,
    Oportunidad,
    PrediccionSalida,
    PropiedadEntrada,
)
from real_estate.persistencia import repositorio
from real_estate.persistencia.bundle import NOMBRE_MODELO, cargar_bundle
from real_estate.persistencia.config import ConfiguracionPostgres
from real_estate.persistencia.db import crear_engine
from real_estate.serving.modelo import ModeloPrediccion

VERSION = "0.2.0"

logger = logging.getLogger(__name__)

TIPOS_PROPIEDAD_CONOCIDOS: set[str] = {
    "departamento",
    "casa",
    "ph",
    "terreno",
    "local",
    "oficina",
    "cochera",
}

# Columnas raw features que se usan para SHAP (orden del modelo)
RAW_FEATURES = [
    "superficie_cubierta",
    "ambientes",
    "dormitorios",
    "banos",
    "antiguedad",
    "expensas_usd",
]

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


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
        fila[indicador] = informado if informado is not None else (1 if valor is not None else 0)

    return pd.DataFrame([fila])


def _oportunidad_a_dataframe(oportunidad: dict[str, object]) -> pd.DataFrame:
    """Convierte una oportunidad de la DB en DataFrame para predecir."""

    fila: dict[str, object] = {
        "tipo_propiedad": oportunidad.get("tipo_propiedad", ""),
        "barrio": oportunidad.get("barrio", ""),
    }

    # Agregar raw features y sus indicadores _informado (usar mapping de schemas)
    from real_estate.api.schemas import COLUMNAS_NUMERICAS_INFORMADO

    for columna, indicador in COLUMNAS_NUMERICAS_INFORMADO:
        valor = oportunidad.get(columna)
        fila[columna] = valor
        fila[indicador] = (
            1 if valor is not None and not (isinstance(valor, float) and math.isnan(valor)) else 0
        )

    # Agregar columnas faltantes que el modelo espera (serán descartadas por seleccionar_columnas)
    for col in [
        "cocheras",
        "superficie_semicubierta",
        "superficie_total",
        "sub_barrio",
        "link",
        "titulo",
        "descripcion",
        "fecha_scrape",
        "id",
        "idtipopropiedad",
        "precio",
        "moneda",
        "tipo_cambio_ars_usd",
        "expensas",
        "cocheras_informado",
        "superficie_semicubierta_informado",
        "superficie_total_informado",
    ]:
        if col not in fila:
            fila[col] = None

    return pd.DataFrame([fila])


def _construir_explicacion(
    modelo: ModeloPrediccion,
    oportunidad: dict[str, object],
) -> ExplicacionPublicacion:
    """Calcula los valores SHAP para una publicación."""

    from real_estate.explainability.shap_analysis import calcular_shap

    df = _oportunidad_a_dataframe(oportunidad)

    # Predecir
    log_precio = float(modelo.predecir_log(df)[0])
    precio_usd = float(np.exp(log_precio))

    # Calcular SHAP
    x_matriz = modelo._construir_matriz(df)
    explicacion = calcular_shap(modelo.modelo_xgboost, x_matriz)

    # Extraer contribuciones de la primera (única) fila
    valores_shap = explicacion.valores[0]
    base = explicacion.base

    features = []
    for i, nombre in enumerate(explicacion.nombres):
        valor_raw = x_matriz.iloc[0, i]
        valor = float(str(valor_raw)) if valor_raw is not None else 0.0
        contrib = float(valores_shap[i])
        contrib_usd = float(np.exp(contrib) - 1) * 100  # % de impacto
        features.append(
            ExplicacionFeature(
                nombre=nombre,
                valor=valor,
                contribucion=round(contrib, 4),
                contribucion_usd=round(contrib_usd, 1),
            )
        )

    # Ordenar por importancia absoluta
    features.sort(key=lambda f: abs(f.contribucion), reverse=True)

    # Generar resumen
    top_positivas = [f for f in features if f.contribucion > 0][:3]
    top_negativas = [f for f in features if f.contribucion < 0][:3]

    partes = []
    if top_positivas:
        positivos = ", ".join(f"{f.nombre} (+{f.contribucion_usd}%)" for f in top_positivas)
        partes.append(f"Aumenta el precio: {positivos}")
    if top_negativas:
        negativos = ", ".join(f"{f.nombre} ({f.contribucion_usd}%)" for f in top_negativas)
        partes.append(f"Reduce el precio: {negativos}")

    resumen = "; ".join(partes) if partes else "Sin contribuciones significativas"

    return ExplicacionPublicacion(
        id=str(oportunidad.get("id", "")),
        precio_predicho_usd=round(precio_usd, 2),
        base_shap=round(base, 4),
        features=features,
        resumen=resumen,
    )


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
            "Predicción + oportunidades de compra + Frontend (Fases 10, 12, Frontend)"
        ),
        version=VERSION,
        lifespan=lifespan,
    )

    # Rate limiting: 60 requests/min por IP en /predict, libre en /health.
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter

    # Servir frontend como archivos estáticos
    if FRONTEND_DIR.exists():
        app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

    @app.get("/")
    async def root() -> FileResponse:
        """Redirige al frontend."""
        index_path = FRONTEND_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        raise HTTPException(status_code=404, detail="Frontend no encontrado")

    @app.get("/health")
    def health() -> dict[str, object]:
        modelo: ModeloPrediccion = app.state.modelo

        # Check de base de datos (perezoso: solo consulta si el engine existe).
        db_status = "no_configurado"
        engine = getattr(app.state, "db_engine", None)
        if engine is not None:
            try:
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                db_status = "ok"
            except Exception:
                db_status = "no_disponible"

        return {
            "estado": "ok",
            "modelo": "xgboost",
            "version": app.version,
            "base_datos": db_status,
            "metricas_xgboost_test": modelo.metadata.get("metricas_xgboost_test"),
        }

    @app.post("/predict", response_model=PrediccionSalida)
    @limiter.limit("60/minute")
    def predecir(request: Request, entrada: PropiedadEntrada) -> PrediccionSalida:
        modelo: ModeloPrediccion = app.state.modelo

        if entrada.tipo_propiedad.lower() not in TIPOS_PROPIEDAD_CONOCIDOS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Tipo de propiedad '{entrada.tipo_propiedad}' no reconocido. "
                    f"Valores válidos: {', '.join(sorted(TIPOS_PROPIEDAD_CONOCIDOS))}"
                ),
            )

        if entrada.barrio:
            logger.warning(
                "Barrio '%s' no fue validado contra el set de entrenamiento", entrada.barrio
            )

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

    @app.get("/oportunidades/{id_publicacion}/explain", response_model=ExplicacionPublicacion)
    def explicar_oportunidad(id_publicacion: str) -> ExplicacionPublicacion:
        """Explica por qué el modelo predijo ese precio para la publicación."""
        modelo: ModeloPrediccion = app.state.modelo

        try:
            oportunidad = repositorio.obtener_oportunidad(app.state.db_engine, id_publicacion)
        except (OperationalError, ProgrammingError):
            raise _base_no_disponible() from None

        if oportunidad is None:
            raise HTTPException(
                status_code=404, detail=f"No existe la oportunidad {id_publicacion}"
            )

        try:
            return _construir_explicacion(modelo, oportunidad)
        except Exception as exc:
            logger.error("Error calculando explicación para %s: %s", id_publicacion, exc)
            raise HTTPException(
                status_code=500,
                detail=f"Error calculando explicación: {exc}",
            ) from exc

    return app


# Instancia por defecto para `uvicorn real_estate.api.app:app`.
app = crear_app()
