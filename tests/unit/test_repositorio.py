"""Tests unitarios de la capa de persistencia de oportunidades (Fase 12).

Usan SQLite en memoria (un solo `Engine` compartido por test): cubren el
upsert idempotente, el dedup por ids, el listado con filtros y paginado, y la
traducción de `NaN`/`inf` a `NULL`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine

from real_estate.persistencia.config import ConfiguracionPostgres
from real_estate.persistencia.db import crear_engine
from real_estate.persistencia.esquema import crear_tablas
from real_estate.persistencia.repositorio import (
    ids_procesados,
    listar_oportunidades,
    obtener_oportunidad,
    upsert_oportunidades,
)

FILAS_BASE = pd.DataFrame(
    [
        {
            "id": "1",
            "titulo": "Depto Palermo",
            "link": "link_1",
            "barrio": "Palermo",
            "tipo_propiedad": "departamento",
            "precio_usd": 250000.0,
            "precio_predicho_usd": 200000.0,
            "ratio_precio": 0.8,
            "diferencia_usd": -50000.0,
            "diferencia_porcentual": -20.0,
            "clasificacion": "mala_compra",
            "fecha_prediccion": "2026-08-17",
        },
        {
            "id": "2",
            "titulo": "Casa Belgrano",
            "link": "link_2",
            "barrio": "Belgrano",
            "tipo_propiedad": "casa",
            "precio_usd": 100000.0,
            "precio_predicho_usd": 140000.0,
            "ratio_precio": 1.4,
            "diferencia_usd": 40000.0,
            "diferencia_porcentual": 40.0,
            "clasificacion": "buena_compra",
            "fecha_prediccion": "2026-08-17",
        },
    ]
)


@pytest.fixture
def engine() -> Iterator[Engine]:
    """Engine de SQLite en memoria con el esquema creado."""

    config = ConfiguracionPostgres(dsn="sqlite://")
    motor = crear_engine(config)
    crear_tablas(motor)
    yield motor


class TestUpsert:
    def test_persiste_y_lee_ids(self, engine: Engine) -> None:
        upsert_oportunidades(engine, FILAS_BASE, modelo_version="xgboost-test")

        assert ids_procesados(engine) == {"1", "2"}

    def test_es_idempotente_ante_el_mismo_id(self, engine: Engine) -> None:
        upsert_oportunidades(engine, FILAS_BASE, modelo_version="xgboost-test")
        upsert_oportunidades(engine, FILAS_BASE, modelo_version="xgboost-test")

        assert len(listar_oportunidades(engine, limit=100)) == 2

    def test_reupsert_actualiza_la_fila(self, engine: Engine) -> None:
        upsert_oportunidades(engine, FILAS_BASE, modelo_version="xgboost-test")
        actualizada = FILAS_BASE.copy()
        actualizada.loc[0, "clasificacion"] = "precio_justo"
        actualizada.loc[0, "diferencia_porcentual"] = 0.0

        upsert_oportunidades(engine, actualizada, modelo_version="xgboost-v2")

        fila = obtener_oportunidad(engine, "1")
        assert fila is not None
        assert fila["clasificacion"] == "precio_justo"
        assert fila["modelo_version"] == "xgboost-v2"
        assert len(listar_oportunidades(engine, limit=100)) == 2

    def test_nan_e_inf_se_guardan_como_null(self, engine: Engine) -> None:
        filas = FILAS_BASE.copy()
        filas.loc[0, "diferencia_usd"] = np.nan
        filas.loc[1, "diferencia_porcentual"] = math.inf

        upsert_oportunidades(engine, filas, modelo_version="xgboost-test")

        fila_1 = obtener_oportunidad(engine, "1")
        assert fila_1 is not None
        assert fila_1["diferencia_usd"] is None
        fila_2 = obtener_oportunidad(engine, "2")
        assert fila_2 is not None
        assert fila_2["diferencia_porcentual"] is None

    def test_descarta_filas_sin_id_valido(self, engine: Engine) -> None:
        filas = FILAS_BASE.copy()
        filas.loc[len(filas)] = {
            **FILAS_BASE.loc[0].to_dict(),
            "id": np.nan,
            "titulo": "sin id",
        }

        cantidad = upsert_oportunidades(engine, filas, modelo_version="xgboost-test")

        assert cantidad == 2
        assert ids_procesados(engine) == {"1", "2"}


class TestListar:
    def test_ordena_por_diferencia_porcentual_descendente(self, engine: Engine) -> None:
        upsert_oportunidades(engine, FILAS_BASE, modelo_version="xgboost-test")

        ids = [fila["id"] for fila in listar_oportunidades(engine)]

        assert ids == ["2", "1"]

    def test_filtra_por_clasificacion(self, engine: Engine) -> None:
        upsert_oportunidades(engine, FILAS_BASE, modelo_version="xgboost-test")

        ids = [fila["id"] for fila in listar_oportunidades(engine, clasificacion="mala_compra")]

        assert ids == ["1"]

    def test_filtra_por_barrio(self, engine: Engine) -> None:
        upsert_oportunidades(engine, FILAS_BASE, modelo_version="xgboost-test")

        ids = [fila["id"] for fila in listar_oportunidades(engine, barrio="Belgrano")]

        assert ids == ["2"]

    def test_pagina_con_limit_y_offset(self, engine: Engine) -> None:
        upsert_oportunidades(engine, FILAS_BASE, modelo_version="xgboost-test")

        pagina = listar_oportunidades(engine, limit=1, offset=1)

        assert [fila["id"] for fila in pagina] == ["1"]


class TestObtener:
    def test_devuelve_la_oportunidad_por_id(self, engine: Engine) -> None:
        upsert_oportunidades(engine, FILAS_BASE, modelo_version="xgboost-test")

        fila = obtener_oportunidad(engine, "1")

        assert fila is not None
        assert fila["id"] == "1"
        assert fila["precio_usd"] == pytest.approx(250000.0)
        assert fila["modelo_version"] == "xgboost-test"

    def test_id_inexistente_devuelve_none(self, engine: Engine) -> None:
        upsert_oportunidades(engine, FILAS_BASE, modelo_version="xgboost-test")

        assert obtener_oportunidad(engine, "999") is None
