"""Test de integración del ETL periódico de oportunidades (Fase 12).

Ejecuta `ejecutar_etl` sobre un CSV de nuevas publicaciones sintético (formato
scraper), un bundle de serving entrenado en `tmp_path` y PostgreSQL (SQLite en
memoria). Verifica el dedup contra la base, la predicción + clasificación por
propiedad, la persistencia idempotente y el CSV de salida.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine

from real_estate.curation import transformations as tr
from real_estate.features import pipeline as pl
from real_estate.features import transformations as ftr
from real_estate.models import entrenamiento as en
from real_estate.persistencia.bundle import guardar_bundle
from real_estate.persistencia.config import ConfiguracionPostgres
from real_estate.persistencia.db import crear_engine
from real_estate.persistencia.esquema import crear_tablas
from real_estate.persistencia.etl_oportunidades import ejecutar_etl
from real_estate.persistencia.repositorio import (
    COLUMNAS_OPORTUNIDADES,
    ids_procesados,
    listar_oportunidades,
)

FECHA = "2026-07-15"
TASA = 1200.0

COLUMNAS_SCRAPER = [
    "id",
    "link",
    "titulo",
    "descripcion",
    "tipo_propiedad",
    "idtipopropiedad",
    "barrio",
    "sub_barrio",
    "precio",
    "moneda",
    "expensas",
    "superficie_cubierta",
    "superficie_semicubierta",
    "superficie_total",
    "ambientes",
    "dormitorios",
    "banos",
    "cocheras",
    "antiguedad",
    "fecha_scrape",
]

FILAS = [
    {
        "id": "1",
        "titulo": "Departamento en Palermo",
        "tipo_propiedad": "departamento",
        "barrio": "Palermo",
        "precio": "250000",
        "moneda": "USD",
        "superficie_cubierta": "120 m² cubie.",
        "ambientes": "3",
        "dormitorios": "2",
        "banos": "2",
        "antiguedad": "17 años",
        "fecha_scrape": "2026-07-15T10:00:00+00:00",
    },
    {
        "id": "2",
        "titulo": "Casa en Belgrano",
        "tipo_propiedad": "casa",
        "barrio": "Belgrano",
        "precio": "1200000",
        "moneda": "ARS",
        "superficie_cubierta": "300 m² cubie.",
        "ambientes": "5",
        "fecha_scrape": "2026-07-15T11:00:00+00:00",
    },
    {
        "id": "3",
        "titulo": "PH en Villa Crespo",
        "tipo_propiedad": "ph",
        "barrio": "Villa Crespo",
        "precio": "90000",
        "moneda": "USD",
        "fecha_scrape": "2026-07-15T12:00:00+00:00",
    },
]


def _df_curado_sintetico(n_filas: int = 96) -> pd.DataFrame:
    """Dataset curado sintético con señal de precio aprendible (igual al de serving)."""

    rng = np.random.default_rng(7)

    tipos = rng.choice(["departamento", "casa", "ph"], n_filas)
    barrios = rng.choice(["Palermo", "Caballito", "Belgrano", "Recoleta"], n_filas)
    superficie = rng.uniform(35, 180, n_filas)
    expensas = np.where(rng.random(n_filas) < 0.35, np.nan, rng.uniform(0, 400, n_filas))

    precio_base = np.where(
        tipos == "casa",
        150000.0,
        np.where(tipos == "ph", 30000.0, 80000.0),
    )
    precio_barrio = np.where(
        barrios == "Recoleta",
        90000.0,
        np.where(barrios == "Belgrano", 50000.0, 0.0),
    )
    precio = precio_base + precio_barrio + superficie * 1000 + rng.normal(0, 25000, n_filas)
    precio = np.clip(precio, 10000, None)

    return pd.DataFrame(
        {
            "id": np.arange(n_filas),
            "link": [f"link_{i}" for i in range(n_filas)],
            "titulo": [f"titulo_{i}" for i in range(n_filas)],
            "descripcion": [f"desc_{i}" for i in range(n_filas)],
            "tipo_propiedad": tipos,
            "idtipopropiedad": [1] * n_filas,
            "barrio": barrios,
            "sub_barrio": [None] * n_filas,
            "precio": precio,
            "moneda": "USD",
            "expensas": expensas,
            "superficie_cubierta": superficie,
            "superficie_semicubierta": [None] * n_filas,
            "superficie_total": [None] * n_filas,
            "ambientes": rng.integers(1, 6, n_filas).astype(float),
            "dormitorios": rng.integers(1, 5, n_filas).astype(float),
            "banos": rng.integers(1, 4, n_filas).astype(float),
            "cocheras": [None] * n_filas,
            "antiguedad": rng.integers(0, 50, n_filas).astype(float),
            "fecha_scrape": FECHA,
            "tipo_cambio_ars_usd": TASA,
            "precio_usd": precio,
            "expensas_usd": expensas / TASA,
            "expensas_informado": np.where(pd.isna(expensas), 0, 1),
            "superficie_cubierta_informado": 1,
            "superficie_semicubierta_informado": 0,
            "superficie_total_informado": 0,
            "ambientes_informado": 1,
            "dormitorios_informado": 1,
            "banos_informado": 1,
            "cocheras_informado": 0,
            "antiguedad_informado": 1,
        }
    )


@pytest.fixture
def sin_red(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evita cualquier consulta de tipo de cambio real."""

    monkeypatch.setattr(
        tr, "construir_tabla_tipo_cambio", lambda fechas, ruta_historico=None: {FECHA: TASA}
    )


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    """Entrena un modelo sobre datos sintéticos y guarda el bundle en `tmp_path`."""

    matriz = ftr.crear_target_log(ftr.seleccionar_columnas(_df_curado_sintetico()))
    train, val, test = pl.dividir_train_val_test(matriz, random_state=42)
    resultado = en.entrenar_y_evaluar(train, val, test, random_state=42)

    train_proc = en.aplicar_preprocesamiento(train, resultado.ajustes)
    columnas_features = list(en.separar_features_target(train_proc)[0].columns)

    guardar_bundle(
        directorio=tmp_path,
        modelo=resultado.modelo_xgboost,
        ajustes=resultado.ajustes,
        columnas_features=columnas_features,
        metadata={
            "metricas_xgboost_test": resultado.metricas_xgboost_test,
            "tipo_modelo": "xgboost",
            "fecha_exportacion": "2026-08-17T10:00:00+00:00",
        },
    )

    return tmp_path


@pytest.fixture
def motor() -> Engine:
    """Engine de SQLite en memoria con el esquema creado (compartido por el test)."""

    config = ConfiguracionPostgres(dsn="sqlite://")
    motor = crear_engine(config)
    crear_tablas(motor)
    return motor


def _escribir_csv_nuevas(tmp_path: Path, ids: list[str] | None = None) -> pd.DataFrame:
    """Escribe el CSV de nuevas publicaciones sintético y devuelve las filas."""

    filas = [fila for fila in FILAS if ids is None or fila["id"] in ids]
    df = pd.DataFrame([{col: fila.get(col, "") for col in COLUMNAS_SCRAPER} for fila in filas])
    df.to_csv(tmp_path / "nuevas.csv", index=False)
    return df


class TestEjecutarEtl:
    def test_persiste_clasifica_y_exporta(
        self, tmp_path: Path, sin_red: None, bundle: Path, motor: Engine
    ) -> None:
        _escribir_csv_nuevas(tmp_path)

        salida = ejecutar_etl(
            engine=motor,
            input_file=tmp_path / "nuevas.csv",
            output_file=tmp_path / "oportunidades.csv",
            directorio_modelo=bundle,
        )

        assert salida == tmp_path / "oportunidades.csv"
        assert salida.exists()

        filas = listar_oportunidades(motor, limit=100)
        assert len(filas) == 3
        assert ids_procesados(motor) == {"1", "2", "3"}

        # La clasificación por propiedad y el modelo_version quedaron persistidos.
        por_id = {fila["id"]: fila for fila in filas}
        assert por_id["1"]["clasificacion"] in {"buena_compra", "precio_justo", "mala_compra"}
        assert por_id["1"]["modelo_version"] == "xgboost-2026-08-17"
        assert por_id["1"]["fecha_prediccion"] is not None

        csv = pd.read_csv(salida, dtype={"id": str})
        assert list(csv.columns) == COLUMNAS_OPORTUNIDADES
        assert set(csv["id"]) == {"1", "2", "3"}
        assert csv["diferencia_usd"].notna().all()
        assert csv["diferencia_porcentual"].notna().all()

    def test_segunda_corrida_detecta_cero_nuevas(
        self, tmp_path: Path, sin_red: None, bundle: Path, motor: Engine
    ) -> None:
        _escribir_csv_nuevas(tmp_path)

        ejecutar_etl(
            engine=motor,
            input_file=tmp_path / "nuevas.csv",
            output_file=tmp_path / "oportunidades.csv",
            directorio_modelo=bundle,
        )
        ejecutar_etl(
            engine=motor,
            input_file=tmp_path / "nuevas.csv",
            output_file=tmp_path / "oportunidades.csv",
            directorio_modelo=bundle,
        )

        assert len(listar_oportunidades(motor, limit=100)) == 3

    def test_solo_procesa_ids_no_persistidos(
        self, tmp_path: Path, sin_red: None, bundle: Path, motor: Engine
    ) -> None:
        _escribir_csv_nuevas(tmp_path)

        ejecutar_etl(
            engine=motor,
            input_file=tmp_path / "nuevas.csv",
            output_file=tmp_path / "oportunidades.csv",
            directorio_modelo=bundle,
        )

        # Nueva corrida con una publicación extra: solo esa debe procesarse.
        _escribir_csv_nuevas(tmp_path, ids=["1", "2", "3"])
        extra = pd.DataFrame([{"id": "4", **{col: "" for col in COLUMNAS_SCRAPER if col != "id"}}])
        extra["tipo_propiedad"] = "departamento"
        extra["barrio"] = "Palermo"
        extra["precio"] = "100000"
        extra["moneda"] = "USD"
        extra["fecha_scrape"] = "2026-07-16T09:00:00+00:00"
        df_previo = pd.read_csv(tmp_path / "nuevas.csv", low_memory=False)
        pd.concat([df_previo, extra], ignore_index=True).to_csv(
            tmp_path / "nuevas.csv", index=False
        )

        ejecutar_etl(
            engine=motor,
            input_file=tmp_path / "nuevas.csv",
            output_file=tmp_path / "oportunidades.csv",
            directorio_modelo=bundle,
        )

        ids = ids_procesados(motor)
        assert "4" in ids
        assert len(ids) == 4

    def test_archivo_inexistente_lanza_file_not_found(
        self, tmp_path: Path, sin_red: None, bundle: Path, motor: Engine
    ) -> None:
        with pytest.raises(FileNotFoundError):
            ejecutar_etl(
                engine=motor,
                input_file=tmp_path / "no-existe.csv",
                output_file=tmp_path / "oportunidades.csv",
                directorio_modelo=bundle,
            )
