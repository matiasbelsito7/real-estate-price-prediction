"""Test de integración de `real_estate.serving.evaluar` (Fase 2 del roadmap).

Ejecuta `evaluar_nuevas` (curar CSV crudo de nuevas -> predecir con el bundle
de serving -> escribir CSV evaluado) sobre un dataset sintético que replica las
columnas que produce el scraper, con el tipo de cambio mockeado (sin red) y un
bundle entrenado sobre datos sintéticos en `tmp_path`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from real_estate.curation import transformations as tr
from real_estate.features import pipeline as pl
from real_estate.features import transformations as ftr
from real_estate.models import entrenamiento as en
from real_estate.serving.evaluar import COLUMNAS_SALIDA, evaluar_nuevas
from real_estate.serving.persistencia import guardar_bundle

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
        # USD: el precio publicado no se convierte
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
        # ARS: se convierte con el tipo de cambio
        "id": "2",
        "titulo": "Casa en Belgrano",
        "tipo_propiedad": "casa",
        "barrio": "Belgrano",
        "precio": "1200000",
        "moneda": "ARS",
        "expensas": "&plus; $24.000\nexpensas",
        "superficie_cubierta": "300 m² cubie.",
        "ambientes": "5",
        "fecha_scrape": "2026-07-15T11:00:00+00:00",
    },
    {
        # Datos faltantes: se imputan y se marca con los indicadores *_informado
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
def csv_nuevas(tmp_path: Path) -> Path:
    """Escribe el CSV de nuevas publicaciones sintético y devuelve su ruta."""
    ruta = tmp_path / "nuevas.csv"
    pd.DataFrame([{col: fila.get(col, "") for col in COLUMNAS_SCRAPER} for fila in FILAS]).to_csv(
        ruta, index=False
    )
    return ruta


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
        metadata={"metricas_xgboost_test": resultado.metricas_xgboost_test},
    )

    return tmp_path


def _leer_csv(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta, low_memory=False)


class TestEvaluarNuevas:
    def test_genera_archivo_evaluado(
        self, csv_nuevas: Path, tmp_path: Path, sin_red: None, bundle: Path
    ) -> None:
        salida = tmp_path / "evaluadas.csv"

        evaluar_nuevas(csv_nuevas, salida, directorio_modelo=bundle)

        assert salida.exists()
        df = _leer_csv(salida)
        assert len(df) == len(FILAS)

    def test_columnas_salida(
        self, csv_nuevas: Path, tmp_path: Path, sin_red: None, bundle: Path
    ) -> None:
        salida = tmp_path / "evaluadas.csv"

        evaluar_nuevas(csv_nuevas, salida, directorio_modelo=bundle)

        df = _leer_csv(salida)
        for columna in COLUMNAS_SALIDA:
            assert columna in df.columns, f"Falta la columna {columna}"
        for columna in ["precio_usd", "precio_predicho_usd", "fecha_prediccion"]:
            assert columna in df.columns, f"Falta la columna {columna}"

    def test_predicciones_positivas_y_finitas(
        self, csv_nuevas: Path, tmp_path: Path, sin_red: None, bundle: Path
    ) -> None:
        salida = tmp_path / "evaluadas.csv"

        evaluar_nuevas(csv_nuevas, salida, directorio_modelo=bundle)

        df = _leer_csv(salida)
        predicciones = df["precio_predicho_usd"].astype(float)
        assert predicciones.notna().all()
        assert (predicciones > 0).all()
        assert np.isfinite(predicciones).all()

    def test_conserva_precio_publicado_usd(
        self, csv_nuevas: Path, tmp_path: Path, sin_red: None, bundle: Path
    ) -> None:
        salida = tmp_path / "evaluadas.csv"

        evaluar_nuevas(csv_nuevas, salida, directorio_modelo=bundle)

        df = _leer_csv(salida)
        # La fila USD conserva el precio publicado
        assert df.loc[0, "precio_usd"] == pytest.approx(250000.0)
        # La fila ARS se convierte: 1.200.000 / 1200 = 1000 USD
        assert df.loc[1, "precio_usd"] == pytest.approx(1000.0)
        # La fila 3 no informó superficie pero igual se predice su precio
        assert df["precio_predicho_usd"].astype(float).iloc[2] > 0

    def test_fecha_prediccion_es_hoy(
        self, csv_nuevas: Path, tmp_path: Path, sin_red: None, bundle: Path
    ) -> None:
        salida = tmp_path / "evaluadas.csv"

        evaluar_nuevas(csv_nuevas, salida, directorio_modelo=bundle)

        df = _leer_csv(salida)
        hoy = datetime.now(UTC).date().isoformat()
        assert (df["fecha_prediccion"] == hoy).all()

    def test_csv_inexistente_lanza_file_not_found(self, tmp_path: Path, bundle: Path) -> None:
        with pytest.raises(FileNotFoundError):
            evaluar_nuevas(
                tmp_path / "no-existe.csv",
                tmp_path / "out.csv",
                directorio_modelo=bundle,
            )
