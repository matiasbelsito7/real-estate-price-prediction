"""Test de integración del pipeline de Data Curation.

Ejecuta `curar_csv` (leer CSV crudo -> curar -> escribir CSV) sobre un
dataset sintético que replica las columnas que produce el scraper, con el
tipo de cambio mockeado para no tocar la red.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from real_estate.curation import transformations as tr
from real_estate.curation.pipeline import curar_csv

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
        # USD: el precio no se convierte
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
        # Datos faltantes: se marca con los indicadores *_informado
        "id": "3",
        "titulo": "PH en Villa Crespo",
        "tipo_propiedad": "ph",
        "barrio": "Villa Crespo",
        "precio": "90000",
        "moneda": "USD",
        "fecha_scrape": "2026-07-15T12:00:00+00:00",
    },
]


@pytest.fixture
def csv_crudo(tmp_path: Path) -> Path:
    """Escribe el CSV de entrada sintético y devuelve su ruta."""
    ruta = tmp_path / "raw.csv"
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


def _leer_csv(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta, low_memory=False)


class TestCurarCsv:
    def test_genera_archivo_curado(self, csv_crudo: Path, tmp_path: Path, sin_red: None) -> None:
        salida = tmp_path / "curado.csv"

        curar_csv(csv_crudo, salida)

        assert salida.exists()
        df = _leer_csv(salida)
        assert len(df) == len(FILAS)

    def test_columnas_agregadas_por_la_curacion(
        self, csv_crudo: Path, tmp_path: Path, sin_red: None
    ) -> None:
        salida = tmp_path / "curado.csv"

        curar_csv(csv_crudo, salida)

        df = _leer_csv(salida)
        for columna in [
            "precio_usd",
            "tipo_cambio_ars_usd",
            "expensas_usd",
            "superficie_cubierta_informado",
            "ambientes_informado",
            "dormitorios_informado",
            "banos_informado",
            "antiguedad_informado",
        ]:
            assert columna in df.columns, f"Falta la columna {columna}"

    def test_conversion_de_moneda(self, csv_crudo: Path, tmp_path: Path, sin_red: None) -> None:
        salida = tmp_path / "curado.csv"

        curar_csv(csv_crudo, salida)

        df = _leer_csv(salida)
        # La fila USD conserva el precio
        assert df.loc[0, "precio_usd"] == pytest.approx(250000.0)
        # La fila ARS se convierte: 1.200.000 / 1200 = 1000 USD
        assert df.loc[1, "precio_usd"] == pytest.approx(1000.0)
        # La fila 3 quedó sin precio_usd (solo se deja NaN si la moneda es desconocida)
        assert pd.notna(df.loc[2, "precio_usd"])

    def test_indicadores_de_missing(self, csv_crudo: Path, tmp_path: Path, sin_red: None) -> None:
        salida = tmp_path / "curado.csv"

        curar_csv(csv_crudo, salida)

        df = _leer_csv(salida)
        # La fila 3 no informó superficie ni ambientes
        assert df.loc[2, "superficie_cubierta_informado"] == 0
        assert df.loc[2, "ambientes_informado"] == 0
        # La fila 1 sí
        assert df.loc[0, "superficie_cubierta_informado"] == 1
        assert df.loc[0, "ambientes_informado"] == 1

    def test_campos_textuales_convertidos_a_numericos(
        self, csv_crudo: Path, tmp_path: Path, sin_red: None
    ) -> None:
        salida = tmp_path / "curado.csv"

        curar_csv(csv_crudo, salida)

        df = _leer_csv(salida)
        assert df.loc[0, "superficie_cubierta"] == pytest.approx(120.0)
        assert df.loc[0, "ambientes"] == pytest.approx(3.0)
        assert df.loc[0, "dormitorios"] == pytest.approx(2.0)
        assert df.loc[0, "antiguedad"] == pytest.approx(17.0)
        assert df.loc[1, "expensas"] == pytest.approx(24_000.0)

    def test_csv_inexistente_lanza_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            curar_csv(tmp_path / "no-existe.csv", tmp_path / "out.csv")
