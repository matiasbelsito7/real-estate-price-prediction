"""Tests unitarios de `real_estate.curation.cleaning`.

Cubre `limpiar_numero` (formatos reales de Argenprop) y las funciones
de limpieza de columnas (`limpiar_columnas_numericas`, `limpiar_expensas`,
`preparar_fecha`).
"""

from __future__ import annotations

import pandas as pd
import pytest

from real_estate.curation.cleaning import (
    limpiar_columnas_numericas,
    limpiar_expensas,
    limpiar_numero,
    preparar_fecha,
)


class TestLimpiarNumero:
    """Casos de `limpiar_numero` con formatos reales del dataset crudo."""

    @pytest.mark.parametrize(
        ("entrada", "esperado"),
        [
            # Superficies
            ("300 m² cubie.", 300.0),
            ("90 m²", 90.0),
            ("12 m² semicub.", 12.0),
            ("40 m² total", 40.0),
            # Características
            ("2 dorm.", 2.0),
            ("3 ambientes", 3.0),
            ("1 baño", 1.0),
            ("17 años", 17.0),
            # Precios y expensas
            ("$250.000", 250000.0),
            ("&plus; $2.200.000\nexpensas", 2200000.0),
            ("&plus; $330.000\nexpensas", 330000.0),
            # Decimales y separadores de miles
            ("1.500,50", 1500.50),
            ("2.200.000", 2200000.0),
            ("250.000", 250000.0),
            ("1500,50", 1500.50),
            ("1.250,50", 1250.50),
        ],
    )
    def test_valores_validos(self, entrada: str, esperado: float) -> None:
        resultado = limpiar_numero(entrada)
        assert isinstance(resultado, float)
        assert resultado == esperado

    @pytest.mark.parametrize(
        "entrada",
        [
            "",
            "   ",
            "sin dato",
            "----",
        ],
    )
    def test_valores_vacios_o_invalidos(self, entrada: str) -> None:
        assert pd.isna(limpiar_numero(entrada))

    @pytest.mark.parametrize("entrada", [None, float("nan")])
    def test_nulos(self, entrada: object) -> None:
        assert pd.isna(limpiar_numero(entrada))

    def test_no_modifica_numeros(self) -> None:
        assert limpiar_numero(42) == 42.0


class TestLimpiezaColumnas:
    def test_limpia_solo_columnas_existentes(self) -> None:
        df = pd.DataFrame(
            {
                "precio": ["$250.000"],
                "superficie_cubierta": ["90 m²"],
                "ambientes": ["3 ambientes"],
            }
        )
        resultado = limpiar_columnas_numericas(df.copy())

        # Las columnas numéricas quedan float64
        assert resultado["precio"].dtype == "float64"
        assert resultado["superficie_cubierta"].dtype == "float64"
        assert resultado["ambientes"].dtype == "float64"
        assert resultado.loc[0, "precio"] == 250000.0
        assert resultado.loc[0, "superficie_cubierta"] == 90.0
        assert resultado.loc[0, "ambientes"] == 3.0

    def test_ignora_columnas_ausentes(self) -> None:
        df = pd.DataFrame({"otra_columna": ["x"]})
        resultado = limpiar_columnas_numericas(df.copy())
        assert list(resultado.columns) == ["otra_columna"]


class TestLimpiarExpensas:
    def test_convierte_expensas(self) -> None:
        df = pd.DataFrame({"expensas": ["&plus; $2.200.000\nexpensas"]})
        resultado = limpiar_expensas(df.copy())
        assert resultado["expensas"].dtype == "float64"
        assert resultado.loc[0, "expensas"] == 2200000.0

    def test_sin_columna_expensas_no_falla(self) -> None:
        df = pd.DataFrame({"precio": [1000]})
        resultado = limpiar_expensas(df.copy())
        assert list(resultado.columns) == ["precio"]


class TestPrepararFecha:
    def test_convierte_a_datetime_utc(self) -> None:
        df = pd.DataFrame({"fecha_scrape": ["2026-07-15T10:00:00+00:00"]})
        resultado = preparar_fecha(df.copy())
        assert pd.api.types.is_datetime64_any_dtype(resultado["fecha_scrape"])
        assert resultado.loc[0, "fecha_scrape"] == pd.Timestamp("2026-07-15 10:00:00+00:00")

    def test_fechas_invalidas_a_nat(self) -> None:
        df = pd.DataFrame({"fecha_scrape": ["no-es-fecha", None]})
        resultado = preparar_fecha(df.copy())
        assert resultado["fecha_scrape"].isna().all()

    def test_sin_columna_no_falla(self) -> None:
        df = pd.DataFrame({"precio": [1]})
        resultado = preparar_fecha(df.copy())
        assert list(resultado.columns) == ["precio"]
