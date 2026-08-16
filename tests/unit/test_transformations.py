"""Tests unitarios de `real_estate.curation.transformations`.

Todas las funciones que tocan la red (`obtener_tipo_cambio`,
`construir_tabla_tipo_cambio`, `normalizar_moneda`) se prueban con el
módulo `requests` mockeado: los tests no hacen requests reales.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from real_estate.curation import transformations as tr

FECHA = "2026-07-15"
TASA = 1200.0


class FakeResponse:
    """Respuesta mínima de requests con status_code y json()."""

    def __init__(self, status_code: int, data: object) -> None:
        self.status_code = status_code
        self._data = data

    def json(self) -> object:
        return self._data


class TestObtenerTipoCambio:
    def test_respuesta_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = SimpleNamespace(get=lambda url, timeout: FakeResponse(200, {"venta": TASA}))
        monkeypatch.setattr(tr, "requests", fake)
        assert tr.obtener_tipo_cambio(FECHA) == TASA

    def test_respuesta_lista(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = SimpleNamespace(get=lambda url, timeout: FakeResponse(200, [{"venta": TASA}]))
        monkeypatch.setattr(tr, "requests", fake)
        assert tr.obtener_tipo_cambio(FECHA) == TASA

    def test_respuesta_sin_venta_devuelve_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = SimpleNamespace(get=lambda url, timeout: FakeResponse(200, {"compra": 1150.0}))
        monkeypatch.setattr(tr, "requests", fake)
        assert tr.obtener_tipo_cambio(FECHA, max_intentos=2) is None

    def test_error_de_red_devuelve_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def get_fallido(url: str, timeout: float) -> FakeResponse:
            raise requests.Timeout("timeout")

        fake = SimpleNamespace(get=get_fallido, RequestException=requests.RequestException)
        monkeypatch.setattr(tr, "requests", fake)
        assert tr.obtener_tipo_cambio(FECHA, max_intentos=2) is None

    def test_retrocede_un_dia_si_no_hay_cotizacion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # El domingo 2026-07-12 no tiene cotización; el sábado 2026-07-11 sí.
        def get_con_sabado(url: str, timeout: float) -> FakeResponse:
            if url.endswith("2026/07/12"):
                return FakeResponse(404, None)
            return FakeResponse(200, {"venta": TASA})

        fake = SimpleNamespace(get=get_con_sabado)
        monkeypatch.setattr(tr, "requests", fake)
        assert tr.obtener_tipo_cambio("2026-07-12", max_intentos=3) == TASA


class TestConstruirTablaTipoCambio:
    def test_una_sola_consulta_por_fecha(self, monkeypatch: pytest.MonkeyPatch) -> None:
        consultas: list[str] = []

        def obtener_con_registro(fecha: str, market: str = "blue") -> float:
            consultas.append(fecha)
            return TASA

        monkeypatch.setattr(tr, "obtener_tipo_cambio", obtener_con_registro)

        fechas = pd.Series(pd.to_datetime(["2026-07-15", "2026-07-15", "2026-07-16"]))
        tabla = tr.construir_tabla_tipo_cambio(fechas)

        assert tabla == {"2026-07-15": TASA, "2026-07-16": TASA}
        assert consultas == ["2026-07-15", "2026-07-16"]

    def test_fechas_sin_cotizacion_no_entran(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tr, "obtener_tipo_cambio", lambda fecha, market="blue": None)
        fechas = pd.Series(pd.to_datetime(["2026-07-15"]))
        assert tr.construir_tabla_tipo_cambio(fechas) == {}


class TestNormalizarMoneda:
    def _monedas(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "precio": [250000.0, 1_200_000.0],
                "moneda": ["USD", "ARS"],
                "fecha_scrape": pd.to_datetime([FECHA, FECHA]),
            }
        )

    def test_convierte_ars_a_usd_y_deja_usd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tr, "construir_tabla_tipo_cambio", lambda fechas: {FECHA: TASA})
        df = self._monedas()
        resultado = tr.normalizar_moneda(df.copy())

        # USD no se toca
        assert resultado.loc[0, "precio_usd"] == 250000.0
        # ARS / tipo de cambio
        assert resultado.loc[1, "precio_usd"] == 1_200_000.0 / TASA
        # El tipo de cambio queda registrado por fila
        assert list(resultado["tipo_cambio_ars_usd"]) == [TASA, TASA]
        # No modifica columnas originales
        assert list(df["precio"]) == list(resultado["precio"])
        assert list(df["moneda"]) == list(resultado["moneda"])

    def test_moneda_desconocida_deja_precio_usd_vacio(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tr, "construir_tabla_tipo_cambio", lambda fechas: {FECHA: TASA})
        df = pd.DataFrame(
            {
                "precio": [100000.0],
                "moneda": ["EUR"],
                "fecha_scrape": pd.to_datetime([FECHA]),
            }
        )
        resultado = tr.normalizar_moneda(df.copy())
        assert pd.isna(resultado.loc[0, "precio_usd"])

    def test_falta_columna_precio_lanza_error(self) -> None:
        df = pd.DataFrame({"moneda": ["USD"]})
        with pytest.raises(ValueError, match="precio"):
            tr.normalizar_moneda(df)

    def test_falta_columna_fecha_scrape_lanza_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        df = pd.DataFrame({"precio": [1.0], "moneda": ["USD"]})
        with pytest.raises(ValueError, match="fecha_scrape"):
            tr.normalizar_moneda(df)


class TestNormalizarExpensas:
    def test_convierte_expensas_a_usd(self) -> None:
        df = pd.DataFrame({"expensas": [24_000.0], "tipo_cambio_ars_usd": [TASA]})
        resultado = tr.normalizar_expensas(df.copy())
        assert resultado.loc[0, "expensas_usd"] == 24_000.0 / TASA

    def test_sin_tipo_de_cambio_no_crea_columna(self) -> None:
        df = pd.DataFrame({"expensas": [24_000.0]})
        resultado = tr.normalizar_expensas(df.copy())
        assert "expensas_usd" not in resultado.columns

    def test_sin_columna_expensas_no_falla(self) -> None:
        df = pd.DataFrame({"precio": [1.0]})
        resultado = tr.normalizar_expensas(df.copy())
        assert list(resultado.columns) == ["precio"]


class TestCrearIndicadoresMissing:
    def test_crea_indicadores_binarios(self) -> None:
        df = pd.DataFrame(
            {
                "expensas": [24_000.0, pd.NA],
                "banos": [pd.NA, 2.0],
                "ambientes": [3.0, 4.0],
            }
        )
        resultado = tr.crear_indicadores_missing(df.copy())

        assert list(resultado["expensas_informado"]) == [1, 0]
        assert list(resultado["banos_informado"]) == [0, 1]
        assert list(resultado["ambientes_informado"]) == [1, 1]
        assert resultado["expensas_informado"].dtype == "int8"

    def test_ignora_columnas_ausentes(self) -> None:
        df = pd.DataFrame({"precio": [1.0]})
        resultado = tr.crear_indicadores_missing(df.copy())
        assert "precio_informado" not in resultado.columns
