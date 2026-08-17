"""Tests unitarios de `real_estate.serving.clasificacion` (roadmap fase 3).

La clasificación es lógica pura sobre el dataset evaluado (no requiere modelo):
ratio predicho/publicado y zona neutra `1 ± std` del lote. Los tests cubren el
cálculo del ratio (incluidos precios publicados inválidos), la clasificación
en las tres categorías, el caso degenerado de std y el flujo `clasificar_y_exportar`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from real_estate.serving.clasificacion import (
    BUENA_COMPRA,
    COLUMNAS_OFERTAS,
    MALA_COMPRA,
    PRECIO_JUSTO,
    SIN_CLASIFICAR,
    clasificar_oportunidades,
    clasificar_por_diferencia,
    clasificar_y_exportar,
)


def _df_ratios(ratios: list[float], publicado: float = 100.0) -> pd.DataFrame:
    """DataFrame evaluado con ratios exactos: predicho = publicado * ratio."""

    return pd.DataFrame(
        {
            "id": [str(i) for i in range(len(ratios))],
            "titulo": [f"titulo_{i}" for i in range(len(ratios))],
            "link": [f"link_{i}" for i in range(len(ratios))],
            "barrio": ["Palermo"] * len(ratios),
            "tipo_propiedad": ["departamento"] * len(ratios),
            "precio_usd": [float(publicado)] * len(ratios),
            "precio_predicho_usd": [publicado * r for r in ratios],
            "fecha_prediccion": ["2026-08-16"] * len(ratios),
        }
    )


def _clasificaciones_esperadas(ratios: list[float]) -> list[str]:
    """Referencia independiente de la zona neutra `1 ± std` (ddof=1)."""

    ratios_arr = np.array(ratios, dtype=float)
    std = ratios_arr.std(ddof=1)
    limite_superior = 1 + std
    limite_inferior = 1 - std

    clasificaciones = []
    for r in ratios_arr:
        if r > limite_superior:
            clasificaciones.append(BUENA_COMPRA)
        elif r < limite_inferior:
            clasificaciones.append(MALA_COMPRA)
        else:
            clasificaciones.append(SIN_CLASIFICAR)
    return clasificaciones


class TestRatioPrecio:
    def test_ratio_es_predicho_sobre_publicado(self) -> None:
        df = _df_ratios([1.5, 1.0, 0.5])

        resultado = clasificar_oportunidades(df)

        ratios = resultado["ratio_precio"].tolist()
        assert ratios == pytest.approx([1.5, 1.0, 0.5])

    def test_agrega_columnas_ratio_y_clasificacion(self) -> None:
        df = _df_ratios([1.0])

        resultado = clasificar_oportunidades(df)

        assert "ratio_precio" in resultado.columns
        assert "clasificacion" in resultado.columns
        assert resultado["ratio_precio"].notna().all()
        assert resultado["clasificacion"].isin({BUENA_COMPRA, MALA_COMPRA, SIN_CLASIFICAR}).all()

    def test_no_modifica_el_dataframe_original(self) -> None:
        df = _df_ratios([1.0])
        columnas_originales = list(df.columns)

        clasificar_oportunidades(df)

        assert list(df.columns) == columnas_originales

    def test_precio_publicado_cero_da_ratio_nan_y_sin_clasificar(self) -> None:
        df = _df_ratios([1.0, 1.0])
        df.loc[1, "precio_usd"] = 0.0

        resultado = clasificar_oportunidades(df)

        assert resultado["ratio_precio"].isna().loc[1]
        assert resultado.loc[1, "clasificacion"] == SIN_CLASIFICAR

    def test_precio_publicado_faltante_da_ratio_nan(self) -> None:
        df = _df_ratios([1.0])
        df.loc[0, "precio_usd"] = np.nan

        resultado = clasificar_oportunidades(df)

        assert resultado["ratio_precio"].isna().loc[0]
        assert resultado.loc[0, "clasificacion"] == SIN_CLASIFICAR


class TestZonaNeutra:
    def test_clasifica_las_tres_categorias(self) -> None:
        # Ratios simétricos alrededor de 1: extremos fuera de la zona 1 ± std,
        # el resto dentro.
        ratios = [1.10, 1.05, 1.00, 0.95, 0.90]

        resultado = clasificar_oportunidades(_df_ratios(ratios))

        esperado = _clasificaciones_esperadas(ratios)
        assert resultado["clasificacion"].tolist() == esperado
        assert BUENA_COMPRA in esperado
        assert MALA_COMPRA in esperado
        assert SIN_CLASIFICAR in esperado

    def test_lote_con_tendencia_buena(self) -> None:
        # Ejemplo del roadmap: ratios 1.25 y 1.50 -> ambas buenas compras,
        # la de mayor ratio es relativamente mejor (score descendente).
        ratios = [1.25, 1.50]

        resultado = clasificar_oportunidades(_df_ratios(ratios))

        assert resultado["clasificacion"].tolist() == [BUENA_COMPRA, BUENA_COMPRA]
        # El ranking ordena por ratio descendente: 1.50 primero.
        orden = resultado.sort_values("ratio_precio", ascending=False)
        assert orden["ratio_precio"].tolist() == pytest.approx([1.50, 1.25])

    def test_std_degenerada_con_un_solo_ratio_compara_contra_1(self) -> None:
        # Un único ratio: no hay desviación muestral (NaN) -> se usa std = 0,
        # zona neutra [1, 1]: 1.5 > 1 -> buena compra.
        df = _df_ratios([1.5])

        resultado = clasificar_oportunidades(df)

        assert resultado.loc[0, "clasificacion"] == BUENA_COMPRA

    def test_todos_iguales_a_1_sin_clasificar(self) -> None:
        # std = 0 y ratio = 1 -> zona neutra [1, 1] -> sin clasificar.
        df = _df_ratios([1.0, 1.0, 1.0])

        resultado = clasificar_oportunidades(df)

        assert resultado["clasificacion"].tolist() == [SIN_CLASIFICAR] * 3

    def test_ratio_invalido_no_contamina_la_std(self) -> None:
        # La fila con precio publicado 0 se excluye del cálculo de std y queda
        # sin clasificar; el resto se clasifica con la zona de los válidos.
        ratios_validos = [1.10, 1.05, 1.00, 0.95, 0.90]
        df = _df_ratios(ratios_validos)
        df.loc[len(df)] = {
            "id": "x",
            "titulo": "x",
            "link": "x",
            "barrio": "x",
            "tipo_propiedad": "x",
            "precio_usd": 0.0,
            "precio_predicho_usd": 100.0,
            "fecha_prediccion": "x",
        }

        resultado = clasificar_oportunidades(df)

        esperado_validos = _clasificaciones_esperadas(ratios_validos)
        assert resultado["clasificacion"].tolist()[:-1] == esperado_validos
        assert resultado["clasificacion"].tolist()[-1] == SIN_CLASIFICAR


def _df_precios(publicados: list[float], predichos: list[float]) -> pd.DataFrame:
    """DataFrame evaluado con precios publicados y predichos explícitos."""

    return pd.DataFrame(
        {
            "id": [str(i) for i in range(len(publicados))],
            "titulo": [f"titulo_{i}" for i in range(len(publicados))],
            "link": [f"link_{i}" for i in range(len(publicados))],
            "barrio": ["Palermo"] * len(publicados),
            "tipo_propiedad": ["departamento"] * len(publicados),
            "precio_usd": [float(p) for p in publicados],
            "precio_predicho_usd": [float(p) for p in predichos],
            "fecha_prediccion": ["2026-08-16"] * len(publicados),
        }
    )


class TestClasificarPorDiferencia:
    def test_agrega_columnas_de_diferencia_y_clasificacion(self) -> None:
        df = _df_precios([100.0], [110.0])

        resultado = clasificar_por_diferencia(df)

        assert "diferencia_usd" in resultado.columns
        assert "diferencia_porcentual" in resultado.columns
        assert "clasificacion" in resultado.columns

    def test_clasifica_las_tres_categorias_con_umbral_del_10(self) -> None:
        # +20 % -> buena compra; -20 % -> mala compra; 0 % -> precio justo.
        df = _df_precios([100.0, 100.0, 100.0], [120.0, 80.0, 100.0])

        resultado = clasificar_por_diferencia(df)

        assert resultado["clasificacion"].tolist() == [BUENA_COMPRA, MALA_COMPRA, PRECIO_JUSTO]

    def test_limites_del_umbral_quedan_en_precio_justo(self) -> None:
        # Exactamente ±10 % no supera el umbral (comparación estricta).
        df = _df_precios([100.0, 100.0], [110.0, 90.0])

        resultado = clasificar_por_diferencia(df)

        assert resultado["clasificacion"].tolist() == [PRECIO_JUSTO, PRECIO_JUSTO]

    def test_diferencias_en_usd_y_porcentual_correctas(self) -> None:
        df = _df_precios([200.0, 50.0], [260.0, 40.0])

        resultado = clasificar_por_diferencia(df)

        assert resultado["diferencia_usd"].tolist() == pytest.approx([60.0, -10.0])
        assert resultado["diferencia_porcentual"].tolist() == pytest.approx([30.0, -20.0])

    def test_umbral_configurable(self) -> None:
        # Con umbral 30 %: +20 % queda en precio justo.
        df = _df_precios([100.0], [120.0])

        resultado = clasificar_por_diferencia(df, umbral=0.30)

        assert resultado.loc[0, "clasificacion"] == PRECIO_JUSTO

    def test_precio_publicado_invalido_sin_clasificar(self) -> None:
        df = _df_precios([100.0, 0.0], [120.0, 120.0])

        resultado = clasificar_por_diferencia(df)

        assert resultado.loc[0, "clasificacion"] == BUENA_COMPRA
        assert resultado.loc[1, "clasificacion"] == SIN_CLASIFICAR
        assert resultado["diferencia_usd"].isna().loc[1]
        assert resultado["diferencia_porcentual"].isna().loc[1]

    def test_precio_publicado_faltante_sin_clasificar(self) -> None:
        df = _df_precios([100.0, np.nan], [120.0, 120.0])

        resultado = clasificar_por_diferencia(df)

        assert resultado.loc[1, "clasificacion"] == SIN_CLASIFICAR
        assert resultado["diferencia_porcentual"].isna().loc[1]

    def test_no_modifica_el_dataframe_original(self) -> None:
        df = _df_precios([100.0], [120.0])
        columnas_originales = list(df.columns)

        clasificar_por_diferencia(df)

        assert list(df.columns) == columnas_originales


class TestClasificarYExportar:
    def test_escribe_ranking_ordenado_por_ratio_descendente(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        entrada = tmp_path / "evaluadas.csv"
        salida = tmp_path / "ofertas.csv"
        df = _df_ratios([1.5, 1.0, 0.5, 2.0])
        df.to_csv(entrada, index=False)

        clasificar_y_exportar(entrada, salida)

        assert salida.exists()
        ofertas = pd.read_csv(salida)
        assert list(ofertas.columns) == COLUMNAS_OFERTAS
        assert ofertas["ratio_precio"].tolist() == pytest.approx([2.0, 1.5, 1.0, 0.5])
        # La publicación de mayor ratio es una buena compra (mejor oportunidad).
        assert ofertas["clasificacion"].iloc[0] == BUENA_COMPRA

    def test_crea_directorio_padre(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        entrada = tmp_path / "evaluadas.csv"
        salida = tmp_path / "anidado" / "ofertas.csv"
        _df_ratios([1.0]).to_csv(entrada, index=False)

        clasificar_y_exportar(entrada, salida)

        assert salida.exists()

    def test_csv_inexistente_lanza_file_not_found(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(FileNotFoundError):
            clasificar_y_exportar(tmp_path / "no-existe.csv", tmp_path / "out.csv")
