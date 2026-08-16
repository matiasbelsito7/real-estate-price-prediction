"""Tests unitarios de `real_estate.features.transformations` y `pipeline`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from real_estate.features import pipeline as pl
from real_estate.features import transformations as tr


def _df_sintetico() -> pd.DataFrame:
    """Dataset curado sintético con el shape de las columnas del real."""

    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "link": ["a", "b", "c", "d"],
            "titulo": ["t1", "t2", "t3", "t4"],
            "descripcion": ["d1", "d2", "d3", "d4"],
            "tipo_propiedad": ["departamento", "casa", "departamento", "ph"],
            "idtipopropiedad": [1, 2, 1, 3],
            "barrio": ["Palermo", "Caballito", "Palermo", "Belgrano"],
            "sub_barrio": ["Palermo Soho", None, None, None],
            "precio": [100000, 200000, 150000, 400000],
            "moneda": ["USD", "USD", "USD", "USD"],
            "expensas": [10, None, 30, 40],
            "superficie_cubierta": [50, None, 80, 100],
            "superficie_semicubierta": [None, None, 10, None],
            "superficie_total": [None, None, None, 120],
            "ambientes": [2, 3, 4, None],
            "dormitorios": [1, 2, 3, None],
            "banos": [1, 2, None, 2],
            "cocheras": [None, None, None, None],
            "antiguedad": [0, 10, None, 30],
            "fecha_scrape": ["2026-07-15", "2026-07-15", "2026-07-15", "2026-07-15"],
            "tipo_cambio_ars_usd": [1200, 1200, 1200, 1200],
            "precio_usd": [100000, 200000, 150000, 0],
            "expensas_usd": [1.0, None, 3.0, 4.0],
            "expensas_informado": [1, 0, 1, 1],
            "superficie_cubierta_informado": [1, 0, 1, 1],
            "superficie_semicubierta_informado": [0, 0, 1, 0],
            "superficie_total_informado": [0, 0, 0, 1],
            "ambientes_informado": [1, 1, 1, 0],
            "dormitorios_informado": [1, 1, 1, 0],
            "banos_informado": [1, 1, 0, 1],
            "cocheras_informado": [0, 0, 0, 0],
            "antiguedad_informado": [1, 1, 0, 1],
        }
    )


class TestSeleccionarColumnas:
    def test_descarta_sin_senal_y_conserva_features_y_target(self) -> None:
        df = _df_sintetico()
        resultado = tr.seleccionar_columnas(df)

        for columna in tr.COLUMNAS_DESCARTAR:
            assert columna not in resultado.columns

        esperadas = {
            "superficie_cubierta",
            "ambientes",
            "dormitorios",
            "banos",
            "antiguedad",
            "expensas_usd",
            "expensas_informado",
            "superficie_cubierta_informado",
            "ambientes_informado",
            "dormitorios_informado",
            "banos_informado",
            "antiguedad_informado",
            "barrio",
            "tipo_propiedad",
            "precio_usd",
        }
        assert esperadas.issubset(resultado.columns)

    def test_no_modifica_el_dataframe_original(self) -> None:
        df = _df_sintetico()
        columnas = df.columns.tolist()

        tr.seleccionar_columnas(df)

        assert df.columns.tolist() == columnas

    def test_ignora_columnas_ausentes(self) -> None:
        df = _df_sintetico().drop(columns=["cocheras"])

        assert "cocheras" not in tr.seleccionar_columnas(df).columns


class TestCrearTargetLog:
    def test_filtra_precios_invalidos_y_crea_log(self) -> None:
        df = tr.seleccionar_columnas(_df_sintetico())
        resultado = tr.crear_target_log(df)

        # La fila con precio_usd == 0 se descarta.
        assert len(resultado) == 3
        assert tr.TARGET_LOG in resultado.columns

        esperado = np.log(resultado["precio_usd"])

        pd.testing.assert_series_equal(
            resultado[tr.TARGET_LOG],
            esperado,
            check_names=False,
        )

    def test_elimina_nan_de_precio(self) -> None:
        df = tr.seleccionar_columnas(_df_sintetico()).copy()
        df.loc[0, "precio_usd"] = np.nan

        resultado = tr.crear_target_log(df)

        assert resultado[tr.TARGET_LOG].notna().all()

    def test_descarta_precios_artefacto_bajo_minimo(self) -> None:
        # Un precio de 1 USD es un artefacto del scraping (p. ej. "U$S A B2").
        df = tr.seleccionar_columnas(_df_sintetico()).copy()
        df.loc[2, "precio_usd"] = 1

        resultado = tr.crear_target_log(df)

        # Se descartan la fila 2 (precio 1) y la fila 3 (precio 0).
        assert len(resultado) == 2

    def test_falla_sin_columna_precio(self) -> None:
        df = tr.seleccionar_columnas(_df_sintetico()).drop(columns=["precio_usd"])

        with pytest.raises(ValueError, match="precio_usd"):
            tr.crear_target_log(df)


class TestCrearOrdenMediana:
    def test_ordena_por_mediana_de_precio_ascendente(self) -> None:
        df = tr.seleccionar_columnas(_df_sintetico())

        # barrio: Palermo (100k, 150k) < Caballito (200k) < Belgrano (400k).
        orden = tr.crear_orden_mediana(df, "barrio")

        medianas = df.groupby("barrio")["precio_usd"].median()

        assert len(orden) == df["barrio"].nunique()
        assert all(isinstance(categoria, str) for categoria in orden)

        # La categoría con menor mediana queda primera y la mayor última.
        assert orden[0] == medianas.idxmin()
        assert orden[-1] == medianas.idxmax()

    def test_excluye_categoria_nan(self) -> None:
        df = tr.seleccionar_columnas(_df_sintetico()).copy()
        df.loc[0, "barrio"] = np.nan

        orden = tr.crear_orden_mediana(df, "barrio")

        assert len(orden) == df["barrio"].dropna().nunique()

    def test_falla_sin_columna(self) -> None:
        df = tr.seleccionar_columnas(_df_sintetico()).drop(columns=["barrio"])

        with pytest.raises(ValueError, match="barrio"):
            tr.crear_orden_mediana(df, "barrio")


class TestCodificarOrdinal:
    def test_mapea_categorias_por_ranking(self) -> None:
        df = tr.seleccionar_columnas(_df_sintetico())
        orden = ["Palermo", "Caballito", "Belgrano"]

        resultado = tr.codificar_ordinal(df, "barrio", orden)

        assert "barrio" not in resultado.columns
        assert "barrio_ordinal" in resultado.columns

        codigos = resultado["barrio_ordinal"].tolist()
        assert codigos == [0, 1, 0, 2]

    def test_categoria_desconocida_se_codifica_como_desconocida(self) -> None:
        df = tr.seleccionar_columnas(_df_sintetico())
        orden = ["Palermo"]

        resultado = tr.codificar_ordinal(df, "barrio", orden)

        # Caballito y Belgrano no están en el orden -> CODIGO_DESCONOCIDO.
        assert (resultado["barrio_ordinal"] == tr.CODIGO_DESCONOCIDO).sum() == 2

    def test_nan_se_codifica_como_desconocida(self) -> None:
        df = tr.seleccionar_columnas(_df_sintetico()).copy()
        df.loc[0, "barrio"] = np.nan

        resultado = tr.codificar_ordinal(df, "barrio", ["Caballito", "Belgrano"])

        assert resultado["barrio_ordinal"].iloc[0] == tr.CODIGO_DESCONOCIDO

    def test_no_modifica_el_original(self) -> None:
        df = tr.seleccionar_columnas(_df_sintetico())
        columnas = df.columns.tolist()

        tr.codificar_ordinal(df, "barrio", ["Palermo"])

        assert df.columns.tolist() == columnas


class TestImputacion:
    def test_crear_imputador_usa_mediana(self) -> None:
        df = tr.seleccionar_columnas(_df_sintetico())

        imputador = tr.crear_imputador(df, ["superficie_cubierta", "banos"])

        assert imputador["superficie_cubierta"] == pytest.approx(df["superficie_cubierta"].median())
        assert imputador["banos"] == pytest.approx(df["banos"].median())

    def test_aplicar_imputacion_rellena_nan(self) -> None:
        df = tr.seleccionar_columnas(_df_sintetico())
        imputador = tr.crear_imputador(df, ["superficie_cubierta"])

        resultado = tr.aplicar_imputacion(df, imputador)

        assert resultado["superficie_cubierta"].isna().sum() == 0
        assert resultado["superficie_cubierta"].iloc[1] == pytest.approx(
            df["superficie_cubierta"].median()
        )

    def test_aplicar_imputacion_no_toca_valores_presentes(self) -> None:
        df = tr.seleccionar_columnas(_df_sintetico())
        imputador = tr.crear_imputador(df, ["superficie_cubierta"])

        resultado = tr.aplicar_imputacion(df, imputador)

        assert resultado["superficie_cubierta"].iloc[0] == 50

    def test_columna_sin_valores_validos_queda_fuera(self) -> None:
        df = tr.seleccionar_columnas(_df_sintetico()).copy()
        df["vacia"] = np.nan

        imputador = tr.crear_imputador(df, ["vacia"])

        assert "vacia" not in imputador


class TestConstruirFeatures:
    def test_matriz_lista_para_modelar(self) -> None:
        df = _df_sintetico()

        resultado = pl.construir_features(df)

        # La fila con precio_usd == 0 se descartó.
        assert len(resultado) == 3

        # Sin NaN en las features numéricas.
        assert resultado.isna().sum().sum() == 0

        # Columnas esperadas.
        assert "barrio_ordinal" in resultado.columns
        assert "tipo_propiedad_ordinal" in resultado.columns
        assert tr.TARGET_LOG in resultado.columns
        assert tr.TARGET_PRECIO in resultado.columns

        # Sin columnas descartadas.
        for columna in tr.COLUMNAS_DESCARTAR:
            assert columna not in resultado.columns

    def test_ordinal_consistente_con_mediana_de_precio(self) -> None:
        df = _df_sintetico()
        resultado = pl.construir_features(df)

        # departamento (mediana 125k) < casa (200k); ph se descartó (precio 0).
        sub = df[df["precio_usd"] > 0]
        orden = sub.groupby("tipo_propiedad")["precio_usd"].median().sort_values().index.tolist()

        esperado = (
            sub["tipo_propiedad"].map({tipo: i for i, tipo in enumerate(orden)}).astype("int32")
        )

        pd.testing.assert_series_equal(
            resultado["tipo_propiedad_ordinal"],
            esperado,
            check_names=False,
        )


class TestDividirTrainValTest:
    def _matriz_con_muchas_filas(self) -> pd.DataFrame:
        # 30 filas: train 24, temporal 6, val 3, test 3 (proporciones 80/10/10).
        return pl.construir_features(pd.concat([_df_sintetico()] * 10, ignore_index=True))

    def test_tamanos_y_disjuncion(self) -> None:
        df = self._matriz_con_muchas_filas()

        train, val, test = pl.dividir_train_val_test(df)

        assert len(train) + len(val) + len(test) == len(df)
        assert len(val) == len(test)
        assert train.index.isin(val.index).sum() == 0
        assert train.index.isin(test.index).sum() == 0
        assert val.index.isin(test.index).sum() == 0

    def test_reproducible_con_misma_semilla(self) -> None:
        df = self._matriz_con_muchas_filas()

        train_a, val_a, test_a = pl.dividir_train_val_test(df, random_state=7)
        train_b, val_b, test_b = pl.dividir_train_val_test(df, random_state=7)

        pd.testing.assert_frame_equal(train_a, train_b)
        pd.testing.assert_frame_equal(val_a, val_b)
        pd.testing.assert_frame_equal(test_a, test_b)
