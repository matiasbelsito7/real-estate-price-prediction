"""Tests unitarios de `real_estate.models.entrenamiento`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from real_estate.features import pipeline as pl
from real_estate.features import transformations as tr
from real_estate.models import entrenamiento as en


def _df_curado_sintetico(n_filas: int = 96) -> pd.DataFrame:
    """Dataset curado sintético con señal de precio aprendible.

    Precio = base por tipo + prima por barrio + superficie * 1.000 + ruido.
    Todas las filas tienen precio válido (> PRECIO_MINIMO_USD).
    """

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
            "fecha_scrape": "2026-07-15",
            "tipo_cambio_ars_usd": 1200.0,
            "precio_usd": precio,
            "expensas_usd": expensas / 1200,
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


def _matriz_sintetica(n_filas: int = 96) -> pd.DataFrame:
    """Aplica selección de columnas y target log (como en la fase 4)."""

    return tr.crear_target_log(tr.seleccionar_columnas(_df_curado_sintetico(n_filas)))


def _splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits 80/10/10 reproducibles de la matriz sintética."""

    return pl.dividir_train_val_test(_matriz_sintetica(), random_state=42)


def _x_y_train() -> tuple[pd.DataFrame, pd.Series]:
    """X/y de train tras preprocesar (como hace `entrenar_y_evaluar`)."""

    train, _, _ = _splits()

    ajustes = en.ajustar_preprocesamiento(train)

    return en.separar_features_target(en.aplicar_preprocesamiento(train, ajustes))


class TestAjustarPreprocesamiento:
    def test_aprende_ordenes_e_imputador(self) -> None:
        train, _, _ = _splits()

        ajustes = en.ajustar_preprocesamiento(train)

        assert set(ajustes.ordenes) == {"barrio", "tipo_propiedad"}
        assert "expensas_usd" in ajustes.imputador
        assert ajustes.imputador["expensas_usd"] == pytest.approx(train["expensas_usd"].median())

    def test_ignora_columnas_ausentes(self) -> None:
        train, _, _ = _splits()
        train = train.drop(columns=["barrio"])

        ajustes = en.ajustar_preprocesamiento(train)

        assert "barrio" not in ajustes.ordenes


class TestAplicarPreprocesamiento:
    def test_codifica_imputa_y_no_deja_nan(self) -> None:
        train, _, _ = _splits()
        ajustes = en.ajustar_preprocesamiento(train)

        resultado = en.aplicar_preprocesamiento(train, ajustes)

        assert "barrio_ordinal" in resultado.columns
        assert "tipo_propiedad_ordinal" in resultado.columns
        assert "barrio" not in resultado.columns
        assert resultado.isna().sum().sum() == 0

    def test_categoria_solo_en_val_va_a_desconocido(self) -> None:
        # Sin fuga: una categoría ausente del train no puede inventar código.
        train, val, _ = _splits()
        val = val.copy()
        val.at[val.index[0], "barrio"] = "BarrioNuevo"

        ajustes = en.ajustar_preprocesamiento(train)
        val_proc = en.aplicar_preprocesamiento(val, ajustes)

        assert val_proc["barrio_ordinal"].iloc[0] == tr.CODIGO_DESCONOCIDO

    def test_no_modifica_el_original(self) -> None:
        train, _, _ = _splits()
        ajustes = en.ajustar_preprocesamiento(train)
        columnas = train.columns.tolist()

        en.aplicar_preprocesamiento(train, ajustes)

        assert train.columns.tolist() == columnas


class TestSepararFeaturesTarget:
    def test_excluye_precio_y_target(self) -> None:
        train, _, _ = _splits()

        x, y = en.separar_features_target(train)

        assert tr.TARGET_PRECIO not in x.columns
        assert tr.TARGET_LOG not in x.columns
        assert "superficie_cubierta" in x.columns

        pd.testing.assert_series_equal(y, train[tr.TARGET_LOG], check_names=False)

    def test_permite_target_personalizado(self) -> None:
        train, _, _ = _splits()

        x, y = en.separar_features_target(train, target=tr.TARGET_PRECIO)

        assert (y == train[tr.TARGET_PRECIO]).all()


class TestCalcularMetricas:
    def test_prediccion_perfecta(self) -> None:
        y = pd.Series([10.0, 11.0, 12.0])

        metricas = en.calcular_metricas(y, y)

        assert metricas["rmse_log"] == 0.0
        assert metricas["rmse_usd"] == 0.0
        assert metricas["r2"] == 1.0

    def test_rmse_usd_deshace_el_log(self) -> None:
        y_real = np.array([10.0, 11.0, 12.0])
        y_pred = np.array([10.5, 10.5, 10.5])

        metricas = en.calcular_metricas(y_real, y_pred)

        esperado_usd = float(np.sqrt(np.mean((np.exp(y_real) - np.exp(y_pred)) ** 2)))
        assert metricas["rmse_usd"] == pytest.approx(esperado_usd)

    def test_acepta_arrays_de_numpy(self) -> None:
        y_real = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])

        metricas = en.calcular_metricas(y_real, y_pred)

        assert isinstance(metricas["rmse_log"], float)


class TestEntrenarBaseline:
    def test_predice_siempre_la_mediana(self) -> None:
        x_train, y_train = _x_y_train()

        modelo = en.entrenar_baseline(x_train, y_train)

        predicciones = modelo.predict(x_train)

        assert (predicciones == y_train.median()).all()


class TestEntrenarXGBoost:
    def test_entrena_y_predice_con_shape_consistente(self) -> None:
        x_train, y_train = _x_y_train()

        modelo = en.entrenar_xgboost(x_train, y_train)

        predicciones = modelo.predict(x_train)

        assert predicciones.shape == (len(x_train),)
        assert np.isfinite(predicciones).all()

    def test_reproducible_con_misma_semilla(self) -> None:
        x_train, y_train = _x_y_train()

        a = en.entrenar_xgboost(x_train, y_train, random_state=5)
        b = en.entrenar_xgboost(x_train, y_train, random_state=5)

        np.testing.assert_array_equal(a.predict(x_train), b.predict(x_train))

    def test_respeta_params_personalizados(self) -> None:
        x_train, y_train = _x_y_train()

        params: dict[str, object] = {
            "n_estimators": 3,
            "max_depth": 2,
            "learning_rate": 0.1,
        }

        modelo = en.entrenar_xgboost(x_train, y_train, params=params)

        assert modelo.n_estimators == 3
        assert modelo.max_depth == 2


class TestEntrenarYEvaluar:
    def test_pipeline_completo_supera_al_baseline(self) -> None:
        train, val, test = _splits()

        resultado = en.entrenar_y_evaluar(train, val, test)

        assert (
            resultado.metricas_xgboost_val["rmse_log"]
            < (resultado.metricas_baseline_val["rmse_log"])
        )
        assert resultado.metricas_xgboost_test["r2"] > 0.5

        # El preprocesamiento queda disponible para reutilizarlo (p. ej. SHAP).
        assert "barrio" in resultado.ajustes.ordenes
