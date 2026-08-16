"""Tests unitarios de `real_estate.models.modelos_lineales` (fase 4).

Los lineales deben usar el mismo preprocesamiento y las mismas features que
XGBoost (comparación justa) con un `StandardScaler` dentro del pipeline
ajustado solo sobre train. Los tests cubren el pipeline escalado, el
entrenamiento de Lasso y Ridge, y el flujo completo `entrenar_y_evaluar_lineales`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

from real_estate.features import pipeline as pl
from real_estate.features import transformations as tr
from real_estate.models import entrenamiento as en
from real_estate.models import modelos_lineales as ml


def _df_curado_sintetico(n_filas: int = 96) -> pd.DataFrame:
    """Dataset curado sintético con señal de precio aprendible (igual al de modelos)."""

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


class TestCrearPipelineLineal:
    def test_escala_antes_del_modelo(self) -> None:
        modelo = ml.crear_pipeline_lineal(Lasso(alpha=1.0, random_state=42))

        assert [nombre for nombre, _ in modelo.steps] == ["scaler", "modelo"]
        assert isinstance(modelo.named_steps["scaler"], StandardScaler)

    def test_acepta_lasso_y_ridge(self) -> None:
        for estimador in (Lasso(alpha=1.0), Ridge(alpha=1.0)):
            modelo = ml.crear_pipeline_lineal(estimador)

            assert [nombre for nombre, _ in modelo.steps] == ["scaler", "modelo"]


class TestEntrenarLasso:
    def test_entrena_y_predice_con_shape_consistente(self) -> None:
        x_train, y_train = _x_y_train()

        modelo = ml.entrenar_lasso(x_train, y_train)

        predicciones = modelo.predict(x_train)

        assert predicciones.shape == (len(x_train),)
        assert np.isfinite(predicciones).all()

    def test_respeta_alpha(self) -> None:
        x_train, y_train = _x_y_train()

        modelo = ml.entrenar_lasso(x_train, y_train, alpha=0.5)

        assert modelo.named_steps["modelo"].alpha == 0.5


class TestEntrenarRidge:
    def test_entrena_y_predice_con_shape_consistente(self) -> None:
        x_train, y_train = _x_y_train()

        modelo = ml.entrenar_ridge(x_train, y_train)

        predicciones = modelo.predict(x_train)

        assert predicciones.shape == (len(x_train),)
        assert np.isfinite(predicciones).all()

    def test_respeta_alpha(self) -> None:
        x_train, y_train = _x_y_train()

        modelo = ml.entrenar_ridge(x_train, y_train, alpha=0.5)

        assert modelo.named_steps["modelo"].alpha == 0.5


class TestEntrenarYEvaluarLineales:
    def test_pipeline_completo_mejora_al_baseline(self) -> None:
        train, val, test = _splits()

        resultado = ml.entrenar_y_evaluar_lineales(train, val, test)

        # Baseline: predice siempre la mediana (referencia mínima).
        ajustes = en.ajustar_preprocesamiento(train)
        x_train, y_train = en.separar_features_target(en.aplicar_preprocesamiento(train, ajustes))
        x_val, y_val = en.separar_features_target(en.aplicar_preprocesamiento(val, ajustes))
        baseline = DummyRegressor(strategy="median").fit(x_train, y_train)
        rmse_log_baseline = en.calcular_metricas(y_val, baseline.predict(x_val))["rmse_log"]

        assert resultado.metricas_lasso_val["rmse_log"] < rmse_log_baseline
        assert resultado.metricas_ridge_val["rmse_log"] < rmse_log_baseline
        assert resultado.mejor in {"lasso", "ridge"}

        # El mejor de ambos sobre val, evaluado en test, captura la señal.
        assert resultado.metricas_mejor_test["r2"] > 0.5

    def test_el_mejor_en_val_se_evalua_en_test(self) -> None:
        train, val, test = _splits()

        resultado = ml.entrenar_y_evaluar_lineales(train, val, test)

        metricas_val = {
            "lasso": resultado.metricas_lasso_val,
            "ridge": resultado.metricas_ridge_val,
        }
        assert resultado.mejor == min(metricas_val, key=lambda k: metricas_val[k]["rmse_log"])

    def test_ajustes_disponibles_para_reutilizar(self) -> None:
        train, val, test = _splits()

        resultado = ml.entrenar_y_evaluar_lineales(train, val, test)

        assert "barrio" in resultado.ajustes.ordenes
