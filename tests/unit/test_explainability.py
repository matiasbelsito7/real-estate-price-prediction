"""Tests unitarios de `real_estate.explainability.shap_analysis` (fase 7)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin display (CI / headless)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from real_estate.explainability import (  # noqa: E402
    ExplicacionSHAP,
    calcular_shap,
    grafico_barras,
    grafico_resumen,
    guardar_figuras,
)
from real_estate.features import pipeline as pl  # noqa: E402
from real_estate.features import transformations as tr  # noqa: E402
from real_estate.models import entrenamiento as en  # noqa: E402


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


def _splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits 80/10/10 reproducibles de la matriz sintética."""

    matriz = tr.crear_target_log(tr.seleccionar_columnas(_df_curado_sintetico()))

    return pl.dividir_train_val_test(matriz, random_state=42)


@pytest.fixture(scope="module")
def modelo_y_val() -> tuple[XGBRegressor, pd.DataFrame]:
    """XGBoost ajustado (sin fuga) y la matriz X de validación, una sola vez."""

    train, val, test = _splits()
    resultado = en.entrenar_y_evaluar(train, val, test, random_state=42)
    val_proc = en.aplicar_preprocesamiento(val, resultado.ajustes)
    x_val, _ = en.separar_features_target(val_proc)

    return resultado.modelo_xgboost, x_val


def _cerrar_figuras(figuras: list[Figure]) -> None:
    """Cierra las figuras para no dejar handles abiertos entre tests."""

    for figura in figuras:
        plt.close(figura)


class TestCalcularShap:
    def test_devuelve_valores_con_forma_y_nombres(
        self, modelo_y_val: tuple[XGBRegressor, pd.DataFrame]
    ) -> None:
        modelo, x_val = modelo_y_val

        explicacion = calcular_shap(modelo, x_val)

        assert isinstance(explicacion, ExplicacionSHAP)
        assert explicacion.valores.shape == (len(x_val), x_val.shape[1])
        assert explicacion.valores.ndim == 2
        assert isinstance(explicacion.base, float)
        assert np.isfinite(explicacion.base)
        assert explicacion.nombres == tuple(x_val.columns)

    def test_propiedad_aditiva_suma_mas_base_igual_prediccion(
        self, modelo_y_val: tuple[XGBRegressor, pd.DataFrame]
    ) -> None:
        """La propiedad central de SHAP: contribuciones + base ≈ predicción."""

        modelo, x_val = modelo_y_val
        explicacion = calcular_shap(modelo, x_val)

        reconstruida = explicacion.base + explicacion.valores.sum(axis=1)
        predicha = modelo.predict(x_val)

        assert np.allclose(reconstruida, predicha, atol=1e-3)


class TestImportanciaGlobal:
    def test_serie_ordenada_descendente_con_todas_las_features(
        self, modelo_y_val: tuple[XGBRegressor, pd.DataFrame]
    ) -> None:
        modelo, x_val = modelo_y_val
        explicacion = calcular_shap(modelo, x_val)

        importancia = explicacion.importancia_global()

        assert len(importancia) == x_val.shape[1]
        assert set(importancia.index) == set(x_val.columns)
        assert (importancia >= 0).all()
        assert importancia.is_monotonic_decreasing


class TestGraficos:
    def test_graficos_devuelven_figuras_matplotlib(
        self, modelo_y_val: tuple[XGBRegressor, pd.DataFrame]
    ) -> None:
        modelo, x_val = modelo_y_val
        explicacion = calcular_shap(modelo, x_val)

        figuras = [grafico_resumen(explicacion, x_val), grafico_barras(explicacion)]

        for figura in figuras:
            assert isinstance(figura, Figure)
            assert figura.axes

        _cerrar_figuras(figuras)

    def test_guardar_figuras_escribe_png(
        self, tmp_path: Path, modelo_y_val: tuple[XGBRegressor, pd.DataFrame]
    ) -> None:
        modelo, x_val = modelo_y_val
        explicacion = calcular_shap(modelo, x_val)

        figuras = {
            "resumen": grafico_resumen(explicacion, x_val),
            "barras": grafico_barras(explicacion),
        }

        rutas = guardar_figuras(figuras, tmp_path)

        assert len(rutas) == 2
        for ruta in rutas:
            assert ruta.exists()
            assert ruta.suffix == ".png"
            assert ruta.stat().st_size > 0

        _cerrar_figuras(list(figuras.values()))
