"""Tests unitarios de `real_estate.models.tuning` (Fase 5 del roadmap)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from real_estate.features import pipeline as pl
from real_estate.features import transformations as tr
from real_estate.models import tuning as tu


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


def _grid_pequeno() -> dict[str, list[object]]:
    """Grid mínimo (2x2 = 4 combinaciones) para tests rápidos."""

    return {"max_depth": [3, 4], "n_estimators": [50, 100]}


class TestEspacioHiperparametros:
    def test_cubre_los_nueve_hiperparametros_del_roadmap(self) -> None:
        for clave in (
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
        ):
            assert clave in tu.ESPACIO_HIPERPARAMETROS
            assert len(tu.ESPACIO_HIPERPARAMETROS[clave]) >= 1

    def test_grid_reducido_es_subconjunto_del_espacio(self) -> None:
        for clave, valores in tu.GRID_REDUCIDO.items():
            assert clave in tu.ESPACIO_HIPERPARAMETROS
            assert set(valores) <= set(tu.ESPACIO_HIPERPARAMETROS[clave])


class TestTunearXgboost:
    def test_grid_evalua_todas_las_combinaciones(self) -> None:
        train, val, test = _splits()

        resultado = tu.tunear_xgboost(
            train,
            val,
            test,
            metodo="grid",
            espacio=_grid_pequeno(),
            cv=2,
            n_jobs=1,
        )

        assert resultado.metodo == "grid"
        assert resultado.n_trials == 4
        assert len(resultado.cv_resultados) == 4
        assert resultado.mejor_params["max_depth"] in (3, 4)
        assert resultado.mejor_params["n_estimators"] in (50, 100)
        assert resultado.mejor_puntaje_cv > 0
        assert "rmse_log" in resultado.metricas_tunedo_val
        assert "rmse_log" in resultado.metricas_tunedo_test

    def test_random_respeta_n_iter_y_semilla(self) -> None:
        train, val, test = _splits()

        resultado_1 = tu.tunear_xgboost(
            train,
            val,
            test,
            metodo="random",
            espacio=_grid_pequeno(),
            n_iter=3,
            cv=2,
            n_jobs=1,
            random_state=42,
        )
        resultado_2 = tu.tunear_xgboost(
            train,
            val,
            test,
            metodo="random",
            espacio=_grid_pequeno(),
            n_iter=3,
            cv=2,
            n_jobs=1,
            random_state=42,
        )

        assert resultado_1.metodo == "random"
        assert resultado_1.n_iter == 3
        assert resultado_1.n_trials == 3
        assert len(resultado_1.cv_resultados) == 3
        assert resultado_1.mejor_params == resultado_2.mejor_params

    def test_el_tunedo_aprende_sobre_val(self) -> None:
        train, val, test = _splits()

        resultado = tu.tunear_xgboost(
            train,
            val,
            test,
            metodo="grid",
            espacio=_grid_pequeno(),
            cv=2,
            n_jobs=1,
        )

        # La búsqueda no puede degradar al modelo por debajo del baseline: el
        # tunedo debe capturar la señal de precio del dataset sintético.
        assert resultado.metricas_tunedo_val["r2"] > 0.5

    def test_cv_resultados_ordenados_por_ranking(self) -> None:
        train, val, test = _splits()

        resultado = tu.tunear_xgboost(
            train,
            val,
            test,
            metodo="grid",
            espacio=_grid_pequeno(),
            cv=2,
            n_jobs=1,
        )

        df = resultado.cv_resultados
        assert list(df["rank_test_score"]) == sorted(df["rank_test_score"])
        assert (df["cv_rmse_log"] > 0).all()
        assert "param_max_depth" in df.columns
        assert "param_n_estimators" in df.columns
        assert df.iloc[0]["rank_test_score"] == 1

    def test_rechaza_metodo_invalido(self) -> None:
        train, val, test = _splits()

        with pytest.raises(ValueError, match="método de búsqueda inválido"):
            tu.tunear_xgboost(train, val, test, metodo="optuna")
