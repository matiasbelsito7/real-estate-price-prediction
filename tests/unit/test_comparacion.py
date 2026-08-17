"""Tests unitarios de `real_estate.tracking.comparacion` (champion, fase 6)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from real_estate.features import pipeline as pl
from real_estate.features import transformations as tr
from real_estate.models import entrenamiento as en
from real_estate.models import modelos_lineales as ml
from real_estate.tracking import comparacion as cp
from real_estate.tracking import experimentos as ex

EXPERIMENTO = "prediccion_precios_propiedades"
METRICA = cp.METRICA_DEFAULT


def _df_curado_sintetico(n_filas: int = 96) -> pd.DataFrame:
    """Dataset curado sintético con señal de precio aprendible (igual al de tracking)."""

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


@pytest.fixture(autouse=True)
def _tracking_aislado(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Aísla el tracking entre tests: sin corridas activas y sin URI heredada."""

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    if mlflow.active_run() is not None:
        mlflow.end_run()

    yield

    if mlflow.active_run() is not None:
        mlflow.end_run()


def _registrar_run(tracking_uri: str, random_state: int = 42) -> str:
    """Entrena y registra una corrida de XGBoost completa, devolviendo su run_id."""

    train, val, test = _splits()
    resultado = en.entrenar_y_evaluar(train, val, test, random_state=random_state)
    ex.configurar_tracking(tracking_uri=tracking_uri)

    run_id, _ = ex.registrar_resultado(resultado, train, random_state=random_state)

    return run_id


class TestCompararRuns:
    def test_tabla_ordenada_mejor_a_peor(self, tmp_path: Path) -> None:
        uri = tmp_path.as_uri()
        run_a = _registrar_run(uri, random_state=42)
        run_b = _registrar_run(uri, random_state=43)

        tabla = cp.comparar_runs(EXPERIMENTO, METRICA)

        assert list(tabla.columns) == ["run_id", "tipo_modelo", "valor"]
        assert set(tabla["run_id"]) == {run_a, run_b}
        assert set(tabla["tipo_modelo"]) == {"xgboost"}
        assert list(tabla["valor"]) == sorted(tabla["valor"])

    def test_omite_corridas_sin_la_metrica(self, tmp_path: Path) -> None:
        uri = tmp_path.as_uri()
        _registrar_run(uri, random_state=42)

        # Los lineales no loguean `xgboost_test_rmse_log`: deben quedar fuera.
        train, val, test = _splits()
        resultado = ml.entrenar_y_evaluar_lineales(train, val, test, random_state=42)
        ex.configurar_tracking(tracking_uri=uri)
        ex.registrar_lineales(resultado, train, random_state=42)

        tabla = cp.comparar_runs(EXPERIMENTO, METRICA)

        assert len(tabla) == 1
        assert set(tabla["tipo_modelo"]) == {"xgboost"}

    def test_experimento_sin_corridas_devuelve_tabla_vacia(self, tmp_path: Path) -> None:
        ex.configurar_tracking(tracking_uri=tmp_path.as_uri(), experimento=EXPERIMENTO)

        tabla = cp.comparar_runs(EXPERIMENTO, METRICA)

        assert tabla.empty
        assert list(tabla.columns) == ["run_id", "tipo_modelo", "valor"]

    def test_experimento_inexistente_levanta_error(self, tmp_path: Path) -> None:
        ex.configurar_tracking(tracking_uri=tmp_path.as_uri(), experimento="otro_exp")

        with pytest.raises(ValueError, match="no existe"):
            cp.comparar_runs("exp_que_no_existe", METRICA)


class TestElegirChampion:
    def test_devuelve_la_corrida_con_mejor_metrica(self, tmp_path: Path) -> None:
        uri = tmp_path.as_uri()
        run_a = _registrar_run(uri, random_state=42)
        run_b = _registrar_run(uri, random_state=43)

        champion = cp.elegir_champion(EXPERIMENTO, METRICA)

        assert isinstance(champion, cp.Champion)
        assert champion.metrica == METRICA
        assert champion.tipo_modelo == "xgboost"
        assert champion.run_id in {run_a, run_b}

        valores = {
            MlflowClient().get_run(run_id).data.metrics[METRICA] for run_id in (run_a, run_b)
        }
        assert champion.valor == min(valores)

        tabla = cp.comparar_runs(EXPERIMENTO, METRICA)
        assert champion.run_id == tabla.iloc[0]["run_id"]
        assert champion.valor == tabla.iloc[0]["valor"]

    def test_sin_corridas_con_metrica_levanta_error(self, tmp_path: Path) -> None:
        ex.configurar_tracking(tracking_uri=tmp_path.as_uri(), experimento=EXPERIMENTO)

        with pytest.raises(ValueError, match="no hay corridas"):
            cp.elegir_champion(EXPERIMENTO, METRICA)

    def test_experimento_inexistente_levanta_error(self, tmp_path: Path) -> None:
        ex.configurar_tracking(tracking_uri=tmp_path.as_uri(), experimento="otro_exp")

        with pytest.raises(ValueError, match="no existe"):
            cp.elegir_champion("exp_que_no_existe", METRICA)
