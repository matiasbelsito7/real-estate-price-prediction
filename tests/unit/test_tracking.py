"""Tests unitarios de `real_estate.tracking.experimentos` (MLflow, fase 6)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient

from real_estate.features import pipeline as pl
from real_estate.features import transformations as tr
from real_estate.models import entrenamiento as en
from real_estate.models import modelos_lineales as ml
from real_estate.tracking import experimentos as ex


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


@pytest.fixture(autouse=True)
def _tracking_aislado(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Aísla el tracking entre tests: sin corridas activas y sin URI heredada."""

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    if mlflow.active_run() is not None:
        mlflow.end_run()

    yield

    if mlflow.active_run() is not None:
        mlflow.end_run()


class TestConfigurarTracking:
    def test_crea_experimento_en_store_local(self, tmp_path: Path) -> None:
        nombre = ex.configurar_tracking(tracking_uri=tmp_path.as_uri(), experimento="exp_test")

        assert nombre == "exp_test"

        experimento = mlflow.get_experiment_by_name("exp_test")
        assert experimento is not None
        assert experimento.name == "exp_test"

    def test_respeta_env_mlflow_tracking_uri(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        uri = tmp_path.as_uri()
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

        ex.configurar_tracking()

        assert mlflow.get_tracking_uri() == uri


class TestRegistrarResultado:
    def test_devuelve_run_id_y_version_y_cierra_corrida(self, tmp_path: Path) -> None:
        train, val, test = _splits()
        resultado = en.entrenar_y_evaluar(train, val, test, random_state=42)
        ex.configurar_tracking(tracking_uri=tmp_path.as_uri())

        run_id, version = ex.registrar_resultado(
            resultado,
            train,
            random_state=42,
            dataset_info="sintetico",
            split_sizes={"train": len(train), "val": len(val), "test": len(test)},
        )

        assert run_id
        assert version == "1"
        assert mlflow.active_run() is None

    def test_loguea_params_metricas_y_artefacto(self, tmp_path: Path) -> None:
        train, val, test = _splits()
        resultado = en.entrenar_y_evaluar(train, val, test, random_state=42)
        ex.configurar_tracking(tracking_uri=tmp_path.as_uri())

        run_id, _ = ex.registrar_resultado(
            resultado,
            train,
            random_state=42,
            split_sizes={"train": len(train), "val": len(val), "test": len(test)},
        )

        corrida = MlflowClient().get_run(run_id)

        params = corrida.data.params
        assert params["tipo_modelo"] == "xgboost"
        assert params["random_state"] == "42"
        assert params["xgboost_n_estimators"] == "300"
        assert params["n_train"] == str(len(train))
        assert params["n_val"] == str(len(val))
        assert params["n_test"] == str(len(test))
        assert params["n_features"]

        metricas = corrida.data.metrics
        assert "baseline_val_rmse_log" in metricas
        assert "xgboost_val_rmse_log" in metricas
        assert "xgboost_test_r2" in metricas
        assert metricas["xgboost_val_rmse_log"] < metricas["baseline_val_rmse_log"]

        rutas = [artefacto.path for artefacto in MlflowClient().list_artifacts(run_id)]
        assert "resumen_entrenamiento.json" in rutas

        ruta_json = Path(MlflowClient().download_artifacts(run_id, "resumen_entrenamiento.json"))
        resumen = json.loads(ruta_json.read_text(encoding="utf-8"))
        assert resumen["tipo_modelo"] == "xgboost"
        assert resumen["random_state"] == 42
        assert "metricas_xgboost_test" in resumen
        assert resumen["parametros_xgboost"]["n_estimators"] == 300

        # El modelo quedó versionado en el Model Registry (en MLflow 3.x vive
        # en el repositorio de modelos, no en los artefactos de la corrida).
        versiones = MlflowClient().get_latest_versions(ex.MODELO_DEFAULT)
        assert versiones
        assert str(versiones[0].version) == "1"

    def test_modelo_logueado_con_firma(self, tmp_path: Path) -> None:
        train, val, test = _splits()
        resultado = en.entrenar_y_evaluar(train, val, test, random_state=42)
        ex.configurar_tracking(tracking_uri=tmp_path.as_uri())

        _, version = ex.registrar_resultado(resultado, train, random_state=42)

        version_registrada = MlflowClient().get_model_version(ex.MODELO_DEFAULT, version)
        destino = tmp_path / "descarga_modelo"
        ruta_modelo = download_artifacts(
            artifact_uri=version_registrada.source, dst_path=str(destino)
        )
        contenido = Path(ruta_modelo, "MLmodel").read_text(encoding="utf-8")

        assert "signature:" in contenido
        assert "inputs:" in contenido


class TestRegistrarLineales:
    def test_devuelve_una_corrida_por_modelo(self, tmp_path: Path) -> None:
        train, val, test = _splits()
        resultado = ml.entrenar_y_evaluar_lineales(train, val, test, random_state=42)
        ex.configurar_tracking(tracking_uri=tmp_path.as_uri())

        runs = ex.registrar_lineales(resultado, train, random_state=42)

        assert [nombre for nombre, _ in runs] == ["lasso", "ridge"]
        assert mlflow.active_run() is None

    def test_loguea_params_metricas_y_artefacto(self, tmp_path: Path) -> None:
        train, val, test = _splits()
        resultado = ml.entrenar_y_evaluar_lineales(train, val, test, random_state=42)
        ex.configurar_tracking(tracking_uri=tmp_path.as_uri())

        runs = ex.registrar_lineales(
            resultado,
            train,
            random_state=42,
            split_sizes={"train": len(train), "val": len(val), "test": len(test)},
        )

        for nombre, run_id in runs:
            corrida = MlflowClient().get_run(run_id)

            params = corrida.data.params
            assert params["tipo_modelo"] == nombre
            assert params["alpha"] == "1.0"
            assert params["random_state"] == "42"
            assert params["n_train"] == str(len(train))
            assert params["n_val"] == str(len(val))
            assert params["n_test"] == str(len(test))
            assert params["n_features"]

            metricas = corrida.data.metrics
            assert "val_rmse_log" in metricas
            assert "val_rmse_usd" in metricas
            assert "val_r2" in metricas
            if nombre == resultado.mejor:
                assert "test_rmse_log" in metricas
                assert "test_rmse_usd" in metricas
                assert "test_r2" in metricas

            rutas = [artefacto.path for artefacto in MlflowClient().list_artifacts(run_id)]
            assert "resumen_lineal.json" in rutas

            ruta_json = Path(MlflowClient().download_artifacts(run_id, "resumen_lineal.json"))
            resumen = json.loads(ruta_json.read_text(encoding="utf-8"))
            assert resumen["tipo_modelo"] == nombre
            assert resumen["mejor_modelo"] == resultado.mejor
            assert resumen["metricas_val"]["rmse_log"] == metricas["val_rmse_log"]

    def test_no_versiona_en_el_model_registry(self, tmp_path: Path) -> None:
        train, val, test = _splits()
        resultado = ml.entrenar_y_evaluar_lineales(train, val, test, random_state=42)
        ex.configurar_tracking(tracking_uri=tmp_path.as_uri())

        ex.registrar_lineales(resultado, train, random_state=42)

        versiones = MlflowClient().search_model_versions(f"name = '{ex.MODELO_DEFAULT}'")
        assert versiones == []


class TestVersionadoModelo:
    def test_dos_corridas_generan_v1_y_v2(self, tmp_path: Path) -> None:
        train, val, test = _splits()
        ex.configurar_tracking(tracking_uri=tmp_path.as_uri())

        _, version_1 = ex.registrar_resultado(en.entrenar_y_evaluar(train, val, test), train)
        _, version_2 = ex.registrar_resultado(en.entrenar_y_evaluar(train, val, test), train)

        assert version_1 == "1"
        assert version_2 == "2"

        versiones = MlflowClient().search_model_versions(f"name = '{ex.MODELO_DEFAULT}'")
        numeros = sorted(int(version.version) for version in versiones)

        assert numeros == [1, 2]


class TestFinalizarCorrida:
    def test_no_op_sin_corrida_activa(self) -> None:
        ex.finalizar_corrida()

        assert mlflow.active_run() is None
