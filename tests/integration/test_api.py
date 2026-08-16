"""Tests de integración de la API FastAPI (fase 10).

Entrena un bundle de serving sobre datos sintéticos en un `tmp_path` y lo
inyecta a la app vía `crear_app(config)` para no depender del bundle real
(que está gitignoreado y no existe en CI).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from real_estate.api import ConfiguracionServicio, crear_app
from real_estate.features import pipeline as pl
from real_estate.features import transformations as tr
from real_estate.models import entrenamiento as en
from real_estate.serving import guardar_bundle


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


PAYLOAD_VALIDO: dict[str, object] = {
    "tipo_propiedad": "departamento",
    "barrio": "Palermo",
    "superficie_cubierta": 75.0,
    "ambientes": 3,
    "dormitorios": 2,
    "banos": 2,
    "antiguedad": 30,
    "expensas_usd": 150.0,
}


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    """Entrena sobre datos sintéticos y guarda el bundle de serving en tmp_path."""

    matriz = tr.crear_target_log(tr.seleccionar_columnas(_df_curado_sintetico()))
    train, val, test = pl.dividir_train_val_test(matriz, random_state=42)
    resultado = en.entrenar_y_evaluar(train, val, test, random_state=42)

    train_proc = en.aplicar_preprocesamiento(train, resultado.ajustes)
    x_train, _ = en.separar_features_target(train_proc)

    guardar_bundle(
        directorio=tmp_path,
        modelo=resultado.modelo_xgboost,
        ajustes=resultado.ajustes,
        columnas_features=list(x_train.columns),
        metadata={"metricas_xgboost_test": resultado.metricas_xgboost_test},
    )

    return tmp_path


@pytest.fixture
def client(bundle_dir: Path) -> Iterator[TestClient]:
    """Cliente de la app con el bundle sintético inyectado."""

    config = ConfiguracionServicio(modelo_dir=bundle_dir)
    app = crear_app(config)

    with TestClient(app) as client:
        yield client


class TestHealth:
    def test_health_ok(self, client: TestClient) -> None:
        respuesta = client.get("/health")

        assert respuesta.status_code == 200
        cuerpo = respuesta.json()
        assert cuerpo["estado"] == "ok"
        assert cuerpo["modelo"] == "xgboost"
        assert cuerpo["version"] == "0.1.0"
        assert "metricas_xgboost_test" in cuerpo

    def test_health_reporta_metricas_del_bundle(self, client: TestClient) -> None:
        cuerpo = client.get("/health").json()

        metricas = cuerpo["metricas_xgboost_test"]
        assert "rmse_log" in metricas
        assert "r2" in metricas


class TestPredict:
    def test_predice_precio_razonable(self, client: TestClient) -> None:
        respuesta = client.post("/predict", json=PAYLOAD_VALIDO)

        assert respuesta.status_code == 200
        cuerpo = respuesta.json()
        assert cuerpo["precio_usd"] > 0
        assert np.isfinite(cuerpo["precio_usd"])
        assert cuerpo["log_precio_usd"] == pytest.approx(np.log(cuerpo["precio_usd"]))

    def test_prediccion_estable_ante_reorden_de_payload(self, client: TestClient) -> None:
        reordenado = dict(reversed(list(PAYLOAD_VALIDO.items())))

        a = client.post("/predict", json=PAYLOAD_VALIDO).json()["precio_usd"]
        b = client.post("/predict", json=reordenado).json()["precio_usd"]

        assert a == pytest.approx(b)

    def test_deriva_indicadores_informado_del_valor(self, client: TestClient) -> None:
        # Sin indicadores: deben derivarse (1 si el valor viene informado).
        sin_indicadores = client.post("/predict", json=PAYLOAD_VALIDO)
        con_indicadores = client.post(
            "/predict",
            json={
                **PAYLOAD_VALIDO,
                "superficie_cubierta_informado": 1,
                "ambientes_informado": 1,
                "dormitorios_informado": 1,
                "banos_informado": 1,
                "antiguedad_informado": 1,
                "expensas_informado": 1,
            },
        )

        assert sin_indicadores.status_code == 200
        assert sin_indicadores.json()["precio_usd"] == pytest.approx(
            con_indicadores.json()["precio_usd"]
        )

    def test_valores_faltantes_se_imputan(self, client: TestClient) -> None:
        payload = {**PAYLOAD_VALIDO, "superficie_cubierta": None, "expensas_usd": None}

        respuesta = client.post("/predict", json=payload)

        assert respuesta.status_code == 200
        assert respuesta.json()["precio_usd"] > 0

    def test_categoria_desconocida_no_falla(self, client: TestClient) -> None:
        payload = {**PAYLOAD_VALIDO, "barrio": "BarrioInexistente"}

        respuesta = client.post("/predict", json=payload)

        assert respuesta.status_code == 200
        assert respuesta.json()["precio_usd"] > 0

    def test_faltan_campos_requeridos_devuelve_422(self, client: TestClient) -> None:
        respuesta = client.post("/predict", json={"barrio": "Palermo"})

        assert respuesta.status_code == 422

    def test_indicador_fuera_de_rango_devuelve_422(self, client: TestClient) -> None:
        respuesta = client.post("/predict", json={**PAYLOAD_VALIDO, "ambientes_informado": 5})

        assert respuesta.status_code == 422

    def test_valor_negativo_devuelve_422(self, client: TestClient) -> None:
        respuesta = client.post("/predict", json={**PAYLOAD_VALIDO, "superficie_cubierta": -10})

        assert respuesta.status_code == 422


class TestArranque:
    def test_bundle_inexistente_no_arranca(self, tmp_path: Path) -> None:
        config = ConfiguracionServicio(modelo_dir=tmp_path / "no-existe")
        app = crear_app(config)

        with pytest.raises(RuntimeError, match="bundle de serving"), TestClient(app):
            pass
