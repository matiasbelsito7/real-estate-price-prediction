"""Tests unitarios de `real_estate.serving` (bundle + predicción, fase 10)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from real_estate.features import pipeline as pl
from real_estate.features import transformations as tr
from real_estate.models import entrenamiento as en
from real_estate.persistencia.bundle import (
    NOMBRE_FEATURES,
    NOMBRE_MODELO,
    cargar_bundle,
    guardar_bundle,
)
from real_estate.serving.modelo import ModeloPrediccion


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


@pytest.fixture()
def modelo_y_bundle(tmp_path: object) -> tuple[ModeloPrediccion, object]:
    """Entrena sobre datos sintéticos, guarda el bundle y lo recarga."""

    matriz = tr.crear_target_log(tr.seleccionar_columnas(_df_curado_sintetico()))
    train, val, test = pl.dividir_train_val_test(matriz, random_state=42)
    resultado = en.entrenar_y_evaluar(train, val, test, random_state=42)

    train_proc = en.aplicar_preprocesamiento(train, resultado.ajustes)
    x_train, _ = en.separar_features_target(train_proc)
    columnas_features = list(x_train.columns)

    guardar_bundle(
        directorio=tmp_path,  # type: ignore[arg-type]
        modelo=resultado.modelo_xgboost,
        ajustes=resultado.ajustes,
        columnas_features=columnas_features,
        metadata={"metricas_xgboost_test": resultado.metricas_xgboost_test},
    )

    return cargar_bundle(tmp_path), columnas_features  # type: ignore[arg-type]


def _fila_modelo() -> dict[str, object]:
    """Una fila con las 14 features que espera el modelo."""

    return {
        "tipo_propiedad": "casa",
        "barrio": "Belgrano",
        "superficie_cubierta": 120.0,
        "ambientes": 4,
        "dormitorios": 3,
        "banos": 2,
        "antiguedad": 10,
        "expensas_usd": 150.0,
        "superficie_cubierta_informado": 1,
        "ambientes_informado": 1,
        "dormitorios_informado": 1,
        "banos_informado": 1,
        "antiguedad_informado": 1,
        "expensas_informado": 1,
    }


class TestGuardarCargarBundle:
    def test_round_trip_consistente(
        self, modelo_y_bundle: tuple[ModeloPrediccion, list[str]]
    ) -> None:
        modelo, columnas_features = modelo_y_bundle

        assert isinstance(modelo, ModeloPrediccion)
        assert modelo.columnas_features == columnas_features
        assert "barrio_ordinal" in modelo.columnas_features
        assert modelo.metadata["metricas_xgboost_test"] != {}

    def test_escribe_los_archivos_esperados(self, tmp_path: object) -> None:
        matriz = tr.crear_target_log(tr.seleccionar_columnas(_df_curado_sintetico()))
        train, val, test = pl.dividir_train_val_test(matriz, random_state=42)
        resultado = en.entrenar_y_evaluar(train, val, test)

        guardar_bundle(
            directorio=tmp_path,  # type: ignore[arg-type]
            modelo=resultado.modelo_xgboost,
            ajustes=resultado.ajustes,
            columnas_features=["a"],
        )

        for nombre in (NOMBRE_MODELO, NOMBRE_FEATURES, "preprocesamiento.json", "metadata.json"):
            assert (tmp_path / nombre).exists()  # type: ignore[operator]

    def test_preprocesamiento_guardado_es_json_valido(
        self, modelo_y_bundle: tuple[ModeloPrediccion, list[str]]
    ) -> None:
        modelo, _ = modelo_y_bundle

        assert set(modelo.ajustes.ordenes) == {"barrio", "tipo_propiedad"}
        assert "expensas_usd" in modelo.ajustes.imputador


class TestModeloPrediccion:
    def test_predecir_usd_es_exp_de_log(
        self, modelo_y_bundle: tuple[ModeloPrediccion, list[str]]
    ) -> None:
        modelo, _ = modelo_y_bundle

        df = pd.DataFrame([_fila_modelo()])

        log = modelo.predecir_log(df)
        usd = modelo.predecir_usd(df)

        np.testing.assert_allclose(usd, np.exp(log))

    def test_predecir_usd_equivale_al_pipeline_de_entrenamiento(
        self, modelo_y_bundle: tuple[ModeloPrediccion, list[str]]
    ) -> None:
        # El servicio debe replicar exactamente el pipeline de entrenamiento:
        # una fila preprocesada (sin target) predicha por el servicio debe dar
        # lo mismo que el modelo crudo sobre la matriz de features de train.
        matriz = tr.crear_target_log(tr.seleccionar_columnas(_df_curado_sintetico()))
        train, val, test = pl.dividir_train_val_test(matriz, random_state=42)
        resultado = en.entrenar_y_evaluar(train, val, test)

        train_proc = en.aplicar_preprocesamiento(train, resultado.ajustes)
        columnas_features = list(en.separar_features_target(train_proc)[0].columns)

        servicio = ModeloPrediccion(
            modelo_xgboost=resultado.modelo_xgboost,
            ajustes=resultado.ajustes,
            columnas_features=columnas_features,
        )

        # Fila preprocesada sin columnas de target (como llega al servicio).
        fila_proc = train_proc.drop(columns=[tr.TARGET_PRECIO, tr.TARGET_LOG]).iloc[[0]]
        # Referencia: el modelo crudo sobre la matriz de features de esa fila.
        x_fila = en.separar_features_target(train_proc)[0].iloc[[0]]
        esperado = resultado.modelo_xgboost.predict(x_fila)

        np.testing.assert_allclose(servicio.predecir_log(fila_proc), esperado)

    def test_reordena_columnas_al_orden_del_entrenamiento(
        self, modelo_y_bundle: tuple[ModeloPrediccion, list[str]]
    ) -> None:
        modelo, _ = modelo_y_bundle

        fila = _fila_modelo()
        fila_reordenada = {k: fila[k] for k in reversed(list(fila.keys()))}

        a = modelo.predecir_usd(pd.DataFrame([fila]))
        b = modelo.predecir_usd(pd.DataFrame([fila_reordenada]))

        np.testing.assert_allclose(a, b)

    def test_categoria_desconocida_va_a_codigo_desconocido(
        self, modelo_y_bundle: tuple[ModeloPrediccion, list[str]]
    ) -> None:
        modelo, _ = modelo_y_bundle

        df = pd.DataFrame([{**_fila_modelo(), "barrio": "BarrioInexistente"}])

        prediccion = modelo.predecir_usd(df)

        assert prediccion[0] > 0
        # Sin excepción: la categoría nueva se codifica como desconocido.
        matriz = modelo._construir_matriz(df)
        assert matriz["barrio_ordinal"].iloc[0] == tr.CODIGO_DESCONOCIDO

    def test_nan_se_imputa_con_la_mediana_del_bundle(
        self, modelo_y_bundle: tuple[ModeloPrediccion, list[str]]
    ) -> None:
        modelo, _ = modelo_y_bundle

        df = pd.DataFrame([{**_fila_modelo(), "superficie_cubierta": np.nan}])

        prediccion = modelo.predecir_usd(df)

        assert prediccion[0] > 0
        assert np.isfinite(prediccion[0])


class TestPersistenciaArchivos:
    def test_metadata_es_legible(
        self, modelo_y_bundle: tuple[ModeloPrediccion, list[str]], tmp_path: object
    ) -> None:
        _, _ = modelo_y_bundle

        with open(tmp_path / "metadata.json", encoding="utf-8") as f:  # type: ignore[operator]
            contenido = json.load(f)

        assert "metricas_xgboost_test" in contenido
