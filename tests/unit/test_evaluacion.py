"""Tests unitarios de `real_estate.evaluacion.analisis` (fase 8)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin display (CI / headless)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from real_estate.evaluacion import (  # noqa: E402
    bias_por_rango_precio,
    grafico_error_segmento,
    grafico_residuos,
    grafico_sesgo_rango,
    guardar_figuras,
    metricas_detalladas,
    metricas_por_segmento,
    resumen_errores,
    tabla_residuos,
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
def datos_evaluacion() -> tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Series, np.ndarray]:
    """
    Splits originales y predicciones del XGBoost (en log), una sola vez.

    Devuelve `(val, y_val, y_pred_val, y_test, y_pred_test)`: `val` conserva
    las categóricas originales (para segmentar) y los targets/predicciones
    están en el espacio logarítmico.
    """

    train, val, test = _splits()
    resultado = en.entrenar_y_evaluar(train, val, test, random_state=42)

    val_proc = en.aplicar_preprocesamiento(val, resultado.ajustes)
    test_proc = en.aplicar_preprocesamiento(test, resultado.ajustes)
    x_val, y_val = en.separar_features_target(val_proc)
    x_test, y_test = en.separar_features_target(test_proc)

    return (
        val,
        y_val,
        resultado.modelo_xgboost.predict(x_val),
        y_test,
        resultado.modelo_xgboost.predict(x_test),
    )


def _cerrar_figuras(figuras: list[Figure]) -> None:
    """Cierra las figuras para no dejar handles abiertos entre tests."""

    for figura in figuras:
        plt.close(figura)


class TestMetricasDetalladas:
    def test_prediccion_perfecta_da_error_cero_y_r2_uno(
        self, datos_evaluacion: tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Series, np.ndarray]
    ) -> None:
        _, _, _, y_test, _ = datos_evaluacion

        metricas = metricas_detalladas(y_test, y_test)

        assert metricas["rmse_log"] == pytest.approx(0.0)
        assert metricas["rmse_usd"] == pytest.approx(0.0)
        assert metricas["mae_usd"] == pytest.approx(0.0)
        assert metricas["medae_usd"] == pytest.approx(0.0)
        assert metricas["mape_usd"] == pytest.approx(0.0)
        assert metricas["r2"] == pytest.approx(1.0)

    def test_coincide_con_calcular_metricas_de_entrenamiento(
        self, datos_evaluacion: tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Series, np.ndarray]
    ) -> None:
        """Las métricas compartidas con la fase 5 deben ser idénticas."""

        _, _, _, y_test, y_pred_test = datos_evaluacion

        detalladas = metricas_detalladas(y_test, y_pred_test)
        de_entrenamiento = en.calcular_metricas(y_test, y_pred_test)

        for clave in ("rmse_log", "rmse_usd", "r2"):
            assert detalladas[clave] == pytest.approx(de_entrenamiento[clave])


class TestTablaResiduos:
    def test_columnas_y_longitud(
        self, datos_evaluacion: tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Series, np.ndarray]
    ) -> None:
        _, _, _, y_test, y_pred_test = datos_evaluacion

        tabla = tabla_residuos(y_test, y_pred_test)

        assert list(tabla.columns) == [
            "precio_real_usd",
            "precio_pred_usd",
            "residuo_log",
            "residuo_usd",
            "error_relativo",
        ]
        assert len(tabla) == len(y_test)

    def test_relaciones_internas(
        self, datos_evaluacion: tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Series, np.ndarray]
    ) -> None:
        """residuo_log = pred - real; residuo_usd = exp(pred) - exp(real); error = expm1(residuo)."""

        _, _, _, y_test, y_pred_test = datos_evaluacion
        tabla = tabla_residuos(y_test, y_pred_test)

        # XGBoost predice en float32; se pasa a float64 para comparar con el
        # módulo (que convierte a float64) sin pérdida de precisión en `exp`.
        real_log = np.asarray(y_test, dtype=float)
        pred_log = np.asarray(y_pred_test, dtype=float)

        assert tabla["residuo_log"].to_numpy() == pytest.approx(pred_log - real_log)
        assert tabla["residuo_usd"].to_numpy() == pytest.approx(np.exp(pred_log) - np.exp(real_log))
        assert tabla["error_relativo"].to_numpy() == pytest.approx(
            np.expm1(tabla["residuo_log"].to_numpy())
        )

    def test_prediccion_exacta_da_residuos_cero(
        self, datos_evaluacion: tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Series, np.ndarray]
    ) -> None:
        _, _, _, y_test, _ = datos_evaluacion

        tabla = tabla_residuos(y_test, y_test)

        assert (tabla["residuo_log"] == 0.0).all()
        assert (tabla["residuo_usd"] == 0.0).all()
        assert (tabla["error_relativo"] == 0.0).all()


class TestResumenErrores:
    def test_serie_con_todas_las_claves_y_n(
        self, datos_evaluacion: tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Series, np.ndarray]
    ) -> None:
        _, _, _, y_test, y_pred_test = datos_evaluacion
        tabla = tabla_residuos(y_test, y_pred_test)

        resumen = resumen_errores(tabla)

        assert set(resumen.index) == {
            "n",
            "sesgo_log_medio",
            "sesgo_pct_medio",
            "error_pct_mediana",
            "error_pct_p75",
            "error_pct_p90",
            "error_pct_p95",
            "error_pct_max",
        }
        assert resumen["n"] == len(tabla)

    def test_coherente_con_la_tabla(
        self, datos_evaluacion: tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Series, np.ndarray]
    ) -> None:
        _, _, _, y_test, y_pred_test = datos_evaluacion
        tabla = tabla_residuos(y_test, y_pred_test)

        resumen = resumen_errores(tabla)

        assert resumen["sesgo_log_medio"] == pytest.approx(tabla["residuo_log"].mean())
        assert resumen["sesgo_pct_medio"] == pytest.approx(tabla["error_relativo"].mean() * 100)
        assert resumen["error_pct_mediana"] == pytest.approx(
            tabla["error_relativo"].abs().median() * 100
        )

    def test_prediccion_exacta_da_sesgo_cero(
        self, datos_evaluacion: tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Series, np.ndarray]
    ) -> None:
        _, _, _, y_test, _ = datos_evaluacion

        resumen = resumen_errores(tabla_residuos(y_test, y_test))

        assert resumen["sesgo_log_medio"] == pytest.approx(0.0)
        assert resumen["error_pct_max"] == pytest.approx(0.0)


class TestMetricasPorSegmento:
    def test_una_fila_por_segmento_con_n_y_metricas(
        self, datos_evaluacion: tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Series, np.ndarray]
    ) -> None:
        val, y_val, y_pred_val, _, _ = datos_evaluacion

        resultado = metricas_por_segmento(val["tipo_propiedad"], y_val, y_pred_val)

        assert len(resultado) == val["tipo_propiedad"].dropna().nunique()
        assert resultado.index.name == "segmento"
        assert resultado["n"].sum() == len(val)
        assert resultado["n"].is_monotonic_decreasing
        assert {"n", "rmse_log", "rmse_usd", "mae_usd", "medae_usd", "mape_usd", "r2"} <= set(
            resultado.columns
        )

    @pytest.mark.filterwarnings("ignore:.*less than two samples.*")
    def test_con_na_conserva_el_segmento_desconocido(
        self, datos_evaluacion: tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Series, np.ndarray]
    ) -> None:
        """Los valores faltantes se agrupan aparte (dropna=False) sin perderse filas."""

        val, y_val, y_pred_val, _, _ = datos_evaluacion
        segmentos = val["tipo_propiedad"].copy()
        segmentos.iloc[0] = np.nan

        resultado = metricas_por_segmento(segmentos, y_val, y_pred_val)

        assert resultado["n"].sum() == len(val)
        assert resultado.index.hasnans


class TestBiasPorRangoPrecio:
    def test_bandas_balanceadas_y_limites_consistentes(
        self, datos_evaluacion: tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Series, np.ndarray]
    ) -> None:
        _, _, _, y_test, y_pred_test = datos_evaluacion
        tabla = tabla_residuos(y_test, y_pred_test)

        resultado = bias_por_rango_precio(
            tabla["precio_real_usd"], tabla["residuo_log"], n_bandas=5
        )

        assert len(resultado) <= 5
        assert resultado.index.name == "rango_precio"
        assert resultado["n"].sum() == len(tabla)
        assert (resultado["precio_min_usd"] <= resultado["precio_medio_usd"]).all()
        assert (resultado["precio_medio_usd"] <= resultado["precio_max_usd"]).all()
        assert resultado["sesgo_pct"].to_numpy() == pytest.approx(
            np.expm1(resultado["sesgo_log"].to_numpy()) * 100
        )

    def test_bandas_ordenadas_por_precio(
        self, datos_evaluacion: tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Series, np.ndarray]
    ) -> None:
        _, _, _, y_test, y_pred_test = datos_evaluacion
        tabla = tabla_residuos(y_test, y_pred_test)

        resultado = bias_por_rango_precio(
            tabla["precio_real_usd"], tabla["residuo_log"], n_bandas=5
        )

        assert resultado["precio_medio_usd"].is_monotonic_increasing


class TestGraficos:
    def test_graficos_devuelven_figuras_matplotlib(
        self, datos_evaluacion: tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Series, np.ndarray]
    ) -> None:
        val, y_val, y_pred_val, y_test, y_pred_test = datos_evaluacion
        tabla = tabla_residuos(y_test, y_pred_test)
        por_segmento = metricas_por_segmento(val["tipo_propiedad"], y_val, y_pred_val)
        por_rango = bias_por_rango_precio(tabla["precio_real_usd"], tabla["residuo_log"])

        figuras = [
            grafico_residuos(tabla),
            grafico_error_segmento(por_segmento),
            grafico_sesgo_rango(por_rango),
        ]

        for figura in figuras:
            assert isinstance(figura, Figure)
            assert figura.axes

        _cerrar_figuras(figuras)

    def test_guardar_figuras_escribe_png(
        self,
        tmp_path: Path,
        datos_evaluacion: tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Series, np.ndarray],
    ) -> None:
        val, y_val, y_pred_val, y_test, y_pred_test = datos_evaluacion
        tabla = tabla_residuos(y_test, y_pred_test)
        por_rango = bias_por_rango_precio(tabla["precio_real_usd"], tabla["residuo_log"])

        figuras = {
            "residuos": grafico_residuos(tabla),
            "sesgo_rango": grafico_sesgo_rango(por_rango),
        }

        rutas = guardar_figuras(figuras, tmp_path)

        assert len(rutas) == 2
        for ruta in rutas:
            assert ruta.exists()
            assert ruta.suffix == ".png"
            assert ruta.stat().st_size > 0

        _cerrar_figuras(list(figuras.values()))
