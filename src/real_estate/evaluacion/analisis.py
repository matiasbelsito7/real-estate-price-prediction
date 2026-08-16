"""
Evaluación profunda de modelos (Fase 8).

Profundiza la evaluación de la fase 5 (RMSE log / USD y R²) con análisis
orientados a *dónde* y *cuánto* se equivoca el modelo, siempre sobre el
target logarítmico `log_precio_usd`:

1. `metricas_detalladas` — métricas de error en log y en USD (MAE, MedAE,
   MAPE, RMSE, R²).
2. `tabla_residuos` / `resumen_errores` — residuos por observación
   (`pred - real` en log) y resumen de la distribución del error relativo.
3. `metricas_por_segmento` — error por segmento de cualquier columna
   (p. ej. `tipo_propiedad`, `barrio`, `ambientes`).
4. `bias_por_rango_precio` — sesgo (sobre/subestimación) por banda de
   precio real, con `pd.qcut` (mismo número de observaciones por banda).
5. Gráficos (`grafico_residuos`, `grafico_error_segmento`,
   `grafico_sesgo_rango`) y `guardar_figuras` para persistir PNG.

Convenciones de interpretación: un residuo log positivo significa que el
modelo *sobreestima* el precio. `error_relativo = exp(residuo_log) - 1` es
la desviación en tanto por uno (0.30 ~ 30 % arriba); en porcentaje se
multiplica por 100.

El módulo se mantiene desacoplado del entrenamiento: recibe los targets
reales y predichos (en log) ya calculados, de modo que se reutiliza
indistintamente sobre val o test y sobre cualquier modelo.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)


def metricas_detalladas(
    y_real_log: pd.Series | np.ndarray,
    y_pred_log: pd.Series | np.ndarray,
) -> dict[str, float]:
    """
    Métricas de error en log y en USD para un conjunto de predicciones.

    Incluye RMSE log (error relativo), RMSE / MAE / MedAE / MAPE en USD
    (deshaciendo el log con `exp`) y R² sobre el target logarítmico.
    """

    real_log = np.asarray(y_real_log, dtype=float)
    pred_log = np.asarray(y_pred_log, dtype=float)
    real_usd = np.exp(real_log)
    pred_usd = np.exp(pred_log)

    return {
        "rmse_log": float(root_mean_squared_error(real_log, pred_log)),
        "rmse_usd": float(root_mean_squared_error(real_usd, pred_usd)),
        "mae_usd": float(mean_absolute_error(real_usd, pred_usd)),
        "medae_usd": float(median_absolute_error(real_usd, pred_usd)),
        "mape_usd": float(mean_absolute_percentage_error(real_usd, pred_usd) * 100),
        "r2": float(r2_score(real_log, pred_log)),
    }


def tabla_residuos(
    y_real_log: pd.Series | np.ndarray,
    y_pred_log: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """
    Tabla de residuos por observación, en log y en USD.

    Columnas:
    - `precio_real_usd` / `precio_pred_usd`: precios deshaciendo el log;
    - `residuo_log`: `pred - real` en log (positivo = sobreestima);
    - `residuo_usd`: diferencia de precios en USD;
    - `error_relativo`: `exp(residuo_log) - 1` (desviación en tanto por uno).
    """

    real_log = np.asarray(y_real_log, dtype=float)
    pred_log = np.asarray(y_pred_log, dtype=float)
    real_usd = np.exp(real_log)
    pred_usd = np.exp(pred_log)

    return pd.DataFrame(
        {
            "precio_real_usd": real_usd,
            "precio_pred_usd": pred_usd,
            "residuo_log": pred_log - real_log,
            "residuo_usd": pred_usd - real_usd,
            "error_relativo": np.expm1(pred_log - real_log),
        }
    )


def resumen_errores(tabla: pd.DataFrame) -> pd.Series:
    """
    Resumen de la distribución del error de una tabla de residuos.

    Devuelve: sesgo medio (en log y en %), y mediana / p75 / p90 / p95 /
    máximo del error relativo **absoluto** en % (cuánto se desvía típicamente
    el modelo, sin importar el signo).
    """

    error_pct = tabla["error_relativo"].abs() * 100

    return pd.Series(
        {
            "n": len(tabla),
            "sesgo_log_medio": float(tabla["residuo_log"].mean()),
            "sesgo_pct_medio": float(tabla["error_relativo"].mean() * 100),
            "error_pct_mediana": float(error_pct.median()),
            "error_pct_p75": float(error_pct.quantile(0.75)),
            "error_pct_p90": float(error_pct.quantile(0.90)),
            "error_pct_p95": float(error_pct.quantile(0.95)),
            "error_pct_max": float(error_pct.max()),
        }
    )


def metricas_por_segmento(
    segmentos: pd.Series | np.ndarray,
    y_real_log: pd.Series | np.ndarray,
    y_pred_log: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """
    Métricas por segmento (p. ej. por `tipo_propiedad` o `barrio`).

    `segmentos` es la columna **original** (sin codificar) que define los
    grupos; se agrupa conservando los NaN (`dropna=False`) y se aplica
    `metricas_detalladas` a cada grupo. Devuelve una fila por segmento,
    ordenada por número de observaciones descendente.
    """

    datos = pd.DataFrame(
        {
            "segmento": np.asarray(segmentos),
            "real_log": np.asarray(y_real_log, dtype=float),
            "pred_log": np.asarray(y_pred_log, dtype=float),
        }
    )

    filas: dict[object, dict[str, float | int]] = {}
    for nombre, grupo in datos.groupby("segmento", dropna=False, observed=True):
        filas[nombre] = {
            "n": len(grupo),
            **metricas_detalladas(grupo["real_log"], grupo["pred_log"]),
        }

    resultado = pd.DataFrame.from_dict(filas, orient="index")
    resultado.index.name = "segmento"

    return resultado.sort_values("n", ascending=False)


def bias_por_rango_precio(
    precio_real_usd: pd.Series | np.ndarray,
    residuo_log: pd.Series | np.ndarray,
    n_bandas: int = 5,
) -> pd.DataFrame:
    """
    Sesgo por banda de precio real (cuantiles con `pd.qcut`).

    Cada banda agrupa el mismo número de observaciones. Reporta los límites
    de la banda, `n`, el sesgo medio en log y en % (`expm1(sesgo_log) * 100`)
    y el RMSE log intra-banda. Detecta sobre/subestimación sistemática por
    rango de precio (p. ej. sobreestimar lo barato y subestimar lo caro).
    """

    datos = pd.DataFrame(
        {
            "precio_real_usd": np.asarray(precio_real_usd, dtype=float),
            "residuo_log": np.asarray(residuo_log, dtype=float),
        }
    )

    datos["rango_precio"] = pd.qcut(
        datos["precio_real_usd"],
        q=n_bandas,
        duplicates="drop",
    )

    filas: dict[object, dict[str, float | int]] = {}
    for banda, grupo in datos.groupby("rango_precio", observed=True):
        sesgo_log = float(grupo["residuo_log"].mean())
        filas[banda] = {
            "n": len(grupo),
            "precio_min_usd": float(grupo["precio_real_usd"].min()),
            "precio_medio_usd": float(grupo["precio_real_usd"].mean()),
            "precio_max_usd": float(grupo["precio_real_usd"].max()),
            "sesgo_log": sesgo_log,
            "sesgo_pct": np.expm1(sesgo_log) * 100,
            "rmse_log": float(np.sqrt(np.mean(grupo["residuo_log"] ** 2))),
        }

    resultado = pd.DataFrame.from_dict(filas, orient="index")
    resultado.index.name = "rango_precio"

    return resultado


def grafico_residuos(tabla: pd.DataFrame) -> Figure:
    """
    Dos paneles: distribución del error relativo (%) y real vs. predicho.

    El histograma centra la mirada en el sesgo (pico desplazado de 0) y los
    extremos del error; el scatter enfrenta el precio predicho al real con
    la diagonal `y = x` como referencia de predicción perfecta.
    """

    figura, ejes = plt.subplots(1, 2, figsize=(13, 5))

    error_pct = tabla["error_relativo"] * 100
    ejes[0].hist(error_pct, bins=60, color="#1f77b4", edgecolor="white")
    ejes[0].axvline(0.0, color="black", linewidth=1)
    ejes[0].set_title("Distribución del error relativo (%)")
    ejes[0].set_xlabel("Error relativo (%)")
    ejes[0].set_ylabel("Frecuencia")

    limite = max(
        float(tabla["precio_real_usd"].max()),
        float(tabla["precio_pred_usd"].max()),
    )
    ejes[1].scatter(
        tabla["precio_real_usd"],
        tabla["precio_pred_usd"],
        s=12,
        alpha=0.5,
    )
    ejes[1].plot([0, limite], [0, limite], "r--", linewidth=1, label="y = x")
    ejes[1].set_title("Precio predicho vs. real (USD)")
    ejes[1].set_xlabel("Precio real (USD)")
    ejes[1].set_ylabel("Precio predicho (USD)")
    ejes[1].legend()

    figura.tight_layout()

    return figura


def grafico_error_segmento(
    tabla_segmentos: pd.DataFrame,
    metrica: str = "rmse_usd",
    max_display: int = 15,
) -> Figure:
    """
    Barras horizontales del error (por defecto RMSE USD) por segmento.

    Muestra los `max_display` segmentos con más observaciones, de menor a
    mayor error, para identificar los grupos donde el modelo falla más.
    """

    figura, eje = plt.subplots(figsize=(9, 6))

    tabla_segmentos[metrica].head(max_display).sort_values().plot.barh(
        ax=eje,
        color="#1f77b4",
    )
    eje.set_title(f"Error por segmento ({metrica})")
    eje.set_xlabel(metrica)

    figura.tight_layout()

    return figura


def grafico_sesgo_rango(tabla_bias: pd.DataFrame) -> Figure:
    """
    Barras del sesgo promedio (%) por rango de precio.

    Barras rojas sobreestiman (sesgo > 0) y azules subestiman (sesgo < 0);
    la línea en 0 separa ambas zonas. Complementa a `bias_por_rango_precio`.
    """

    figura, eje = plt.subplots(figsize=(9, 5))

    sesgo_pct = tabla_bias["sesgo_pct"]
    colores = ["#d62728" if sesgo > 0 else "#1f77b4" for sesgo in sesgo_pct]

    eje.bar(sesgo_pct.index.astype(str), sesgo_pct, color=colores)
    eje.axhline(0.0, color="black", linewidth=1)
    eje.set_title("Sesgo promedio por rango de precio (%)")
    eje.set_ylabel("Sesgo (%) — positivo = sobreestima")

    figura.tight_layout()

    return figura


def guardar_figuras(
    figuras: dict[str, Figure],
    directorio: Path | str,
) -> list[Path]:
    """
    Guarda las figuras como PNG en `directorio` (creándolo si hace falta).

    Cada entrada de `figuras` es `nombre -> figura`; los archivos resultan
    `{directorio}/{nombre}.png`. Devuelve las rutas escritas.
    """

    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)

    rutas: list[Path] = []
    for nombre, figura in figuras.items():
        ruta = directorio / f"{nombre}.png"
        figura.savefig(ruta, bbox_inches="tight", dpi=150)
        rutas.append(ruta)

    return rutas
