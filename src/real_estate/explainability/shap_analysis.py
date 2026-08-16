"""
Explicabilidad de modelos con SHAP (Fase 7).

Interpreta las predicciones del XGBoost de la fase 5 en el espacio del target
`log_precio_usd`: cada valor SHAP es la contribución de una feature al
logaritmo del precio predicho, de modo que la suma de contribuciones más el
valor base reproduce la predicción (propiedad aditiva). Para interpretar en
USD, `exp(contribución)` es el factor multiplicativo sobre el precio.

API pública:

1. `calcular_shap` — explica un modelo XGBoost ajustado sobre un conjunto de
   datos con `shap.TreeExplainer` y devuelve un `ExplicacionSHAP`.
2. `ExplicacionSHAP.importancia_global` — media del |SHAP| por feature.
3. `grafico_resumen` / `grafico_barras` — figuras matplotlib (beeswarm y
   barras de importancia) para el análisis.
4. `guardar_figuras` — persiste las figuras como PNG en un directorio.

El módulo se mantiene desacoplado del entrenamiento: recibe el modelo ya
ajustado y la matriz de features ya preprocesada (misma codificación que
usó el modelo), evitando así fuga o transformaciones inconsistentes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.figure import Figure
from xgboost import XGBRegressor


@dataclass(frozen=True)
class ExplicacionSHAP:
    """Valores SHAP de un conjunto de datos (una fila por observación).

    - `valores`: matriz (n_filas, n_features) con las contribuciones al
      target logarítmico. Cumple `valores.sum(axis=1) + base ≈ predicción`.
    - `base`: valor esperado del modelo (log precio de referencia, el
      "ancho de vía" del que parten todas las contribuciones).
    - `nombres`: nombres de las features en el orden de las columnas de
      `valores`.
    """

    valores: np.ndarray
    base: float
    nombres: tuple[str, ...]

    def importancia_global(self) -> pd.Series:
        """Importancia global: media del |SHAP| por feature, ordenada desc."""
        importancia = np.abs(self.valores).mean(axis=0)
        return pd.Series(importancia, index=list(self.nombres)).sort_values(ascending=False)


def calcular_shap(modelo: XGBRegressor, X: pd.DataFrame) -> ExplicacionSHAP:
    """
    Explica las predicciones de `modelo` sobre `X` con `TreeExplainer`.

    `X` debe ser la matriz preprocesada (misma codificación ordinal e
    imputación con la que se entrenó `modelo`). Devuelve un
    `ExplicacionSHAP` con los valores, el valor base y los nombres de
    features.
    """

    explainer = shap.TreeExplainer(modelo)
    explicacion = explainer(X)

    valores = np.asarray(explicacion.values, dtype=float)
    base = float(np.asarray(explicacion.base_values).ravel()[0])

    return ExplicacionSHAP(valores=valores, base=base, nombres=tuple(X.columns))


def grafico_resumen(
    explicacion: ExplicacionSHAP,
    X: pd.DataFrame,
    max_display: int = 15,
) -> Figure:
    """
    Beeswarm de SHAP: muestra la distribución de contribuciones por feature.

    Cada punto es una observación; el color indica el valor de la feature
    (rojo alto, azul bajo). Las features aparecen ordenadas por importancia
    global, de arriba hacia abajo.
    """

    shap.summary_plot(
        explicacion.valores,
        X,
        feature_names=list(explicacion.nombres),
        max_display=max_display,
        show=False,
    )

    figura = plt.gcf()
    figura.set_size_inches(10, 7)

    return figura


def grafico_barras(
    explicacion: ExplicacionSHAP,
    max_display: int = 15,
) -> Figure:
    """Barras horizontales de importancia global (media |SHAP| por feature)."""

    figura, eje = plt.subplots(figsize=(8, 5))
    explicacion.importancia_global().head(max_display).plot.barh(ax=eje, color="#1f77b4")
    eje.set_title("Importancia global de features (media |SHAP|)")
    eje.set_xlabel("Media |SHAP| (log precio)")

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
