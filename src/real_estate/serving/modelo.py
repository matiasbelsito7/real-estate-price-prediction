"""
Servicio de predicción del modelo entrenado (Fase 10).

`ModeloPrediccion` envuelve el modelo XGBoost y el preprocesamiento aprendido
sobre train (`Preprocesamiento`) y expone `predecir_log` / `predecir_usd`.
Replica exactamente el pipeline de entrenamiento:

    seleccionar_columnas -> aplicar_preprocesamiento ->
    separar_features_target -> predict -> exp

de modo que una propiedad nueva pasa por las mismas transformaciones que el
train (codificación ordinal de categóricas e imputación por mediana) antes de
llegar al modelo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from real_estate.features.transformations import seleccionar_columnas
from real_estate.models.entrenamiento import Preprocesamiento, aplicar_preprocesamiento


@dataclass
class ModeloPrediccion:
    """Modelo + preprocesamiento listos para predecir sobre nuevas propiedades.

    - `modelo_xgboost`: regresor XGBoost entrenado sobre `log_precio_usd`.
    - `ajustes`: preprocesamiento aprendido sobre train (ordenes ordinales e
      imputador por mediana).
    - `columnas_features`: orden de las features que espera el modelo (el
      orden del entrenamiento).
    - `metadata`: métricas y parámetros del entrenamiento (opcional).
    """

    modelo_xgboost: XGBRegressor
    ajustes: Preprocesamiento
    columnas_features: list[str]
    metadata: dict[str, object] = field(default_factory=dict)

    def _construir_matriz(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replica el pipeline de entrenamiento sobre una propiedad nueva.

        Devuelve la matriz de features en el orden exacto del entrenamiento
        (`self.columnas_features`), con las categóricas codificadas y las
        numéricas imputadas. La entrada del servicio no trae target
        (`precio_usd` / `log_precio_usd`), por eso se seleccionan directo las
        columnas del modelo en vez de `separar_features_target`.
        """

        df = seleccionar_columnas(df)
        df = aplicar_preprocesamiento(df, self.ajustes)

        return df[self.columnas_features]

    def predecir_log(self, df: pd.DataFrame) -> np.ndarray:
        """Predice `log_precio_usd` para las propiedades de `df`."""

        x_matriz = self._construir_matriz(df)

        return np.asarray(self.modelo_xgboost.predict(x_matriz))

    def predecir_usd(self, df: pd.DataFrame) -> np.ndarray:
        """Predice `precio_usd` (deshaciendo el log) para las propiedades de `df`."""

        return np.exp(self.predecir_log(df))
