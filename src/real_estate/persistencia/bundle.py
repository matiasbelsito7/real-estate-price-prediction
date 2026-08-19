"""
Persistencia del bundle de serving (modelo + preprocesamiento).

El bundle guardado por `scripts/exportar_modelo.py` en
`models/modelo_precio_propiedades/` contiene:

- `modelo_xgboost.json`: el modelo XGBoost (formato nativo de xgboost).
- `preprocesamiento.json`: ordenes ordinales e imputador (`Preprocesamiento`).
- `features.json`: orden de las features del entrenamiento.
- `metadata.json`: métricas y parámetros del entrenamiento.

`cargar_bundle` lo reconstruye en un `ModeloPrediccion` listo para predecir.
"""

from __future__ import annotations

import json
from pathlib import Path

from xgboost import XGBRegressor

from real_estate.models.entrenamiento import Preprocesamiento
from real_estate.serving.modelo import ModeloPrediccion

NOMBRE_MODELO = "modelo_xgboost.json"
NOMBRE_PREPROCESAMIENTO = "preprocesamiento.json"
NOMBRE_FEATURES = "features.json"
NOMBRE_METADATA = "metadata.json"


def guardar_bundle(
    directorio: str | Path,
    modelo: XGBRegressor,
    ajustes: Preprocesamiento,
    columnas_features: list[str],
    metadata: dict[str, object] | None = None,
) -> Path:
    """
    Guarda el bundle de serving en `directorio` y devuelve su ruta.

    Crea el directorio si no existe. El modelo se serializa con el formato
    nativo de xgboost; el resto de los componentes como JSON UTF-8.
    """

    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)

    modelo.save_model(directorio / NOMBRE_MODELO)

    (directorio / NOMBRE_PREPROCESAMIENTO).write_text(
        json.dumps(
            {"ordenes": ajustes.ordenes, "imputador": ajustes.imputador},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (directorio / NOMBRE_FEATURES).write_text(
        json.dumps(columnas_features, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if metadata:
        (directorio / NOMBRE_METADATA).write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return directorio


def cargar_bundle(directorio: str | Path) -> ModeloPrediccion:
    """
    Carga un bundle de serving desde `directorio`.

    Levanta el modelo XGBoost, el preprocesamiento y el orden de las features.
    """

    directorio = Path(directorio)

    modelo = XGBRegressor()
    modelo.load_model(directorio / NOMBRE_MODELO)

    ajustes_raw = json.loads((directorio / NOMBRE_PREPROCESAMIENTO).read_text())
    ajustes = Preprocesamiento(
        ordenes=ajustes_raw["ordenes"],
        imputador=ajustes_raw["imputador"],
    )

    columnas_features = json.loads((directorio / NOMBRE_FEATURES).read_text())

    metadata = {}
    if (directorio / NOMBRE_METADATA).exists():
        metadata = json.loads((directorio / NOMBRE_METADATA).read_text())

    return ModeloPrediccion(
        modelo_xgboost=modelo,
        ajustes=ajustes,
        columnas_features=columnas_features,
        metadata=metadata,
    )
