"""
Persistencia del bundle de serving (modelo + preprocesamiento).

El bundle guardado por `scripts/exportar_modelo.py` en
`models/modelo_precio_propiedades/` contiene:

- `modelo_xgboost.json`: el modelo XGBoost (formato nativo de xgboost).
- `preprocesamiento.json`: ordenes ordinales e imputador (`Preprocesamiento`).
- `features.json`: orden de las features del entrenamiento.
- `metadata.json`: métricas y parámetros del entrenamiento.
- `checksum.json`: hashes SHA-256 de los archivos del bundle para verificación.

`cargar_bundle` lo reconstruye en un `ModeloPrediccion` listo para predecir.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from xgboost import XGBRegressor

from real_estate.models.entrenamiento import Preprocesamiento
from real_estate.serving.modelo import ModeloPrediccion

NOMBRE_MODELO = "modelo_xgboost.json"
NOMBRE_PREPROCESAMIENTO = "preprocesamiento.json"
NOMBRE_FEATURES = "features.json"
NOMBRE_METADATA = "metadata.json"
NOMBRE_CHECKSUM = "checksum.json"


def _calcular_checksum(ruta: Path) -> str:
    """Calcula el hash SHA-256 de un archivo."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


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

    (directorio / NOMBRE_METADATA).write_text(
        json.dumps(metadata or {}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Genera checksum SHA-256 de los archivos del bundle.
    archivos_bundle = [NOMBRE_MODELO, NOMBRE_PREPROCESAMIENTO, NOMBRE_FEATURES, NOMBRE_METADATA]
    checksums = {}
    for nombre in archivos_bundle:
        ruta_archivo = directorio / nombre
        if ruta_archivo.exists():
            checksums[nombre] = _calcular_checksum(ruta_archivo)

    (directorio / NOMBRE_CHECKSUM).write_text(
        json.dumps(checksums, indent=2),
        encoding="utf-8",
    )

    return directorio


def cargar_bundle(directorio: str | Path) -> ModeloPrediccion:
    """Carga el bundle de serving y devuelve un `ModeloPrediccion` listo.

    Verifica la integridad de los archivos usando el checksum SHA-256.
    """

    directorio = Path(directorio)

    # Verifica checksum si existe.
    ruta_checksum = directorio / NOMBRE_CHECKSUM
    if ruta_checksum.exists():
        checksums_esperados: dict[str, str] = json.loads(ruta_checksum.read_text(encoding="utf-8"))
        for nombre, hash_esperado in checksums_esperados.items():
            ruta_archivo = directorio / nombre
            if ruta_archivo.exists():
                hash_real = _calcular_checksum(ruta_archivo)
                if hash_real != hash_esperado:
                    raise ValueError(
                        f"Checksum mismatch en {nombre}: esperado {hash_esperado}, real {hash_real}"
                    )

    modelo = XGBRegressor()
    modelo.load_model(directorio / NOMBRE_MODELO)

    preprocesamiento_raw = json.loads(
        (directorio / NOMBRE_PREPROCESAMIENTO).read_text(encoding="utf-8")
    )
    ajustes = Preprocesamiento(
        ordenes=preprocesamiento_raw["ordenes"],
        imputador=preprocesamiento_raw["imputador"],
    )

    columnas_features: list[str] = json.loads(
        (directorio / NOMBRE_FEATURES).read_text(encoding="utf-8")
    )

    metadata: dict[str, object] = json.loads(
        (directorio / NOMBRE_METADATA).read_text(encoding="utf-8")
    )

    return ModeloPrediccion(
        modelo_xgboost=modelo,
        ajustes=ajustes,
        columnas_features=columnas_features,
        metadata=metadata,
    )
