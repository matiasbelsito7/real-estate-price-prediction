#!/usr/bin/env python3
"""
Exporta el modelo de serving como bundle (Fase 10).

Uso:
    python scripts/exportar_modelo.py
    python scripts/exportar_modelo.py --input data/processed/propiedades_argenprop_curado.csv
    python scripts/exportar_modelo.py --output models/modelo_precio_propiedades --random-state 42
    python scripts/exportar_modelo.py --no-tracking

Etapas:
    1. Carga del dataset curado, selección de columnas y target logarítmico.
    2. Split reproducible train / val / test (80 / 10 / 10).
    3. Preprocesamiento ajustado solo sobre train (sin fuga) y XGBoost.
    4. Guarda el bundle de serving con `real_estate.serving.persistencia`:
       modelo XGBoost, preprocesamiento, orden de features y metadata.
    5. Tracking con MLflow: registra la corrida con `registrar_resultado`
       (params, métricas, importancia de features y el modelo con firma),
       versionando el champion en el Model Registry.

El bundle resultante lo consume el servicio FastAPI (fase 10) vía
`cargar_bundle`. El modelo se serializa con el formato nativo de xgboost; el
tracking se puede desactivar con `--no-tracking` (útil para pruebas). El
tracking URI respeta `MLFLOW_TRACKING_URI` si está definido; si no, usa el
store local `mlruns/`.
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# Permite importar el paquete `real_estate` (layout src/) sin instalarlo.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from real_estate.features.pipeline import dividir_train_val_test  # noqa: E402
from real_estate.features.transformations import (  # noqa: E402
    crear_target_log,
    seleccionar_columnas,
)
from real_estate.models.entrenamiento import (  # noqa: E402
    aplicar_preprocesamiento,
    entrenar_y_evaluar,
    separar_features_target,
)
from real_estate.persistencia.bundle import guardar_bundle  # noqa: E402
from real_estate.tracking import (  # noqa: E402
    MODELO_DEFAULT,
    configurar_tracking,
    registrar_resultado,
)
from real_estate.utils.logging import configurar_logging  # noqa: E402

logger = logging.getLogger(__name__)

OUTPUT_DEFAULT = Path("models/modelo_precio_propiedades")


def exportar_modelo(
    input_file: str | Path,
    output_dir: str | Path = OUTPUT_DEFAULT,
    random_state: int = 42,
    no_tracking: bool = False,
) -> Path:
    """Entrena el modelo final y guarda el bundle de serving en `output_dir`."""

    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

    logger.info(f"Cargando dataset: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)

    df = crear_target_log(seleccionar_columnas(df))

    train, val, test = dividir_train_val_test(df, random_state=random_state)

    if not no_tracking:
        experimento = configurar_tracking()
        logger.info(f"\nTracking MLflow: experimento '{experimento}'")

    resultado = entrenar_y_evaluar(train, val, test, random_state=random_state)

    if not no_tracking:
        run_id, version = registrar_resultado(
            resultado,
            train,
            random_state=random_state,
            dataset_info=str(input_path),
            split_sizes={"train": len(train), "val": len(val), "test": len(test)},
        )
        logger.info(
            f"\nChampion registrado en el Model Registry: '{MODELO_DEFAULT}' versión {version}"
        )
        logger.info(f"Run MLflow: {run_id}")

    # Orden de las features que el modelo espera: el de la matriz preprocesada.
    train_proc = aplicar_preprocesamiento(train, resultado.ajustes)
    x_train, _ = separar_features_target(train_proc)
    columnas_features = list(x_train.columns)

    metadata: dict[str, object] = {
        "tipo_modelo": "xgboost",
        "target": "log_precio_usd",
        "random_state": random_state,
        "n_features": len(columnas_features),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "metricas_xgboost_test": resultado.metricas_xgboost_test,
        "metricas_xgboost_val": resultado.metricas_xgboost_val,
        "dataset_info": str(input_path),
        "fecha_exportacion": datetime.now(UTC).isoformat(),
    }

    ruta = guardar_bundle(
        directorio=output_dir,
        modelo=resultado.modelo_xgboost,
        ajustes=resultado.ajustes,
        columnas_features=columnas_features,
        metadata=metadata,
    )

    logger.info("\n" + "=" * 70)
    logger.info("BUNDLE DE SERVING EXPORTADO")
    logger.info("=" * 70)
    logger.info(f"\nDirectorio: {ruta}")
    logger.info(f"Features: {len(columnas_features)}")
    logger.info(
        "Archivos: modelo_xgboost.json, preprocesamiento.json, features.json, metadata.json"
    )

    resumen = ruta / "resumen_bundle.json"
    resumen.write_text(
        json.dumps(
            {"columnas_features": columnas_features, **metadata}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    logger.info(f"Resumen: {resumen}")

    return ruta


def main() -> None:
    configurar_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Exporta el bundle de serving (modelo XGBoost + preprocesamiento) "
            "que consume la API de predicción (fase 10)"
        )
    )
    parser.add_argument(
        "--input",
        default="data/processed/propiedades_argenprop_curado.csv",
        help="CSV curado de entrada (default: data/processed/propiedades_argenprop_curado.csv)",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DEFAULT),
        help="Directorio de salida del bundle (default: models/modelo_precio_propiedades)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla para el split y el entrenamiento (default: 42)",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Desactiva el tracking con MLflow (por defecto está activo)",
    )
    args = parser.parse_args()

    exportar_modelo(
        input_file=args.input,
        output_dir=args.output,
        random_state=args.random_state,
        no_tracking=args.no_tracking,
    )


if __name__ == "__main__":
    main()
