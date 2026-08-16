#!/usr/bin/env python3
"""
Exporta el modelo de serving como bundle (Fase 10).

Uso:
    python scripts/exportar_modelo.py
    python scripts/exportar_modelo.py --input data/processed/propiedades_argenprop_curado.csv
    python scripts/exportar_modelo.py --output models/modelo_precio_propiedades --random-state 42

Etapas:
    1. Carga del dataset curado, selección de columnas y target logarítmico.
    2. Split reproducible train / val / test (80 / 10 / 10).
    3. Preprocesamiento ajustado solo sobre train (sin fuga) y XGBoost.
    4. Guarda el bundle de serving con `real_estate.serving.persistencia`:
       modelo XGBoost, preprocesamiento, orden de features y metadata.

El bundle resultante lo consume el servicio FastAPI (fase 10) vía
`cargar_bundle`. No depende de MLflow: el modelo se serializa con el formato
nativo de xgboost.
"""

import argparse
import json
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
from real_estate.serving.persistencia import guardar_bundle  # noqa: E402

OUTPUT_DEFAULT = Path("models/modelo_precio_propiedades")


def exportar_modelo(
    input_file: str | Path,
    output_dir: str | Path = OUTPUT_DEFAULT,
    random_state: int = 42,
) -> Path:
    """Entrena el modelo final y guarda el bundle de serving en `output_dir`."""

    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

    print(f"Cargando dataset: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)

    df = crear_target_log(seleccionar_columnas(df))

    train, val, test = dividir_train_val_test(df, random_state=random_state)

    resultado = entrenar_y_evaluar(train, val, test, random_state=random_state)

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

    print("\n" + "=" * 70)
    print("BUNDLE DE SERVING EXPORTADO")
    print("=" * 70)
    print(f"\nDirectorio: {ruta}")
    print(f"Features: {len(columnas_features)}")
    print("Archivos: modelo_xgboost.json, preprocesamiento.json, features.json, metadata.json")

    resumen = ruta / "resumen_bundle.json"
    resumen.write_text(
        json.dumps(
            {"columnas_features": columnas_features, **metadata}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    print(f"Resumen: {resumen}")

    return ruta


def main() -> None:
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
    args = parser.parse_args()

    exportar_modelo(
        input_file=args.input,
        output_dir=args.output,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
