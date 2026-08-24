#!/usr/bin/env python3
"""
Entry point de la etapa de Modelos lineales (Fase 4 del roadmap).

Uso:
    python scripts/train_lineales.py
    python scripts/train_lineales.py --input data/processed/propiedades_argenprop_curado.csv
    python scripts/train_lineales.py --alpha-lasso 0.1 --alpha-ridge 10.0
    python scripts/train_lineales.py --no-tracking

Etapas:
    1. Carga del dataset curado, selección de columnas y target logarítmico.
    2. Split reproducible train / val / test (80 / 10 / 10).
    3. Preprocesamiento ajustado solo sobre train (sin fuga), igual que
       XGBoost para que la comparación sea justa.
    4. Lasso y Ridge entrenados sobre train con escalado (`StandardScaler`)
       dentro del pipeline (los lineales son sensibles a la escala; los
       árboles no).
    5. Evaluación sobre val (comparación) y sobre test (mejor modelo).
    6. Tracking con MLflow: una corrida por modelo (parámetros, métricas,
       artefacto JSON de resumen y pipeline con firma), sin versionar en el
       Model Registry (el champion se elige en la fase 6).

El tracking se puede desactivar con `--no-tracking` (útil para pruebas). El
tracking URI respeta `MLFLOW_TRACKING_URI` si está definido; si no, usa el
store local `mlruns/`.
"""

import argparse
import logging
import sys
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
from real_estate.models.modelos_lineales import entrenar_y_evaluar_lineales  # noqa: E402
from real_estate.tracking import configurar_tracking, registrar_lineales  # noqa: E402
from real_estate.utils.logging import configurar_logging  # noqa: E402

logger = logging.getLogger(__name__)


def entrenar_lineales_pipeline(
    input_file: str | Path,
    random_state: int = 42,
    alpha_lasso: float = 1.0,
    alpha_ridge: float = 1.0,
    no_tracking: bool = False,
) -> None:
    """Corre el pipeline de modelos lineales completo y reporta las métricas."""

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

    resultado = entrenar_y_evaluar_lineales(
        train,
        val,
        test,
        alpha_lasso=alpha_lasso,
        alpha_ridge=alpha_ridge,
        random_state=random_state,
    )

    if not no_tracking:
        runs = registrar_lineales(
            resultado,
            train,
            random_state=random_state,
            dataset_info=str(input_path),
            split_sizes={"train": len(train), "val": len(val), "test": len(test)},
        )
        for nombre, run_id in runs:
            logger.info(f"Run MLflow ({nombre}): {run_id}")

    logger.info("\n" + "=" * 70)
    logger.info("MODELOS LINEALES FINALIZADOS (fase 4)")
    logger.info("=" * 70)


def main() -> None:
    configurar_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Modelos lineales: pipeline train/val/test sin fuga, Lasso y Ridge "
            "con escalado, y tracking de experimentos con MLflow (fase 4)"
        )
    )
    parser.add_argument(
        "--input",
        default="data/processed/propiedades_argenprop_curado.csv",
        help=("CSV curado de entrada (default: data/processed/propiedades_argenprop_curado.csv)"),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla para el split y el entrenamiento (default: 42)",
    )
    parser.add_argument(
        "--alpha-lasso",
        type=float,
        default=1.0,
        help="Regularización de Lasso (default: 1.0)",
    )
    parser.add_argument(
        "--alpha-ridge",
        type=float,
        default=1.0,
        help="Regularización de Ridge (default: 1.0)",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Desactiva el tracking con MLflow (por defecto está activo)",
    )
    args = parser.parse_args()

    entrenar_lineales_pipeline(
        input_file=args.input,
        random_state=args.random_state,
        alpha_lasso=args.alpha_lasso,
        alpha_ridge=args.alpha_ridge,
        no_tracking=args.no_tracking,
    )


if __name__ == "__main__":
    main()
