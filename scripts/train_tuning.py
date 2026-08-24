#!/usr/bin/env python3
"""
Entry point de la etapa de Tuning de XGBoost (Fase 5 del roadmap).

Uso:
    python scripts/train_tuning.py
    python scripts/train_tuning.py --input data/processed/propiedades_argenprop_curado.csv
    python scripts/train_tuning.py --metodo grid
    python scripts/train_tuning.py --metodo random --n-iter 50
    python scripts/train_tuning.py --no-tracking

Etapas:
    1. Carga del dataset curado, selección de columnas y target logarítmico.
    2. Split reproducible train / val / test (80 / 10 / 10).
    3. Preprocesamiento ajustado solo sobre train (sin fuga), igual que el
       resto del pipeline para que la comparación sea justa.
    4. Búsqueda de hiperparámetros con validación cruzada interna sobre
       train: `GridSearchCV` (exhaustivo sobre el grid reducido) o
       `RandomizedSearchCV` (muestreo de `--n-iter` trials del espacio
       completo). Optuna queda excluido por regla del proyecto.
    5. Evaluación del mejor candidato (refit sobre train) sobre val y test,
       comparando contra el XGBoost default.
    6. Tracking con MLflow: un run resumen más un run anidado por trial
       (parámetros, métricas de CV y ranking), sin versionar en el Model
       Registry (el champion se elige en la fase 6).

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
from real_estate.models.tuning import GRID_REDUCIDO, tunear_xgboost  # noqa: E402
from real_estate.tracking import configurar_tracking, registrar_tuning  # noqa: E402
from real_estate.utils.logging import configurar_logging  # noqa: E402

logger = logging.getLogger(__name__)


def tunear_xgboost_pipeline(
    input_file: str | Path,
    random_state: int = 42,
    metodo: str = "random",
    n_iter: int = 30,
    cv: int = 3,
    n_jobs: int = -1,
    no_tracking: bool = False,
) -> None:
    """Corre el pipeline de tuning de XGBoost completo y reporta el resultado."""

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

    resultado = tunear_xgboost(
        train,
        val,
        test,
        metodo=metodo,
        # Con 'grid' se usa el grid reducido (8 combinaciones); con 'random'
        # el espacio completo se muestrea con n_iter trials.
        espacio=GRID_REDUCIDO if metodo == "grid" else None,
        n_iter=n_iter,
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    if not no_tracking:
        run_id, trials = registrar_tuning(
            resultado,
            train,
            random_state=random_state,
            dataset_info=str(input_path),
            split_sizes={"train": len(train), "val": len(val), "test": len(test)},
        )
        logger.info(f"\nRun resumen MLflow: {run_id}")
        logger.info(f"Trials registrados: {len(trials)}")

    logger.info("\n" + "=" * 70)
    logger.info("TUNING DE XGBOOST FINALIZADO (fase 5)")
    logger.info("=" * 70)


def main() -> None:
    configurar_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Tuning de hiperparámetros de XGBoost: búsqueda con "
            "GridSearchCV/RandomizedSearchCV (sin fuga), comparación contra "
            "el default y tracking de experimentos con MLflow (fase 5)"
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
        "--metodo",
        choices=["grid", "random"],
        default="random",
        help=(
            "Método de búsqueda: 'grid' (exhaustivo sobre el grid reducido) "
            "o 'random' (muestreo de --n-iter trials del espacio completo; "
            "default: random)"
        ),
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="Trials a muestrear con RandomizedSearchCV (default: 30)",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Folds de la validación cruzada interna (default: 3)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Procesos paralelos de la búsqueda, -1 = todos los cores (default: -1)",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Desactiva el tracking con MLflow (por defecto está activo)",
    )
    args = parser.parse_args()

    tunear_xgboost_pipeline(
        input_file=args.input,
        random_state=args.random_state,
        metodo=args.metodo,
        n_iter=args.n_iter,
        cv=args.cv,
        n_jobs=args.n_jobs,
        no_tracking=args.no_tracking,
    )


if __name__ == "__main__":
    main()
