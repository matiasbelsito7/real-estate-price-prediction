#!/usr/bin/env python3
"""
Entry point de la explicabilidad de modelos con SHAP (Fase 7).

Uso:
    python scripts/explain.py
    python scripts/explain.py --input data/processed/propiedades_argenprop_features.csv
    python scripts/explain.py --output reports/figures/

Etapas:
    1. Carga del dataset de features y split train / val / test.
    2. Entrenamiento de XGBoost (reutiliza el pipeline de la fase 5).
    3. Cálculo de valores SHAP con TreeExplainer.
    4. Importancia global de features (media |SHAP|).
    5. Gráficos beeswarm (summary) y barras de importancia.
    6. Guardado de figuras en el directorio de salida.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Permite importar el paquete `real_estate` (layout src/) sin instalarlo.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from real_estate.explainability import (  # noqa: E402
    calcular_shap,
    grafico_barras,
    grafico_resumen,
    guardar_figuras,
)
from real_estate.features.pipeline import dividir_train_val_test  # noqa: E402
from real_estate.features.transformations import (  # noqa: E402
    crear_target_log,
    seleccionar_columnas,
)
from real_estate.models.entrenamiento import entrenar_y_evaluar  # noqa: E402
from real_estate.utils.logging import configurar_logging  # noqa: E402

logger = logging.getLogger(__name__)


def explicar_modelo(
    input_file: str | Path,
    random_state: int = 42,
    output_dir: str | Path = "reports/figures/",
) -> None:
    """Calcula SHAP y genera gráficos de explicabilidad."""

    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

    logger.info(f"Cargando dataset: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)

    df = crear_target_log(seleccionar_columnas(df))

    train, val, test = dividir_train_val_test(df, random_state=random_state)

    resultado = entrenar_y_evaluar(train, val, test, random_state=random_state)

    modelo_xgb = resultado.modelo_xgboost
    ajustes = resultado.ajustes

    # Reconstruir X_test aplicando el preprocesamiento
    from real_estate.models.entrenamiento import (
        aplicar_preprocesamiento,
        separar_features_target,
    )

    test_proc = aplicar_preprocesamiento(test, ajustes)
    x_test, _ = separar_features_target(test_proc)

    # --- SHAP ---
    logger.info("Calculando valores SHAP...")
    explicacion = calcular_shap(modelo_xgb, x_test)

    importancia = explicacion.importancia_global()
    logger.info("\nImportancia global (media |SHAP|):\n%s", importancia.to_string())

    # --- Figuras ---
    figuras: dict[str, Any] = {}

    figuras["shap_beeswarm"] = grafico_resumen(explicacion, x_test)
    figuras["shap_importancia"] = grafico_barras(explicacion)

    rutas = guardar_figuras(figuras, output_dir)
    logger.info("\nFiguras guardadas en %s:", output_dir)
    for ruta in rutas:
        logger.info("  %s", ruta)

    logger.info("\n" + "=" * 70)
    logger.info("EXPLICABILIDAD SHAP FINALIZADA (fase 7)")
    logger.info("=" * 70)


def main() -> None:
    configurar_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Explicabilidad: valores SHAP, importancia global y "
            "gráficos beeswarm / barras para el modelo XGBoost"
        )
    )
    parser.add_argument(
        "--input",
        default="data/processed/propiedades_argenprop_features.csv",
        help="CSV de features de entrada (default: data/processed/propiedades_argenprop_features.csv)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla para el split y el entrenamiento (default: 42)",
    )
    parser.add_argument(
        "--output",
        default="reports/figures/",
        help="Directorio de salida para las figuras (default: reports/figures/)",
    )
    args = parser.parse_args()

    explicar_modelo(
        input_file=args.input,
        random_state=args.random_state,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
