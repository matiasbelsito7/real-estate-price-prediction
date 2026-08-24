#!/usr/bin/env python3
"""
Entry point de la evaluación profunda de modelos (Fase 8).

Uso:
    python scripts/evaluate.py
    python scripts/evaluate.py --input data/processed/propiedades_argenprop_features.csv
    python scripts/evaluate.py --no-tracking

Etapas:
    1. Carga del dataset de features y split train / val / test.
    2. Entrenamiento de XGBoost (reutiliza el pipeline de la fase 5).
    3. Métricas detalladas (MAE, MedAE, MAPE, RMSE USD, R²).
    4. Análisis de residuos y distribución del error relativo.
    5. Métricas por segmento (tipo_propiedad, barrio, ambientes).
    6. Sesgo por rango de precio (sobre/subestimación por banda).
    7. Guardado de figuras en reports/figures/.
    8. Tracking con MLflow (opcional).
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

from real_estate.evaluacion import (  # noqa: E402
    bias_por_rango_precio,
    grafico_error_segmento,
    grafico_residuos,
    grafico_sesgo_rango,
    guardar_figuras,
    metricas_detalladas,
    metricas_por_segmento,
    resumen_errores,
    tabla_residuos,
)
from real_estate.features.pipeline import dividir_train_val_test  # noqa: E402
from real_estate.features.transformations import (  # noqa: E402
    TARGET_PRECIO,
    crear_target_log,
    seleccionar_columnas,
)
from real_estate.models.entrenamiento import entrenar_y_evaluar  # noqa: E402
from real_estate.tracking import configurar_tracking  # noqa: E402
from real_estate.utils.logging import configurar_logging  # noqa: E402

logger = logging.getLogger(__name__)

FIGURES_DIR = Path("reports/figures")


def evaluar_modelo(
    input_file: str | Path,
    random_state: int = 42,
    no_tracking: bool = False,
) -> None:
    """Corre la evaluación profunda del modelo XGBoost."""

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

    # Reconstruir X_test / y_test aplicando el preprocesamiento
    from real_estate.models.entrenamiento import (
        aplicar_preprocesamiento,
        separar_features_target,
    )

    test_proc = aplicar_preprocesamiento(test, ajustes)
    x_test, y_test = separar_features_target(test_proc)

    y_pred = pd.Series(modelo_xgb.predict(x_test), index=y_test.index)

    # --- Métricas detalladas ---
    metricas = metricas_detalladas(y_test, y_pred)
    logger.info("\nMétricas detalladas (test):")
    for clave, valor in metricas.items():
        logger.info("  %-15s: %s", clave, f"{valor:,.4f}" if isinstance(valor, float) else valor)

    # --- Residuos ---
    tabla_res = tabla_residuos(y_test, y_pred)
    resumen = resumen_errores(tabla_res)
    logger.info("\nResumen de errores (test):")
    for idx in resumen.index:
        valor = resumen[idx]
        logger.info("  %-20s: %s", idx, f"{valor:,.2f}" if isinstance(valor, float) else valor)

    # --- Métricas por segmento ---
    if "tipo_propiedad" in test.columns:
        segmento_tipo = metricas_por_segmento(
            test["tipo_propiedad"],
            y_test,
            y_pred,
        )
        logger.info("\nMétricas por tipo_propiedad (test):\n%s", segmento_tipo.to_string())

    if "barrio" in test.columns:
        segmento_barrio = metricas_por_segmento(
            test["barrio"],
            y_test,
            y_pred,
        )
        logger.info(
            "\nMétricas por barrio (test, top 15):\n%s", segmento_barrio.head(15).to_string()
        )

    if "ambientes" in test.columns:
        segmento_ambientes = metricas_por_segmento(
            test["ambientes"].astype(str).rename("ambientes"),
            y_test,
            y_pred,
        )
        logger.info("\nMétricas por ambientes (test):\n%s", segmento_ambientes.to_string())

    # --- Sesgo por rango de precio ---
    if TARGET_PRECIO in test.columns:
        precio_real_usd = test[TARGET_PRECIO]
        residuo_log = y_pred - y_test
        tabla_bias = bias_por_rango_precio(precio_real_usd, residuo_log)
        logger.info("\nSesgo por rango de precio (test):\n%s", tabla_bias.to_string())

    # --- Figuras ---
    figuras: dict[str, Any] = {}

    figuras["residuos"] = grafico_residuos(tabla_res)

    if "tipo_propiedad" in test.columns:
        figuras["error_segmento_tipo"] = grafico_error_segmento(segmento_tipo)

    if TARGET_PRECIO in test.columns:
        figuras["sesgo_rango_precio"] = grafico_sesgo_rango(tabla_bias)

    rutas = guardar_figuras(figuras, FIGURES_DIR)
    logger.info("\nFiguras guardadas en %s:", FIGURES_DIR)
    for ruta in rutas:
        logger.info("  %s", ruta)

    # --- Tracking ---
    if not no_tracking:
        configurar_tracking()
        logger.info("\nTracking MLflow habilitado para evaluación.")

    logger.info("\n" + "=" * 70)
    logger.info("EVALUACIÓN PROFUNDA FINALIZADA (fase 8)")
    logger.info("=" * 70)


def main() -> None:
    configurar_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluación profunda: métricas detalladas, residuos, "
            "segmentación y sesgo por rango de precio"
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
        "--no-tracking",
        action="store_true",
        help="Desactiva el tracking con MLflow (por defecto está activo)",
    )
    args = parser.parse_args()

    evaluar_modelo(
        input_file=args.input,
        random_state=args.random_state,
        no_tracking=args.no_tracking,
    )


if __name__ == "__main__":
    main()
