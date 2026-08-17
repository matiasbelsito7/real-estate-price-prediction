#!/usr/bin/env python3
"""
Comparación de corridas de MLflow para elegir el champion (Fase 6).

Uso:
    python scripts/comparar_runs.py
    python scripts/comparar_runs.py --metrica xgboost_test_rmse_log
    python scripts/comparar_runs.py --max-runs 50

Consulta el experimento configurado, arma una tabla de corridas con el valor
de la métrica elegida (mejor a peor) y muestra el champion: la corrida con el
mejor valor. El registro formal del champion en el Model Registry ocurre al
exportar el modelo de serving (`scripts/exportar_modelo.py`), no acá.

El tracking URI respeta `MLFLOW_TRACKING_URI` si está definido; si no, usa el
store local `mlruns/`.
"""

import argparse
import sys
from pathlib import Path

# Permite importar el paquete `real_estate` (layout src/) sin instalarlo.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from real_estate.tracking import (  # noqa: E402
    METRICA_DEFAULT,
    comparar_runs,
    configurar_tracking,
    elegir_champion,
)


def comparar(
    experimento: str,
    metrica: str,
    max_runs: int,
) -> None:
    """Muestra la tabla de corridas y el champion del experimento."""

    configurar_tracking(experimento=experimento)

    print(f"\nComparando corridas del experimento '{experimento}'")
    print(f"Métrica: {metrica} | Máx. corridas: {max_runs}")
    print("=" * 70)

    tabla = comparar_runs(experimento, metrica, max_runs)

    if tabla.empty:
        print(f"No hay corridas con la métrica '{metrica}' en el experimento '{experimento}'.")
        return

    print(f"\n{len(tabla)} corridas con la métrica '{metrica}' (mejor a peor):")
    print(tabla.to_string(index=False))

    champion = elegir_champion(experimento, metrica, max_runs)

    print("\n" + "=" * 70)
    print("CHAMPION")
    print("=" * 70)
    print(f"run_id:      {champion.run_id}")
    print(f"tipo_modelo: {champion.tipo_modelo}")
    print(f"métrica:     {champion.metrica}")
    print(f"valor:       {champion.valor:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Comparación de corridas de MLflow: tabla de corridas con la "
            "métrica elegida y champion (mejor corrida)"
        )
    )
    parser.add_argument(
        "--experimento",
        default=None,
        help=("Nombre del experimento de MLflow (default: 'prediccion_precios_propiedades')"),
    )
    parser.add_argument(
        "--metrica",
        default=METRICA_DEFAULT,
        help=f"Métrica a comparar (default: {METRICA_DEFAULT})",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=100,
        help="Máxima cantidad de corridas a traer (default: 100)",
    )
    args = parser.parse_args()

    comparar(
        experimento=args.experimento or "prediccion_precios_propiedades",
        metrica=args.metrica,
        max_runs=args.max_runs,
    )


if __name__ == "__main__":
    main()
