"""
Comparación de corridas de MLflow para elegir el champion (Fase 6).

Consulta el experimento de MLflow y devuelve una tabla de corridas con una
métrica de interés (p. ej. RMSE log sobre test), ordenada de mejor a peor,
y una función para elegir el champion: la corrida con el mejor valor.

El champion es la corrida ganadora (run_id + valor de la métrica); el
registro formal en el Model Registry ocurre al exportar el modelo de
serving (fase 6, `scripts/exportar_modelo.py`), no acá.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd
from mlflow.entities import Run, ViewType
from mlflow.tracking import MlflowClient

from real_estate.tracking.experimentos import EXPERIMENTO_DEFAULT

#: Métrica por defecto para comparar (RMSE log del XGBoost sobre test).
METRICA_DEFAULT = "xgboost_test_rmse_log"

#: Máxima cantidad de corridas a traer del experimento.
MAX_RUNS_DEFAULT = 100


@dataclass(frozen=True)
class Champion:
    """Corrida ganadora de la comparación.

    - `run_id`: corrida de MLflow con la mejor métrica.
    - `metrica`: nombre de la métrica comparada.
    - `valor`: valor de la métrica en la corrida ganadora.
    - `tipo_modelo`: parámetro `tipo_modelo` de la corrida (p. ej. `xgboost`).
    """

    run_id: str
    metrica: str
    valor: float
    tipo_modelo: str


def _buscar_runs(
    experimento: str,
    max_runs: int,
) -> list[Run]:
    """Corridas activas del experimento, más recientes primero."""

    cliente = MlflowClient()
    experimento_obj = cliente.get_experiment_by_name(experimento)

    if experimento_obj is None:
        raise ValueError(f"experimento '{experimento}' no existe")

    corridas = cliente.search_runs(
        experiment_ids=[experimento_obj.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=max_runs,
    )
    return cast(list[Run], corridas)


def _valor_metrica(run: Run, metrica: str) -> float | None:
    """Valor de la métrica en la corrida, o `None` si no está logueada."""

    valor = run.data.metrics.get(metrica)
    if valor is None:
        return None
    return float(valor)


def comparar_runs(
    experimento: str = EXPERIMENTO_DEFAULT,
    metrica: str = METRICA_DEFAULT,
    max_runs: int = MAX_RUNS_DEFAULT,
) -> pd.DataFrame:
    """
    Tabla de corridas del experimento con el valor de `metrica`.

    Devuelve un DataFrame con `run_id`, `tipo_modelo` (del param) y
    `valor` de la métrica, ordenado de mejor a peor (menor valor primero,
    ya que RMSE se minimiza). Las corridas sin la métrica se omiten.
    """

    filas: list[dict[str, object]] = []

    for run in _buscar_runs(experimento, max_runs):
        valor = _valor_metrica(run, metrica)
        if valor is None:
            continue

        filas.append(
            {
                "run_id": run.info.run_id,
                "tipo_modelo": run.data.params.get("tipo_modelo", ""),
                "valor": valor,
            }
        )

    if not filas:
        return pd.DataFrame(columns=["run_id", "tipo_modelo", "valor"])

    return pd.DataFrame(filas).sort_values("valor", ascending=True).reset_index(drop=True)


def elegir_champion(
    experimento: str = EXPERIMENTO_DEFAULT,
    metrica: str = METRICA_DEFAULT,
    max_runs: int = MAX_RUNS_DEFAULT,
) -> Champion:
    """
    El champion: la corrida con el menor valor de `metrica`.

    Es la corrida ganadora de la comparación; el registro formal en el
    Model Registry se hace al exportar el modelo de serving (fase 6).
    """

    tabla = comparar_runs(experimento, metrica, max_runs)

    if tabla.empty:
        raise ValueError(
            f"no hay corridas con la métrica '{metrica}' en el experimento '{experimento}'"
        )

    ganadora = tabla.iloc[0]

    return Champion(
        run_id=str(ganadora["run_id"]),
        metrica=metrica,
        valor=float(ganadora["valor"]),
        tipo_modelo=str(ganadora["tipo_modelo"]),
    )
