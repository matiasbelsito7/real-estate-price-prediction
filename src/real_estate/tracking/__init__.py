"""Tracking de experimentos con MLflow (Fase 6).

Expone la API pública del subpaquete: `configurar_tracking`,
`registrar_resultado`, `registrar_lineales`, `registrar_tuning`,
`finalizar_corrida` y la comparación de corridas para elegir el champion
(`comparar_runs`, `elegir_champion`).
"""

from real_estate.tracking.comparacion import (
    METRICA_DEFAULT,
    Champion,
    comparar_runs,
    elegir_champion,
)
from real_estate.tracking.experimentos import (
    EXPERIMENTO_DEFAULT,
    MODELO_DEFAULT,
    configurar_tracking,
    finalizar_corrida,
    registrar_lineales,
    registrar_resultado,
    registrar_tuning,
)

__all__ = [
    "Champion",
    "EXPERIMENTO_DEFAULT",
    "METRICA_DEFAULT",
    "MODELO_DEFAULT",
    "comparar_runs",
    "configurar_tracking",
    "elegir_champion",
    "finalizar_corrida",
    "registrar_lineales",
    "registrar_resultado",
    "registrar_tuning",
]
