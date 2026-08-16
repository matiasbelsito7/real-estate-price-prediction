"""Tracking de experimentos con MLflow (Fase 6).

Expone la API pública del subpaquete: `configurar_tracking`,
`registrar_resultado`, `registrar_lineales` y `finalizar_corrida`.
"""

from real_estate.tracking.experimentos import (
    EXPERIMENTO_DEFAULT,
    MODELO_DEFAULT,
    configurar_tracking,
    finalizar_corrida,
    registrar_lineales,
    registrar_resultado,
)

__all__ = [
    "EXPERIMENTO_DEFAULT",
    "MODELO_DEFAULT",
    "configurar_tracking",
    "finalizar_corrida",
    "registrar_lineales",
    "registrar_resultado",
]
