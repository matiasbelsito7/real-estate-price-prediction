"""
Explicabilidad de modelos (Fase 7).

Expone la API pública del subpaquete: `calcular_shap`, `grafico_resumen`,
`grafico_barras` y `guardar_figuras` (además del contenedor
`ExplicacionSHAP`).
"""

from real_estate.explainability.shap_analysis import (
    ExplicacionSHAP,
    calcular_shap,
    grafico_barras,
    grafico_resumen,
    guardar_figuras,
)

__all__ = [
    "ExplicacionSHAP",
    "calcular_shap",
    "grafico_barras",
    "grafico_resumen",
    "guardar_figuras",
]
