"""
Evaluación profunda de modelos (Fase 8).

Expone la API pública del subpaquete: `metricas_detalladas`,
`tabla_residuos`, `resumen_errores`, `metricas_por_segmento`,
`bias_por_rango_precio` y los gráficos / `guardar_figuras`.
"""

from real_estate.evaluacion.analisis import (
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

__all__ = [
    "bias_por_rango_precio",
    "grafico_error_segmento",
    "grafico_residuos",
    "grafico_sesgo_rango",
    "guardar_figuras",
    "metricas_detalladas",
    "metricas_por_segmento",
    "resumen_errores",
    "tabla_residuos",
]
