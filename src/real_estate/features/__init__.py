"""Capa de Feature Engineering.

Responsabilidades:
- `transformations`: selección de columnas, target logarítmico, codificación
  ordinal por mediana de precio y imputación por mediana.
- `pipeline`: orquestación de las etapas sobre un DataFrame (`construir_features`)
  y división train/val/test reproducible.
"""

__all__: list[str] = []
