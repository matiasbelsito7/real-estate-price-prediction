"""Capa de curado de datos (Data Curation).

Responsabilidades:
- `cleaning`: limpieza y conversión de tipos (texto -> número).
- `transformations`: normalización de moneda (ARS->USD con tipo de cambio
  histórico) y creación de indicadores de valores informados.
- `validation`: validación de coherencia del dataset (precio, superficie,
  ambientes).
- `pipeline`: orquestación de las etapas sobre un DataFrame.
"""

__all__: list[str] = []
