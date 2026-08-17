"""
Evaluación de nuevas publicaciones con el modelo exportado (Fase 2 del roadmap).

`evaluar_dataframe` encadena la misma curación del entrenamiento (en memoria,
sin persistir intermedios) con la predicción del bundle de serving sobre un
`DataFrame`:

1. Curación del dataset de nuevas (`real_estate.curation.pipeline`).
2. Carga del bundle de serving (`cargar_bundle`).
3. Predicción de `precio_predicho_usd` (deshace el log del target).
4. `fecha_prediccion` (fecha ISO del día de la corrida, inyectable).

`evaluar_nuevas` es el wrapper de CLI (CSV de entrada -> CSV evaluado) y
`scripts/evaluar_nuevas.py` el entry point de línea de comandos. El ETL
periódico (fase 12) reusa `evaluar_dataframe`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from real_estate.curation.pipeline import curar_dataset, mostrar_dataset, mostrar_dataset_curado
from real_estate.serving.persistencia import cargar_bundle

INPUT_DEFAULT = "data/raw/propiedades_nuevas.csv"
OUTPUT_DEFAULT = "data/processed/propiedades_nuevas_evaluadas.csv"
MODELO_DEFAULT = "models/modelo_precio_propiedades"

# Columnas del CSV evaluado: identificación, atributos, precio publicado (USD)
# y predicción. `precio`/`moneda` conservan el valor original del aviso.
COLUMNAS_SALIDA = [
    "id",
    "titulo",
    "link",
    "barrio",
    "tipo_propiedad",
    "superficie_cubierta",
    "ambientes",
    "dormitorios",
    "banos",
    "antiguedad",
    "expensas_usd",
    "precio",
    "moneda",
    "precio_usd",
    "precio_predicho_usd",
    "fecha_prediccion",
]


def evaluar_dataframe(
    df: pd.DataFrame,
    directorio_modelo: str | Path = MODELO_DEFAULT,
    *,
    fecha_prediccion: str | None = None,
) -> pd.DataFrame:
    """
    Cura el `DataFrame` de nuevas publicaciones y predice su precio en USD.

    La curación se hace en memoria (no se persiste el curado intermedio). El
    modelo se carga del bundle de serving exportado (`cargar_bundle`), de modo
    que la predicción reusa exactamente las features del entrenamiento. Agrega
    `precio_predicho_usd` y `fecha_prediccion` (fecha ISO del día, o la fecha
    inyectada vía `fecha_prediccion`) y devuelve el `DataFrame` evaluado.
    """

    df = curar_dataset(df)

    mostrar_dataset_curado(df)

    print("\n" + "=" * 70)
    print("PREDICCIÓN")
    print("=" * 70)

    modelo = cargar_bundle(directorio_modelo)

    print(f"\nModelo cargado: {directorio_modelo}")
    print(f"Features esperadas: {len(modelo.columnas_features)}")

    df["precio_predicho_usd"] = modelo.predecir_usd(df)

    df["fecha_prediccion"] = fecha_prediccion or datetime.now(UTC).date().isoformat()

    print(f"\nPredichas: {df['precio_predicho_usd'].notna().sum():,} de {len(df):,} publicaciones")
    print(f"Mediana del precio predicho: {df['precio_predicho_usd'].median():,.0f} USD")

    return df


def evaluar_nuevas(
    input_file: str | Path,
    output_file: str | Path,
    directorio_modelo: str | Path = MODELO_DEFAULT,
) -> None:
    """
    Cura el CSV de nuevas publicaciones, predice su precio y guarda el resultado.

    Lee el CSV crudo, delega la curación + predicción en `evaluar_dataframe` y
    exporta el dataset evaluado con las columnas de `COLUMNAS_SALIDA`.
    """

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

    print(f"Cargando dataset de nuevas publicaciones: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)

    mostrar_dataset(df)

    df = evaluar_dataframe(df, directorio_modelo)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df[COLUMNAS_SALIDA].to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("EVALUACIÓN FINALIZADA")
    print("=" * 70)

    print(f"\nDataset evaluado guardado en:\n{output_path}")
