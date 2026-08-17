"""
ETL periódico de oportunidades de compra (Fase 12 del roadmap).

`ejecutar_etl` orquesta el pipeline automatizado sobre el CSV de nuevas
publicaciones scrapeadas (`data/raw/propiedades_nuevas.csv`):

1. Dedup contra PostgreSQL: solo se procesan los ids que no están persistidos
   todavía ("solo nuevas").
2. Curación + predicción con el bundle del champion (`evaluar_dataframe`).
3. Clasificación por propiedad (`clasificar_por_diferencia`), más el
   `ratio_precio` del ranking de ofertas (`clasificar_oportunidades`).
4. Persistencia en PostgreSQL (`upsert_oportunidades`).
5. Exportación del CSV de oportunidades nuevas
   (`reports/oportunidades_nuevas.csv`).

`scripts/etl_oportunidades.py` es el entry point de CLI (wrapper fino que
también puede ejecutar el scraping previo).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from real_estate.persistencia.repositorio import (
    COLUMNAS_OPORTUNIDADES,
    ids_procesados,
    upsert_oportunidades,
)
from real_estate.serving.clasificacion import (
    clasificar_oportunidades,
    clasificar_por_diferencia,
)
from real_estate.serving.evaluar import MODELO_DEFAULT, evaluar_dataframe
from real_estate.serving.persistencia import cargar_bundle

INPUT_DEFAULT = "data/raw/propiedades_nuevas.csv"
OUTPUT_DEFAULT = "reports/oportunidades_nuevas.csv"


def _modelo_version(directorio_modelo: str | Path) -> str:
    """Versión del modelo derivada de la metadata del bundle (tipo + fecha)."""

    metadata = cargar_bundle(directorio_modelo).metadata
    tipo = str(metadata.get("tipo_modelo", "modelo"))
    fecha = str(metadata.get("fecha_exportacion", ""))[:10]
    return f"{tipo}-{fecha}" if fecha else tipo


def _solo_nuevas(df: pd.DataFrame, engine: Engine) -> pd.DataFrame:
    """Filtra las publicaciones cuyo id todavía no está persistido en la base."""

    ya_procesados = ids_procesados(engine)
    if not ya_procesados:
        return df
    return df[~df["id"].astype(str).isin(ya_procesados)]


def ejecutar_etl(
    engine: Engine,
    input_file: str | Path = INPUT_DEFAULT,
    output_file: str | Path = OUTPUT_DEFAULT,
    directorio_modelo: str | Path = MODELO_DEFAULT,
) -> Path:
    """
    Procesa las publicaciones nuevas del CSV y las persiste como oportunidades.

    El dedup es contra PostgreSQL (el store persistente entre corridas del
    pipeline): solo se curan, predicen y persisten los ids nuevos. Devuelve la
    ruta del CSV de oportunidades nuevas exportado.
    """

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

    print(f"Cargando publicaciones scrapeadas: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)

    print(f"\nPublicaciones en el CSV: {len(df):,}")

    df = _solo_nuevas(df, engine)

    print(f"Publicaciones nuevas (no persistidas): {len(df):,}")

    if df.empty:
        print("\nNo hay publicaciones nuevas para procesar.")
        return output_path

    df = evaluar_dataframe(df, directorio_modelo)

    # `ratio_precio` (ranking de ofertas) + clasificación por propiedad; la
    # segunda sobreescribe `clasificacion` con el resultado por propiedad.
    df = clasificar_oportunidades(df)
    df = clasificar_por_diferencia(df)

    print("\n" + "=" * 70)
    print("CLASIFICACIÓN")
    print("=" * 70)
    print(f"\n{df['clasificacion'].value_counts().to_string()}")

    version = _modelo_version(directorio_modelo)

    persistidas = upsert_oportunidades(engine, df, modelo_version=version)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df[COLUMNAS_OPORTUNIDADES].to_csv(output_path, index=False)

    print(f"\nOportunidades persistidas en PostgreSQL: {persistidas:,}")
    print(f"CSV de oportunidades nuevas guardado en:\n{output_path}")

    return output_path
