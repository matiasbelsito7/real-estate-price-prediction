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

from real_estate.persistencia.bundle import cargar_bundle
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

INPUT_DEFAULT = "data/raw/propiedades_nuevas.csv"
OUTPUT_DEFAULT = "reports/oportunidades_nuevas.csv"


def _modelo_version(directorio_modelo: str | Path) -> str:
    """Versión del modelo derivada de la metadata del bundle (tipo + fecha)."""

    metadata = cargar_bundle(directorio_modelo).metadata
    tipo = str(metadata.get("tipo_modelo", "modelo"))
    fecha = str(metadata.get("fecha_exportacion", ""))[:10]
    return f"{tipo}-{fecha}" if fecha else tipo


def _solo_nuevas(df: pd.DataFrame, engine: Engine) -> pd.DataFrame:
    """Filtra las publicaciones cuyo id ya existe en la base de datos."""

    ids_existentes = ids_procesados(engine)
    if not ids_existentes:
        return df

    return df[~df["id"].isin(ids_existentes)].copy()


def ejecutar_etl(
    engine: Engine,
    input_file: str | Path = INPUT_DEFAULT,
    output_file: str | Path = OUTPUT_DEFAULT,
    directorio_modelo: str | Path = MODELO_DEFAULT,
) -> Path | None:
    """Ejecuta el ciclo completo del ETL de nuevas oportunidades."""

    input_file = Path(input_file)
    output_file = Path(output_file)

    if not input_file.exists():
        print(f"Saltando ETL: No existe el archivo de entrada '{input_file}'")
        return None

    df_raw = pd.read_csv(input_file)
    df_nuevas = _solo_nuevas(df_raw, engine)

    if df_nuevas.empty:
        print("No hay publicaciones nuevas para procesar.")
        return None

    print(f"Procesando {len(df_nuevas)} publicaciones nuevas...")

    # 1. Curación y Predicción
    df_pred = evaluar_dataframe(df_nuevas, directorio_modelo=directorio_modelo)

    # 2. Clasificación (Oportunidades)
    df_diff = clasificar_oportunidades(df_pred)
    df_final = clasificar_por_diferencia(df_diff)

    # 3. Metadatos del proceso
    version = _modelo_version(directorio_modelo)
    df_final["modelo_version"] = version

    # 4. Persistencia
    df_db = df_final[COLUMNAS_OPORTUNIDADES]
    upsert_oportunidades(engine, df_db, modelo_version=version)

    # 5. Exportación
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_final[COLUMNAS_OPORTUNIDADES].to_csv(output_file, index=False)

    print(f"ETL finalizado. Oportunidades guardadas en DB y en '{output_file}'")

    return output_file
