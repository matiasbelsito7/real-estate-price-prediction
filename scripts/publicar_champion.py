#!/usr/bin/env python3
"""
Publica el bundle del champion en GitHub Releases y deja un fingerprint que
dispara el CD (Fase 13).

Uso:
    python scripts/publicar_champion.py                        # sube el asset y actualiza el fingerprint
    python scripts/publicar_champion.py --no-upload            # solo regenera el fingerprint (dry run)
    python scripts/publicar_champion.py --no-upload --commit --push

Cómo funciona:
    1. Lee `models/modelo_precio_propiedades/metadata.json` y arma un
       fingerprint determinístico (`models/champion_actual.json`).
    2. Si el fingerprint no cambió, no hace nada: ese champion ya está
       publicado (evita commits vacíos y corridas de CD innecesarias).
    3. Comprime el bundle en un zip con los archivos en la raíz.
    4. Sube el zip al asset del release con `gh release upload --clobber`.
       El tag y el nombre del asset son estables entre versiones, así la URL
       del asset no cambia y puede ir fija en los workflows de CI/CD.
    5. Escribe el fingerprint y (con --commit/--push) lo commitea y pushea:
       ese push es el disparador del workflow de CD (fase 13).

El fingerprint es determinístico (no lleva marcas de tiempo): depende solo
del contenido del modelo (versión, fecha de exportación y métricas). Así,
re-publicar el mismo bundle no cambia el archivo y no re-dispara el CD. El
release `v1-bundle-modelo_precio_propiedades` se crea una única vez a mano
(`gh release create`); este script solo actualiza su asset.

--commit y --push solo tienen efecto cuando se sube el asset (no con
--no-upload), para no committear un fingerprint que apunta a un asset que
todavía no existe.
"""

import argparse
import json
import logging
import subprocess
import sys
import zipfile
from pathlib import Path

# Permite importar el paquete `real_estate` (layout src/) sin instalarlo.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from real_estate.persistencia.bundle import (  # noqa: E402
    NOMBRE_FEATURES,
    NOMBRE_METADATA,
    NOMBRE_MODELO,
    NOMBRE_PREPROCESAMIENTO,
)
from real_estate.utils.logging import configurar_logging  # noqa: E402

logger = logging.getLogger(__name__)

BUNDLE_DIR_DEFAULT = Path("models/modelo_precio_propiedades")
FINGERPRINT_DEFAULT = Path("models/champion_actual.json")
ZIP_DEFAULT = Path("tmp/modelo_precio_propiedades.zip")
TAG_DEFAULT = "v1-bundle-modelo_precio_propiedades"
ASSET_DEFAULT = "modelo_precio_propiedades.zip"

ARCHIVOS_OBLIGATORIOS: tuple[str, ...] = (
    NOMBRE_MODELO,
    NOMBRE_PREPROCESAMIENTO,
    NOMBRE_FEATURES,
    NOMBRE_METADATA,
)
ARCHIVOS_OPCIONALES: tuple[str, ...] = ("resumen_bundle.json",)


def _repo_remoto(remote_url: str | None = None) -> str:
    """Deriva `usuario/repo` desde `git remote get-url origin` (o una URL dada)."""

    if remote_url is None:
        resultado = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        )
        remote_url = resultado.stdout

    limpia = remote_url.strip().removesuffix(".git").rstrip("/")
    for separador in ("github.com/", "github.com:"):
        if separador in limpia:
            return limpia.split(separador, 1)[1]
    if "/" in limpia and not limpia.startswith(("http://", "https://", "git://", "git@")):
        return limpia
    raise ValueError(f"No se pudo derivar usuario/repo de: {remote_url!r}")


def modelo_version_desde_metadata(metadata: dict[str, object]) -> str:
    """Versión legible del modelo: `{tipo_modelo}-{fecha_exportacion[:10]}`."""

    tipo_modelo = str(metadata["tipo_modelo"])
    fecha_exportacion = str(metadata["fecha_exportacion"])
    return f"{tipo_modelo}-{fecha_exportacion[:10]}"


def crear_zip(bundle_dir: str | Path, zip_salida: str | Path) -> Path:
    """Comprime el bundle dejando los archivos en la raíz del zip."""

    bundle_dir = Path(bundle_dir)
    zip_salida = Path(zip_salida)

    faltantes = [n for n in ARCHIVOS_OBLIGATORIOS if not (bundle_dir / n).is_file()]
    if faltantes:
        raise FileNotFoundError(
            f"El bundle en {bundle_dir} no tiene los archivos obligatorios: {', '.join(faltantes)}"
        )

    zip_salida.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_salida, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for nombre in ARCHIVOS_OBLIGATORIOS + ARCHIVOS_OPCIONALES:
            ruta = bundle_dir / nombre
            if ruta.is_file():
                zf.write(ruta, arcname=nombre)
    return zip_salida


def fingerprint_desde_metadata(metadata: dict[str, object], asset_url: str) -> dict[str, object]:
    """Fingerprint determinístico del champion (sin marcas de tiempo).

    Depende solo del contenido del modelo: si el modelo no cambia, el
    fingerprint no cambia y no se dispara el CD.
    """

    return {
        "modelo_version": modelo_version_desde_metadata(metadata),
        "tipo_modelo": str(metadata["tipo_modelo"]),
        "fecha_exportacion": str(metadata["fecha_exportacion"]),
        "metricas_xgboost_test": metadata["metricas_xgboost_test"],
        "asset_url": asset_url,
    }


def _ejecutar(comando: list[str]) -> None:
    """Ejecuta un comando externo (gh/git) mostrando la línea en stdout."""

    logger.info("+ %s", " ".join(comando))
    subprocess.run(comando, check=True)


def _commit_y_push(fingerprint_path: Path, commit: bool, push: bool, modelo_version: str) -> None:
    if not (commit or push):
        return
    _ejecutar(["git", "add", str(fingerprint_path)])
    if commit:
        _ejecutar(["git", "commit", "-m", f"chore(cd): publicar champion {modelo_version}"])
    if push:
        _ejecutar(["git", "push"])


def publicar_champion(
    bundle_dir: str | Path = BUNDLE_DIR_DEFAULT,
    fingerprint_path: str | Path = FINGERPRINT_DEFAULT,
    zip_salida: str | Path = ZIP_DEFAULT,
    repo: str | None = None,
    tag: str = TAG_DEFAULT,
    asset: str = ASSET_DEFAULT,
    no_upload: bool = False,
    commit: bool = False,
    push: bool = False,
) -> bool:
    """Publica el champion y actualiza su fingerprint.

    Devuelve True si el champion cambió (y se publicó) y False si ya estaba
    publicado (el fingerprint no cambió → no hay nada que hacer).
    """

    bundle_dir = Path(bundle_dir)
    fingerprint_path = Path(fingerprint_path)
    zip_salida = Path(zip_salida)

    metadata_ruta = bundle_dir / NOMBRE_METADATA
    if not metadata_ruta.is_file():
        raise FileNotFoundError(
            f"No se encontró {metadata_ruta}; exportá el bundle primero (make export-model)."
        )
    metadata: dict[str, object] = json.loads(metadata_ruta.read_text(encoding="utf-8"))

    repo_efectivo = repo or _repo_remoto()
    asset_url = f"https://github.com/{repo_efectivo}/releases/download/{tag}/{asset}"
    fingerprint = fingerprint_desde_metadata(metadata, asset_url)

    if fingerprint_path.is_file():
        existente: dict[str, object] = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if existente == fingerprint:
            logger.info(
                f"Sin cambios: el champion {fingerprint['modelo_version']} ya está publicado."
            )
            return False

    modelo_version = str(fingerprint["modelo_version"])
    logger.info(f"Nuevo champion detectado: {modelo_version}")

    crear_zip(bundle_dir, zip_salida)
    logger.info(f"Bundle comprimido: {zip_salida}")

    if no_upload:
        logger.info("Modo --no-upload: no se sube el asset a GitHub Releases.")
    else:
        _ejecutar(
            ["gh", "release", "upload", tag, str(zip_salida), "--clobber", "--repo", repo_efectivo]
        )

    fingerprint_path.write_text(
        json.dumps(fingerprint, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(f"Fingerprint actualizado: {fingerprint_path}")

    if not no_upload:
        _commit_y_push(fingerprint_path, commit, push, modelo_version)

    return True


def main() -> None:
    configurar_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Publica el bundle del champion en GitHub Releases y deja el "
            "fingerprint models/champion_actual.json que dispara el CD (fase 13)"
        )
    )
    parser.add_argument(
        "--bundle-dir",
        default=str(BUNDLE_DIR_DEFAULT),
        help=f"Directorio del bundle (default: {BUNDLE_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--fingerprint",
        default=str(FINGERPRINT_DEFAULT),
        help=f"Ruta del fingerprint a escribir (default: {FINGERPRINT_DEFAULT})",
    )
    parser.add_argument(
        "--zip",
        dest="zip_salida",
        default=str(ZIP_DEFAULT),
        help=f"Zip temporal del bundle (default: {ZIP_DEFAULT})",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="usuario/repo de GitHub; por defecto se deriva de `git remote get-url origin`",
    )
    parser.add_argument(
        "--tag",
        default=TAG_DEFAULT,
        help=f"Tag del release cuyo asset se actualiza (default: {TAG_DEFAULT})",
    )
    parser.add_argument(
        "--asset",
        default=ASSET_DEFAULT,
        help=f"Nombre del asset dentro del release (default: {ASSET_DEFAULT})",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Solo regenera el fingerprint sin subir el asset (dry run)",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Commitea el fingerprint (ignorado con --no-upload)",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Pushea el fingerprint (ignorado con --no-upload)",
    )
    args = parser.parse_args()

    publicar_champion(
        bundle_dir=args.bundle_dir,
        fingerprint_path=args.fingerprint,
        zip_salida=args.zip_salida,
        repo=args.repo,
        tag=args.tag,
        asset=args.asset,
        no_upload=args.no_upload,
        commit=args.commit,
        push=args.push,
    )


if __name__ == "__main__":
    main()
