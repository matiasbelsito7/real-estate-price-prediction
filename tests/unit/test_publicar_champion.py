"""Tests unitarios de `scripts/publicar_champion.py` (publicación del champion, fase 13)."""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

# El script vive en scripts/ y no forma parte del paquete instalable: se carga
# por ruta para poder testear sus funciones sin ejecutar el CLI.
_RAIZ = Path(__file__).resolve().parents[2]
_RUTA_SCRIPT = _RAIZ / "scripts" / "publicar_champion.py"
_SPEC = importlib.util.spec_from_file_location("publicar_champion", _RUTA_SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
publicar: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(publicar)

URL_ASSET = (
    "https://github.com/matiasbelsito7/real-estate-price-prediction"
    "/releases/download/v1-bundle-modelo_precio_propiedades/modelo_precio_propiedades.zip"
)


def _metadata(fecha: str = "2026-08-16T10:00:00+00:00") -> dict[str, object]:
    return {
        "tipo_modelo": "xgboost",
        "fecha_exportacion": fecha,
        "metricas_xgboost_test": {"rmse_log": 0.30, "rmse_usd": 87000.0, "r2": 0.78},
    }


def _escribir_bundle(tmp_path: object, fecha: str = "2026-08-16T10:00:00+00:00") -> Path:
    bundle = Path(tmp_path) / "bundle"  # type: ignore[arg-type]
    bundle.mkdir()
    (bundle / "modelo_xgboost.json").write_text("{}", encoding="utf-8")
    (bundle / "preprocesamiento.json").write_text("{}", encoding="utf-8")
    (bundle / "features.json").write_text("[]", encoding="utf-8")
    (bundle / "metadata.json").write_text(json.dumps(_metadata(fecha)), encoding="utf-8")
    (bundle / "resumen_bundle.json").write_text("{}", encoding="utf-8")
    return bundle


class TestRepoRemoto:
    def test_https(self) -> None:
        assert (
            publicar._repo_remoto(
                "https://github.com/matiasbelsito7/real-estate-price-prediction.git"
            )
            == "matiasbelsito7/real-estate-price-prediction"
        )

    def test_ssh(self) -> None:
        assert (
            publicar._repo_remoto("git@github.com:matiasbelsito7/real-estate-price-prediction.git")
            == "matiasbelsito7/real-estate-price-prediction"
        )

    def test_url_sin_git(self) -> None:
        assert publicar._repo_remoto("https://github.com/usuario/repo") == "usuario/repo"

    def test_ya_es_user_repo(self) -> None:
        assert publicar._repo_remoto("usuario/repo") == "usuario/repo"

    def test_url_de_otro_host_raise(self) -> None:
        with pytest.raises(ValueError):
            publicar._repo_remoto("https://gitlab.com/usuario/repo")


class TestModeloVersion:
    def test_formato_tipo_fecha(self) -> None:
        assert publicar.modelo_version_desde_metadata(_metadata()) == "xgboost-2026-08-16"


class TestCrearZip:
    def test_incluye_archivos_en_la_raiz(self, tmp_path: object) -> None:
        bundle = _escribir_bundle(tmp_path)
        zip_path = tmp_path / "bundle.zip"  # type: ignore[operator]

        publicar.crear_zip(bundle, zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            nombres = set(zf.namelist())
        assert nombres == {
            "modelo_xgboost.json",
            "preprocesamiento.json",
            "features.json",
            "metadata.json",
            "resumen_bundle.json",
        }

    def test_falta_archivo_obligatorio_raise(self, tmp_path: object) -> None:
        bundle = tmp_path / "bundle"  # type: ignore[operator]
        bundle.mkdir()
        (bundle / "metadata.json").write_text("{}", encoding="utf-8")

        with pytest.raises(FileNotFoundError):
            publicar.crear_zip(bundle, tmp_path / "bundle.zip")  # type: ignore[operator]


class TestFingerprint:
    def test_deterministico_y_con_url_de_asset(self) -> None:
        a = publicar.fingerprint_desde_metadata(_metadata(), URL_ASSET)
        b = publicar.fingerprint_desde_metadata(_metadata(), URL_ASSET)

        assert a == b
        assert a["modelo_version"] == "xgboost-2026-08-16"
        assert a["tipo_modelo"] == "xgboost"
        assert a["asset_url"] == URL_ASSET

    def test_cambia_con_la_version(self) -> None:
        a = publicar.fingerprint_desde_metadata(_metadata("2026-08-16T10:00:00+00:00"), "url")
        b = publicar.fingerprint_desde_metadata(_metadata("2026-08-20T10:00:00+00:00"), "url")

        assert a != b
        assert a["modelo_version"] == "xgboost-2026-08-16"
        assert b["modelo_version"] == "xgboost-2026-08-20"


class TestPublicarChampion:
    def test_no_upload_genera_fingerprint(self, tmp_path: object) -> None:
        bundle = _escribir_bundle(tmp_path)
        fp = tmp_path / "champion_actual.json"  # type: ignore[operator]

        ok = publicar.publicar_champion(
            bundle_dir=bundle,
            fingerprint_path=fp,
            zip_salida=tmp_path / "bundle.zip",  # type: ignore[operator]
            repo="matiasbelsito7/real-estate-price-prediction",
            no_upload=True,
        )

        assert ok
        guardado = json.loads(fp.read_text(encoding="utf-8"))
        assert guardado["modelo_version"] == "xgboost-2026-08-16"
        assert guardado["asset_url"] == URL_ASSET

    def test_sin_cambios_no_república(self, tmp_path: object) -> None:
        bundle = _escribir_bundle(tmp_path)
        fp = tmp_path / "champion_actual.json"  # type: ignore[operator]

        publicar.publicar_champion(
            bundle_dir=bundle, fingerprint_path=fp, repo="u/r", no_upload=True
        )
        ok = publicar.publicar_champion(
            bundle_dir=bundle, fingerprint_path=fp, repo="u/r", no_upload=True
        )

        assert ok is False

    def test_upload_ejecuta_gh_release_upload(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bundle = _escribir_bundle(tmp_path)
        zip_path = tmp_path / "bundle.zip"  # type: ignore[operator]
        fp = tmp_path / "champion_actual.json"  # type: ignore[operator]
        comandos: list[list[str]] = []
        monkeypatch.setattr(publicar, "_ejecutar", lambda c: comandos.append(c))

        ok = publicar.publicar_champion(
            bundle_dir=bundle,
            fingerprint_path=fp,
            zip_salida=zip_path,
            repo="matiasbelsito7/real-estate-price-prediction",
        )

        assert ok
        assert comandos == [
            [
                "gh",
                "release",
                "upload",
                "v1-bundle-modelo_precio_propiedades",
                str(zip_path),
                "--clobber",
                "--repo",
                "matiasbelsito7/real-estate-price-prediction",
            ]
        ]

    def test_commit_y_push_usan_git(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bundle = _escribir_bundle(tmp_path)
        fp = tmp_path / "champion_actual.json"  # type: ignore[operator]
        comandos: list[list[str]] = []
        monkeypatch.setattr(publicar, "_ejecutar", lambda c: comandos.append(c))

        publicar.publicar_champion(
            bundle_dir=bundle,
            fingerprint_path=fp,
            zip_salida=tmp_path / "bundle.zip",  # type: ignore[operator]
            repo="u/r",
            commit=True,
            push=True,
        )

        assert comandos[0][0] == "gh"
        assert comandos[1] == ["git", "add", str(fp)]
        assert comandos[2] == [
            "git",
            "commit",
            "-m",
            "chore(cd): publicar champion xgboost-2026-08-16",
        ]
        assert comandos[3] == ["git", "push"]

    def test_no_upload_ignora_commit_y_push(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bundle = _escribir_bundle(tmp_path)
        fp = tmp_path / "champion_actual.json"  # type: ignore[operator]
        comandos: list[list[str]] = []
        monkeypatch.setattr(publicar, "_ejecutar", lambda c: comandos.append(c))

        publicar.publicar_champion(
            bundle_dir=bundle,
            fingerprint_path=fp,
            repo="u/r",
            no_upload=True,
            commit=True,
            push=True,
        )

        assert comandos == []
