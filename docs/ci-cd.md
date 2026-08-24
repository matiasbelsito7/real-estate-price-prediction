# CI/CD — Workflows y Calidad de Código

> Documenta los pipelines de integración continua, delivery continua y
> la cadena de calidad del proyecto.

---

## Flujo de calidad

```text
Developer → Code → pre-commit → Ruff + Mypy → Git commit
    ↓
GitHub → GitHub Actions → CI (lint + typecheck + tests)
    ↓
Docker build → Push a GHCR
```

**Gate local:** `make check` = `ruff format --check` + `ruff check` + `mypy` + `pytest`

---

## 1. Pre-commit

**Archivo:** `.pre-commit-config.yaml`

### Hooks

| Hook | Qué hace |
|---|---|
| `ruff-check` | Lint con `--fix` (autocorrección) |
| `ruff-format` | Formateo de código |
| `mypy` | Type checking (con pandas-stubs, types-requests, pytest) |
| `check-yaml` | Valida YAML |
| `check-toml` | Valida TOML |
| `end-of-file-fixer` | Asegura newline final |
| `trailing-whitespace` | Elimina whitespace trailing |
| `check-added-large-files` | Bloquea archivos > 1MB |

---

## 2. CI — GitHub Actions

### ci.yml

**Trigger:** push/PR a `main`

| Job | Python | Qué hace |
|---|---|---|
| `quality` | 3.12 | Ruff check + format, Mypy |
| `tests` | 3.11, 3.12 (matriz) | Pytest con cobertura |

### dvc.yml

**Trigger:** push/PR a `main`, `workflow_dispatch`

- Lista etapas definidas (`dvc stage list`)
- Verifica estado contra el lock (`dvc status`)
- Intenta restaurar datos (`dvc pull`, best-effort)

### etl_oportunidades.yml

**Trigger:** cron `0 11 */4 * *` (cada 4 días, 08:00 ART) + `workflow_dispatch`

- Levanta PostgreSQL efímero (servicio `postgres`)
- Restaura bundle del champion desde secret `MODELO_BUNDLE_URL`
- Ejecuta `python scripts/etl_oportunidades.py --todos-los-barrios`
- Postingestion: sube opportidades al dashboard de Grafana (futuro)

### cd_champion.yml

**Trigger:** push de `models/champion_actual.json` + `workflow_dispatch`

1. Restaura bundle desde `asset_url` del fingerprint
2. Smoke test (carga, predicción, verificación de `tipo_modelo`)
3. Build imagen Docker self-contained
4. Push a GHCR con tags `latest` y versión del modelo

---

## 3. Herramientas de calidad

| Herramienta | Configuración | Uso |
|---|---|---|
| Ruff | `line-length=100`, target `py311`, reglas `E W F I B UP N SIM` | Lint + format |
| Mypy | `strict=true`, `python_version=3.12` | Type checking |
| Pytest | `testpaths=["tests"]`, `pythonpath=["src"]` | Testing |
| Coverage | `branch=true`, `source=["src/real_estate"]` | Cobertura |

---

## 4. Comandos de calidad

```bash
make format      # ruff format src tests
make lint        # ruff check src tests
make typecheck   # mypy src tests
make test        # pytest
make coverage    # pytest --cov
make check       # todos los anteriores
```
