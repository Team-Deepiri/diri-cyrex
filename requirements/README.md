# Requirements (deprecated)

Cyrex uses **Poetry** (`pyproject.toml` + `poetry.lock`) for all Python dependencies.

- **Docker:** `Dockerfile` / `Dockerfile.cpu` run `poetry install` (see `POETRY_EXTRAS`).
- **Local dev:** `poetry install --extras gpu` (or `mps`, `rocm`, `cpu`, `full`).
- **Host GPU detection:** `scripts/utils/detect_gpu.sh` (uses `deepiri-gpu-utils` via Poetry or a lightweight pip fallback).

Do not add new `requirements*.txt` files here.
