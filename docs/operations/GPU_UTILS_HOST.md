# deepiri-gpu-utils on the host (Cyrex builds)

Cyrex [`scripts/utils/detect_gpu.sh`](../scripts/utils/detect_gpu.sh) and [`detect_gpu.ps1`](../scripts/utils/detect_gpu.ps1) call the **`deepiri-gpu`** CLI from the [`deepiri-gpu-utils`](https://github.com/Team-Deepiri/deepiri-gpu-utils) package. Install it in the environment where you run Docker builds (your laptop, CI worker, etc.), not inside the Cyrex container image.

## Monorepo layout

If `Deepiri/` contains both `deepiri-platform/` and `deepiri-gpu-utils/`:

```bash
pip install -e "../deepiri-gpu-utils"
```

Run that from `deepiri-platform/diri-cyrex` or any working directory; adjust the relative path if your layout differs.

## From GitHub (no local clone of gpu-utils)

```bash
pip install "git+https://github.com/Team-Deepiri/deepiri-gpu-utils.git"
```

When the package is published to PyPI, prefer:

```bash
pip install "deepiri-gpu-utils>=0.1.0"
```

## Helox

Helox declares `deepiri-gpu-utils` as a Poetry path dependency (see `diri-helox/pyproject.toml`). Use `poetry install` in `diri-helox/`.

## Compose / BASE_IMAGE

To align `docker compose` Cyrex builds with detection, export build args before building:

```bash
export CYREX_BASE_IMAGE=$(deepiri-gpu build-args --base-image-only)
# optional: eval or map DEVICE_TYPE from `deepiri-gpu build-args`
docker compose -f docker-compose.dev.yml build cyrex
```

`docker-compose.dev.yml` reads `CYREX_BASE_IMAGE` and `CYREX_DEVICE_TYPE` when set.

## Baseline / regression

Run [`../../scripts/record_gpu_integration_baseline.sh`](../../scripts/record_gpu_integration_baseline.sh) or, from `deepiri-platform/`, `bash diri-cyrex/scripts/record_gpu_integration_baseline.sh` before and after integration changes; compare outputs under `/tmp/cyrex-helox-gpu-baseline-*`.
