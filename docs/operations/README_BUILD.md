# Cyrex Build Guide - GPU Detection & CPU Fallback

Detection is implemented by **[deepiri-gpu-utils](https://github.com/Team-Deepiri/deepiri-gpu-utils)** (`deepiri-gpu build-args`).

## Automatic GPU Detection

- **GPU detected with ≥4GB VRAM (NVIDIA)**: typically `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime`
- **No GPU or insufficient VRAM**: `python:3.11-slim` (CPU)

## Quick Start

### From deepiri-platform (Recommended)

```bash
cd deepiri-platform
docker compose -f docker-compose.dev.yml build cyrex
```

### From This Repo (diri-cyrex)

**Linux/Mac:**
```bash
cd diri-cyrex
docker build -t deepiri-dev-cyrex:latest .
```

**Windows (PowerShell):**
```powershell
cd diri-cyrex
docker build -t deepiri-dev-cyrex:latest .
```

### Docker Compose (from deepiri-platform)

```bash
cd deepiri-platform
# Optional: override build args
export CYREX_BASE_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
export CYREX_DEVICE_TYPE=gpu
docker compose -f docker-compose.dev.yml build cyrex
```

Defaults in `docker-compose.dev.yml` keep a CUDA 12.8 base image when env vars are unset.

## Force CPU Build

```bash
# Via environment variables (from deepiri-platform)
export CYREX_BASE_IMAGE=python:3.11-slim
export CYREX_DEVICE_TYPE=cpu
docker compose -f docker-compose.dev.yml build cyrex
```

Or with `docker build` (from diri-cyrex):
```bash
cd diri-cyrex
docker build --build-arg BASE_IMAGE=python:3.11-slim --build-arg DEVICE_TYPE=cpu -t deepiri-dev-cyrex:latest .
```

## GPU Requirements

Policy lives in **deepiri-gpu-utils** (`detect`, `build_args`). Typical rules:

- **Minimum GPU memory**: 4GB VRAM for the CUDA base image
- **NVIDIA**: use `nvidia-smi` when available

## Build Optimizations

1. **Prebuilt Images**: PyTorch + CUDA already included (no large compile-from-source in base)
2. **Split Installation**: Packages installed in chunks to avoid timeouts
3. **Optional Packages**: GPU-specific packages (deepspeed, bitsandbytes) only install if CUDA available
4. **Network Timeouts**: Retries as configured in Dockerfile
5. **Build Cache**: Better layer caching for faster rebuilds

## Troubleshooting

### Build Still Freezing?

1. Use CPU build (see above)
2. Build during off-peak hours
3. Use wired connection instead of WiFi

### GPU Not Detected?

- Install **deepiri-gpu-utils** and check `deepiri-gpu doctor`
- Check `nvidia-smi` on the host
- Run `deepiri-gpu build-args` and read warnings

### Want to Override Detection?

Set `CYREX_BASE_IMAGE` / `CYREX_DEVICE_TYPE` for compose, or pass `--build-arg` to `docker build`.

## Baseline snapshots

Before/after integration, run from `deepiri-platform`:

```bash
bash diri-cyrex/scripts/record_gpu_integration_baseline.sh
```

Compare outputs under `/tmp/cyrex-helox-gpu-baseline-*`.
