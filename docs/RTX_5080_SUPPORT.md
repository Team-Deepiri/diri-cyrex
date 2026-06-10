# RTX 5080/5090 (sm_120) Support

## Overview

CUDA 12.8 support is now **automatic** in all builds. The Dockerfile automatically upgrades PyTorch to CUDA 12.8, which supports all GPUs from sm_50 through sm_120 (including RTX 5080/5090).

## What Happens Automatically

Every build now:
1. Uses the CUDA 12.6 base image (for compatibility)
2. Automatically uninstalls the default PyTorch (CUDA 12.6)
3. Installs PyTorch nightly build with CUDA 12.8 support from: `https://download.pytorch.org/whl/nightly/cu128`
4. This PyTorch version includes kernels for sm_50 through sm_120 (all modern GPUs including RTX 5080/5090)

## Building

Just build normally - CUDA 12.8 support is automatic:

**Using build scripts:**
```bash
cd diri-cyrex
./build.sh
```

**Using Docker Compose:**
```bash
docker-compose -f docker-compose.dev.yml up --build cyrex
```

**Manual Docker build:**
```bash
docker build \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime \
  -f diri-cyrex/Dockerfile \
  -t deepiri-dev-cyrex:latest \
  .
```

No build args needed - it's automatic!

## Verification

After building, verify GPU support:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

# Test GPU operation
x = torch.tensor([1.0], device='cuda')
y = x * 2.0
print(f"GPU test successful: {y.cpu().item()}")
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 5080 Laptop GPU
Compute Capability: (12, 0)
GPU test successful: 2.0
```

## Notes

- PyTorch nightly builds are used for CUDA 12.8 support (stable releases may not have it yet)
- The base image still uses CUDA 12.6, but PyTorch is upgraded to CUDA 12.8 during build
- This only affects the PyTorch installation, not the CUDA runtime libraries
- Ensure your NVIDIA drivers support CUDA 12.8 (Driver version 570+)

## Troubleshooting

If GPU still doesn't work after enabling CUDA_128:

1. **Check NVIDIA drivers:**
   ```bash
   nvidia-smi
   ```
   Should show driver version 570 or higher.

2. **Verify NVIDIA Container Toolkit:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu20.04 nvidia-smi
   ```

3. **Check Docker GPU access:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu20.04 nvidia-smi
   ```

4. **Rebuild the image:**
   ```bash
   docker-compose -f docker-compose.dev.yml build --no-cache cyrex
   ```

