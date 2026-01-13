# Apple Silicon MPS Setup Guide

This guide covers how to set up Metal Performance Shaders (MPS) support for Apple Silicon Macs in the Deepiri development environment.

## Prerequisites

### System Requirements

- macOS 12.3 or later (Monterey or newer)
- Apple Silicon chip (M1, M1 Pro, M1 Max, M2, M2 Pro, M2 Max, M3, M3 Pro, M3 Max, M4, or newer)
- Python 3.8 or later
- pip package manager

### Verify Apple Silicon

Check if you have an Apple Silicon Mac:

```bash
uname -m
```

Expected output: `arm64`

Or check the chip type:

```bash
sysctl -n machdep.cpu.brand_string
```

Expected output should contain "Apple" (e.g., "Apple M1", "Apple M2 Pro", etc.)

## Installation Methods

### Method 1: PyTorch with MPS Support (Recommended)

PyTorch 1.12+ includes built-in MPS support. Install PyTorch with MPS:

```bash
pip install torch torchvision torchaudio
```

For a specific PyTorch version:

```bash
pip install torch==2.9.1 torchvision torchaudio
```

### Method 2: Using PyTorch Nightly (Latest Features)

For the latest MPS features and improvements:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

Note: Nightly builds may be less stable but include the latest MPS optimizations.

### Method 3: Using Conda

If using Conda:

```bash
conda install pytorch torchvision torchaudio -c pytorch
```

## Verification

### Test MPS Availability

Create a test script to verify MPS is working:

```python
import torch

# Check if MPS is available
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("MPS is available")
    
    # Test a simple operation
    try:
        x = torch.tensor([1.0, 2.0, 3.0], device='mps')
        y = x * 2
        print(f"MPS test successful: {y}")
        print(f"Device: {y.device}")
    except Exception as e:
        print(f"MPS test failed: {e}")
else:
    print("MPS is not available")
    print("PyTorch version:", torch.__version__)
    print("macOS version check: Run 'sw_vers' to verify macOS 12.3+")
```

Run the test:

```bash
python test_mps.py
```

Expected output:
```
MPS is available
MPS test successful: tensor([2., 4., 6.], device='mps:0')
Device: mps:0
```

### Check PyTorch Version

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

MPS support requires PyTorch 1.12 or later. For best results, use PyTorch 2.0+.

## Docker Setup for Apple Silicon

### Important: MPS Does NOT Work in Docker

**Critical Limitation**: Metal Performance Shaders (MPS) is a macOS-specific API and does NOT work inside Docker containers on macOS.

- Docker on macOS runs Linux containers in a virtualized environment
- Metal (the underlying GPU API) is macOS-only and cannot be accessed from Linux containers
- Docker containers on macOS can only use CPU, not GPU

### Docker Desktop Configuration

1. Install Docker Desktop for Apple Silicon:
   - Download from: https://www.docker.com/products/docker-desktop/
   - Or install via Homebrew: `brew install --cask docker`

2. Verify Docker is using arm64 architecture:
   ```bash
   docker version --format '{{.Server.Arch}}'
   ```
   Expected output: `arm64`

3. Ensure Docker Desktop is set to use Apple Silicon:
   - Open Docker Desktop
   - Go to Settings > General
   - Ensure "Use Rosetta for x86/amd64 emulation" is unchecked (unless needed for x86 containers)

### Recommended: Hybrid Approach

For MacBook development, use a hybrid approach:

1. **Run GPU-accelerated services natively** (outside Docker):
   - Ollama (uses Metal GPU automatically)
   - PyTorch applications (use MPS)
   - Any service that needs GPU acceleration

2. **Run non-GPU services in Docker**:
   - PostgreSQL
   - Redis
   - Other CPU-only services

### Building Docker Images for Apple Silicon

When building Docker images, specify the platform:

```bash
docker build --platform linux/arm64 -t deepiri-cyrex:arm64 .
```

Or use docker-compose with platform specification:

```yaml
services:
  cyrex:
    platform: linux/arm64
    build:
      context: .
      dockerfile: ./diri-cyrex/Dockerfile
```

**Note**: These containers will run on CPU only, not GPU.

### Connecting Docker Services to Native GPU Services

If you run Ollama natively (uses Metal GPU) and need Docker services to connect:

```yaml
services:
  cyrex:
    environment:
      OLLAMA_BASE_URL: http://host.docker.internal:11434  # Connect to native Ollama
```

### PyTorch in Docker for Apple Silicon

For Docker containers, MPS is not available. Options:

1. **Use CPU-optimized PyTorch builds for arm64** (slower, but works in Docker)
2. **Run PyTorch natively on macOS** (outside Docker) for MPS support (recommended)
3. **Use Docker only for services that don't need GPU acceleration**

For native macOS development with MPS:

```bash
# Install dependencies in a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio
```

## Development Environment Setup

### Virtual Environment Setup

1. Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install PyTorch with MPS:

```bash
pip install torch torchvision torchaudio
```

3. Install project dependencies:

```bash
pip install -r requirements.txt
```

### Environment Variables

No special environment variables are needed for MPS. PyTorch automatically detects and uses MPS when available.

However, you can explicitly set device preference:

```python
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Optional: enable fallback
```

## Integration with Deepiri Codebase

The codebase automatically detects and uses MPS through the device detection utility:

```python
from app.utils.device_detection import get_device

device = get_device()  # Returns 'mps' if available, 'cuda' if NVIDIA GPU, 'cpu' otherwise
```

This is used in:
- RAG pipeline embeddings
- Sentence transformers
- Cross-encoder rerankers
- All PyTorch-based models

## Troubleshooting

### MPS Not Available

**Issue**: `torch.backends.mps.is_available()` returns `False`

**Solutions**:
1. Verify macOS version: `sw_vers` (must be 12.3+)
2. Check PyTorch version: `python -c "import torch; print(torch.__version__)"` (must be 1.12+)
3. Reinstall PyTorch: `pip install --upgrade torch torchvision torchaudio`
4. Verify Apple Silicon: `uname -m` should return `arm64`

### MPS Operations Failing

**Issue**: MPS is available but operations fail with errors

**Solutions**:
1. Update macOS to the latest version
2. Update PyTorch to the latest stable version
3. Check for known issues: https://github.com/pytorch/pytorch/issues
4. Use CPU fallback temporarily: Set device to 'cpu' explicitly

### Docker Architecture Mismatch

**Issue**: Docker builds fail or run slowly on Apple Silicon

**Solutions**:
1. Ensure Docker Desktop is using arm64: `docker version --format '{{.Server.Arch}}'`
2. Use `--platform linux/arm64` in build commands
3. Use multi-platform builds when needed
4. Consider running services natively on macOS for MPS support

### Performance Issues

**Issue**: MPS is slower than expected

**Solutions**:
1. Ensure you're using PyTorch 2.0+ for better MPS performance
2. Check memory usage: MPS shares system RAM
3. Monitor Activity Monitor for memory pressure
4. Close other GPU-intensive applications
5. Use batch processing for better GPU utilization

## Testing MPS Performance

Create a benchmark script:

```python
import torch
import time

def benchmark_device(device_name):
    device = torch.device(device_name)
    size = (1000, 1000)
    
    # Warmup
    x = torch.randn(size, device=device)
    y = torch.randn(size, device=device)
    _ = torch.matmul(x, y)
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        x = torch.randn(size, device=device)
        y = torch.randn(size, device=device)
        z = torch.matmul(x, y)
    end = time.time()
    
    print(f"{device_name}: {(end - start) / 100 * 1000:.2f} ms per operation")

if torch.backends.mps.is_available():
    benchmark_device('mps')
benchmark_device('cpu')
```

## Additional Resources

- PyTorch MPS Documentation: https://pytorch.org/docs/stable/notes/mps.html
- Apple Metal Documentation: https://developer.apple.com/metal/
- PyTorch Installation Guide: https://pytorch.org/get-started/locally/

## Common Commands Reference

```bash
# Check macOS version
sw_vers

# Check architecture
uname -m

# Check chip type
sysctl -n machdep.cpu.brand_string

# Install PyTorch with MPS
pip install torch torchvision torchaudio

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"

# Test MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"

# Check Docker architecture
docker version --format '{{.Server.Arch}}'
```

