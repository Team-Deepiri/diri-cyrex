# CUDA Development Environment Setup

This guide covers how to set up CUDA support for NVIDIA GPUs in the Deepiri development environment.

## Prerequisites

### System Requirements

- NVIDIA GPU with CUDA Compute Capability 7.0 or higher
- NVIDIA GPU drivers installed (version 450.80.02 or later)
- Linux (Ubuntu 20.04+, Debian 11+, or similar) or Windows with WSL2
- Python 3.8 or later
- pip package manager

### Verify NVIDIA GPU

Check if you have an NVIDIA GPU:

**Linux:**
```bash
nvidia-smi
```

**Windows (WSL2):**
```bash
nvidia-smi
```

Expected output should show GPU information including:
- GPU name
- Driver version
- CUDA version
- Memory information

### Check CUDA Compute Capability

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

Minimum required: 7.0 (for PyTorch). Recommended: 8.0 or higher.

## Installation Methods

### Method 1: PyTorch with CUDA Support (Recommended)

Install PyTorch with CUDA support. Choose the CUDA version that matches your system:

**CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.4:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Method 2: Using Conda

If using Conda:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Method 3: Specific PyTorch Version

For a specific PyTorch version with CUDA:

```bash
pip install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Verification

### Test CUDA Availability

Create a test script to verify CUDA is working:

```python
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Get GPU information
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # Test a simple operation
    try:
        x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        y = x * 2
        print(f"\nCUDA test successful: {y}")
        print(f"Device: {y.device}")
    except Exception as e:
        print(f"CUDA test failed: {e}")
else:
    print("CUDA is not available")
    print("PyTorch version:", torch.__version__)
    print("\nTroubleshooting:")
    print("1. Verify NVIDIA drivers: nvidia-smi")
    print("2. Check CUDA installation: nvcc --version")
    print("3. Reinstall PyTorch with CUDA support")
```

Run the test:

```bash
python test_cuda.py
```

Expected output:
```
CUDA is available
CUDA version: 12.1
cuDNN version: 8902
Number of GPUs: 1

GPU 0:
  Name: NVIDIA GeForce RTX 4090
  Capability: (8, 9)
  Memory: 24.00 GB

CUDA test successful: tensor([2., 4., 6.], device='cuda:0')
Device: cuda:0
```

### Check PyTorch Version and CUDA

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

## Docker Setup for CUDA

### NVIDIA Container Toolkit Installation

For Docker containers to access GPU, install NVIDIA Container Toolkit:

**Ubuntu/Debian:**
```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use nvidia runtime
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default

# Restart Docker
sudo systemctl restart docker
```

**WSL2 (Windows):**
```bash
# Same commands as Ubuntu, but ensure:
# 1. WSL2 is updated: wsl --update
# 2. NVIDIA drivers are installed on Windows host
# 3. Docker Desktop is configured for WSL2 backend
```

### Verify Docker GPU Access

```bash
# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu20.04 nvidia-smi
```

Expected output should show GPU information.

### Docker Compose GPU Configuration

For docker-compose, use the `deploy` section:

```yaml
services:
  cyrex:
    build:
      context: .
      dockerfile: ./diri-cyrex/Dockerfile
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Or use runtime:

```yaml
services:
  cyrex:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

## Development Environment Setup

### Virtual Environment Setup

1. Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install PyTorch with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. Install project dependencies:

```bash
pip install -r requirements.txt
```

### Environment Variables

Optional environment variables for CUDA:

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Enable cuDNN benchmarking (faster, uses more memory)
export CUDNN_BENCHMARK=1

# Set CUDA device order
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Memory allocation settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Multi-GPU Setup

For systems with multiple GPUs:

```python
import torch

# Use all available GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    # PyTorch DataParallel
    model = torch.nn.DataParallel(model)
    
    # Or use DistributedDataParallel for better performance
    # torch.distributed.init_process_group(...)
    # model = torch.nn.parallel.DistributedDataParallel(model)
```

## Integration with Deepiri Codebase

The codebase automatically detects and uses CUDA through the device detection utility:

```python
from app.utils.device_detection import get_device

device = get_device()  # Returns 'cuda' if available, 'mps' if Apple Silicon, 'cpu' otherwise
```

This is used in:
- RAG pipeline embeddings
- Sentence transformers
- Cross-encoder rerankers
- All PyTorch-based models
- Training scripts

## Troubleshooting

### CUDA Not Available

**Issue**: `torch.cuda.is_available()` returns `False`

**Solutions**:
1. Verify NVIDIA drivers: `nvidia-smi` (must show GPU)
2. Check CUDA version compatibility: `nvcc --version` (if installed)
3. Reinstall PyTorch with correct CUDA version
4. Verify GPU compute capability: Must be 7.0+
5. Check if GPU is in use by another process

### CUDA Out of Memory

**Issue**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size in training/inference
2. Use gradient accumulation instead of larger batches
3. Clear cache: `torch.cuda.empty_cache()`
4. Use mixed precision training: `torch.cuda.amp`
5. Close other GPU-intensive applications
6. Use model parallelism for very large models

### Docker GPU Not Accessible

**Issue**: Docker containers cannot access GPU

**Solutions**:
1. Verify NVIDIA Container Toolkit is installed
2. Check Docker runtime: `docker info | grep -i runtime`
3. Test GPU access: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu20.04 nvidia-smi`
4. Restart Docker service: `sudo systemctl restart docker`
5. Verify WSL2 backend (if on Windows): Docker Desktop > Settings > General > Use WSL 2 based engine

### PyTorch CUDA Version Mismatch

**Issue**: PyTorch CUDA version doesn't match system CUDA

**Solutions**:
1. Check system CUDA: `nvidia-smi` (shows driver CUDA version)
2. Check PyTorch CUDA: `python -c "import torch; print(torch.version.cuda)"`
3. Install matching PyTorch version:
   - For CUDA 12.1: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
   - For CUDA 11.8: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
4. Note: PyTorch CUDA version can be newer than driver CUDA version (backward compatible)

### Performance Issues

**Issue**: CUDA is slower than expected

**Solutions**:
1. Enable cuDNN benchmarking: `export CUDNN_BENCHMARK=1`
2. Use mixed precision: `torch.cuda.amp.autocast()`
3. Optimize data loading: Use `DataLoader` with `num_workers > 0`
4. Check GPU utilization: `nvidia-smi -l 1` (watch GPU usage)
5. Ensure GPU is not in power-saving mode
6. Use TensorFloat-32 (TF32) for Ampere+ GPUs (enabled by default in PyTorch 1.12+)

## Testing CUDA Performance

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
    if device_name == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        x = torch.randn(size, device=device)
        y = torch.randn(size, device=device)
        z = torch.matmul(x, y)
    if device_name == 'cuda':
        torch.cuda.synchronize()
    end = time.time()
    
    print(f"{device_name}: {(end - start) / 100 * 1000:.2f} ms per operation")

if torch.cuda.is_available():
    benchmark_device('cuda')
benchmark_device('cpu')
```

## CUDA Version Compatibility

| PyTorch Version | CUDA Versions Supported |
|----------------|------------------------|
| 2.9.x | 11.8, 12.1, 12.4 |
| 2.8.x | 11.8, 12.1, 12.4 |
| 2.7.x | 11.8, 12.1 |
| 2.6.x | 11.8, 12.1 |
| 2.5.x | 11.8, 12.1 |
| 2.4.x | 11.8, 12.1 |
| 2.3.x | 11.8, 12.1 |
| 2.2.x | 11.8, 12.1 |
| 2.1.x | 11.8, 12.1 |
| 2.0.x | 11.8, 12.1 |

## Additional Resources

- PyTorch CUDA Documentation: https://pytorch.org/docs/stable/cuda.html
- NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
- PyTorch Installation Guide: https://pytorch.org/get-started/locally/

## Common Commands Reference

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version (if CUDA toolkit installed)
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"

# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu20.04 nvidia-smi

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clear CUDA cache (in Python)
python -c "import torch; torch.cuda.empty_cache()"
```

## GPU Verification for Streaming System

The revolutionary streaming system uses GPU acceleration for both LLM inference (Ollama) and tool execution. Verify GPU access with these commands:

### Check GPU Access in Ollama Container

```bash
docker exec deepiri-ollama-dev nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 565.57.01    Driver Version: 565.57.01    CUDA Version: 12.8   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:01:00.0 Off |                  N/A |
| N/A   45C    P8    15W /  N/A |    734MiB / 16303MiB |     15%      Default |
+-------------------------------+----------------------+----------------------+
```

### Check Tensor Core Support

```bash
docker exec deepiri-ollama-dev bash -c 'nvidia-smi --query-gpu=compute_cap --format=csv,noheader'
```

Expected output:
```
12.0
```

**Compute Capability 12.0 = RTX 5080/5090 = Tensor Cores available**

Tensor cores automatically accelerate:
- Transformer attention (FP16/BF16 matrix multiplications)
- Embedding lookups
- Layer normalization
- No code changes needed (PyTorch/ONNX detect automatically)

### Monitor GPU During Inference

```bash
watch -n 0.5 'docker exec deepiri-ollama-dev nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits'
```

Expected during inference:
```
Idle:      15%, 734 MB
Inference: 85-100%, 8600 MB  (model + batch in VRAM)
```

### Verify GPU in Cyrex Container

```bash
docker exec deepiri-cyrex-dev nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader,nounits
```

Expected:
```
NVIDIA GeForce RTX 5080 Laptop GPU, 0, 9269
```

### Docker Compose GPU Configuration

The streaming system requires GPU access in `docker-compose.dev.yml`:

```yaml
ollama:
  runtime: nvidia  # Enable GPU
  environment:
    CUDA_VISIBLE_DEVICES: "0"
    NVIDIA_VISIBLE_DEVICES: "all"
    NVIDIA_DRIVER_CAPABILITIES: "compute,utility"
    OLLAMA_NUM_GPU: "1"
    OLLAMA_GPU_OVERHEAD: "0"  # Minimize VRAM overhead
    OLLAMA_MAX_LOADED_MODELS: "2"  # Keep 2 models in VRAM
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### Performance Metrics

With GPU acceleration enabled:

| Metric | CPU Only | GPU (RTX 5080) | Improvement |
|--------|----------|----------------|-------------|
| First token latency | 10,400ms | **150-200ms** | **98% faster** |
| Tool call latency | 1,090ms | **450ms** | **59% faster** |
| GPU utilization | 0% | **15-100%** | Fully active |
| Perceived latency | 1-10s | **<200ms** | **Instant** |

### Troubleshooting

**GPU not accessible:**
1. Verify NVIDIA Container Toolkit is installed: `nvidia-container-toolkit --version`
2. Check Docker runtime: `docker info | grep -i runtime`
3. Restart Docker daemon: `sudo systemctl restart docker`
4. Verify GPU in host: `nvidia-smi` (outside Docker)

**Low GPU utilization:**
1. Check if model is loaded: `docker exec deepiri-ollama-dev curl -s http://localhost:11434/api/ps`
2. Verify CUDA_VISIBLE_DEVICES: `docker exec deepiri-ollama-dev env | grep CUDA`
3. Check compute capability: Should be 7.0+ (12.0 for RTX 5080)

**Tensor cores not active:**
- Tensor cores activate automatically for FP16/BF16 operations
- Verify model precision: `docker exec deepiri-ollama-dev curl -s http://localhost:11434/api/show -d '{"name":"mistral-nemo:12b"}' | grep quantization`
- Expected: Q4_0 or Q8_0 quantization (uses tensor cores)

