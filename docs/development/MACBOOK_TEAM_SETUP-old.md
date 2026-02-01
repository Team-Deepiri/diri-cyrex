# MacBook Team Member Setup Guide

## Overview

This guide covers how to run the Deepiri development environment with GPU acceleration on MacBooks with Apple Silicon.

**Important:** MacBooks (M1/M2/M3/M4) do NOT have NVIDIA GPUs and cannot use CUDA. They use Apple's Metal Performance Shaders (MPS) instead.

- **CUDA** = NVIDIA GPUs only (Linux/Windows with NVIDIA GPU)
- **MPS** = Apple Silicon GPUs only (MacBooks)
- These are NOT interchangeable

## Option 1: Native macOS Development (Recommended)

Run everything natively on macOS for best performance and GPU acceleration.

### Installing Ollama for macOS

Ollama on macOS automatically uses Metal (Apple GPU) for acceleration.

```bash
# Install using Homebrew (recommended)
brew install ollama

# Or download from: https://ollama.ai/download
```

Start the Ollama service:

```bash
ollama serve
```

Ollama will automatically use your Mac's GPU (Metal) without any CUDA configuration.

### Installing PyTorch with MPS Support

Create a virtual environment and install PyTorch with MPS support.

```bash
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (includes MPS support automatically)
pip install torch torchvision torchaudio

# Verify MPS is available
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Installing Project Dependencies

Install the Cyrex project requirements.

```bash
cd deepiri-platform/diri-cyrex
pip install -r requirements.txt
```

### Configuring Environment Variables

Set environment variables for native macOS:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export LOCAL_LLM_BACKEND=ollama
export LOCAL_LLM_MODEL=llama3:8b
```

### Pulling Ollama Models

Download the models you need.

```bash
# Pull the default model
ollama pull llama3:8b

# Verify it's using GPU
ollama run llama3:8b "test"
# Check Activity Monitor - you should see GPU usage
```

## Option 2: Docker for Services, Native for GPU Work

Use Docker for services that don't need GPU, but run PyTorch and Ollama natively.

### Docker Setup (CPU-only services)

Start services that don't require GPU acceleration in Docker.

```bash
docker compose -f docker-compose.dev.yml up -d postgres redis
```

Run Ollama natively outside of Docker.

```bash
ollama serve
```

### Connecting Native Ollama to Docker Services

Update `docker-compose.dev.yml` to connect to your host's Ollama instance.

```yaml
services:
  cyrex:
    environment:
      OLLAMA_BASE_URL: http://host.docker.internal:11434
```

## Option 3: Remote GPU Server (For Heavy Workloads)

For CUDA training or large models, connect to a remote Linux server with an NVIDIA GPU.

### Setting Up the Remote Server

- Set up a Linux server with an NVIDIA GPU
- Install CUDA, PyTorch, and Ollama on the server
- Configure SSH access

### Connecting from MacBook

Connect to the remote Ollama instance from your MacBook.

```python
import httpx

OLLAMA_URL = "http://your-gpu-server:11434"
response = httpx.get(f"{OLLAMA_URL}/api/tags")
```

## Docker Limitations on macOS

### Why MPS Doesn't Work in Docker

- Docker on macOS runs Linux containers in a VM
- Metal (MPS) is macOS-specific and doesn't work in Linux containers
- Docker containers on macOS can only use CPU

### Hybrid Approach Workaround

Run GPU-accelerated services natively while using Docker for CPU-only services.

```bash
# Services in Docker (CPU only)
docker compose -f docker-compose.dev.yml up -d postgres redis

# GPU-accelerated services natively
ollama serve  # Uses Metal GPU
python app.py  # Uses MPS for PyTorch
```

## Verification Steps

### Verifying Ollama GPU Usage

Start Ollama and test with a model.

```bash
ollama serve
```

In another terminal:

```bash
ollama run llama3:8b "Hello"
```

Check Activity Monitor by going to **Window > GPU History**. You should see GPU usage when Ollama is running.

### Verifying PyTorch MPS

Run this Python script to verify MPS is working.

```python
import torch

# Check MPS availability
print(f"MPS available: {torch.backends.mps.is_available()}")

# Test MPS
if torch.backends.mps.is_available():
    x = torch.randn(1000, 1000, device='mps')
    y = torch.randn(1000, 1000, device='mps')
    z = torch.matmul(x, y)
    print(f"MPS test successful: {z.device}")
```

### Verifying Deepiri Integration

Check that Deepiri correctly detects your device.

```python
from app.utils.device_detection import get_device

device = get_device()
print(f"Detected device: {device}")  # Should be 'mps' on MacBook
```

## Performance Expectations

### MacBook GPU Performance

| Task | M1/M2/M3 Performance | Notes |
|------|---------------------|-------|
| Ollama inference | 15-40 tokens/sec | Depends on model size |
| PyTorch training | Good for small models | Limited by unified memory |
| Embeddings | Fast | MPS acceleration works well |
| Large model training | Not recommended | Use remote GPU server |

### Memory Considerations

- MacBooks use unified memory (shared between CPU and GPU)
- Monitor memory usage via **Activity Monitor > Memory**
- Close other apps when running large models
- Consider smaller models (`llama3:8b` instead of `llama3:70b`)

## Troubleshooting

### Ollama Not Using GPU

**Symptoms:** Slow inference, no GPU usage in Activity Monitor

**Solutions:**
- Verify Ollama version: `ollama --version` (should be recent)
- Check model size: Smaller models work better on Mac
- Restart Ollama: `pkill ollama && ollama serve`
- Verify Metal is available: **System Information > Graphics/Displays**

### PyTorch MPS Not Available

**Symptoms:** `torch.backends.mps.is_available()` returns `False`

**Solutions:**
- Update macOS: Must be 12.3+ (Monterey or newer)
- Reinstall PyTorch: `pip install --upgrade torch torchvision torchaudio`
- Check Python version: Must be 3.8+
- Verify architecture: `uname -m` should return `arm64`

### Docker Services Can't Connect to Native Ollama

**Symptoms:** Connection refused when connecting to Ollama from Docker

**Solutions:**
- Use `host.docker.internal:11434` instead of `localhost:11434`
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- Check firewall settings
- Note: `network_mode: host` works on Linux only, not macOS

## Recommended Setup for MacBook Team

### Minimal Setup (Fastest to Get Started)

```bash
# Install Ollama
brew install ollama

# Install Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio
pip install -r requirements.txt

# Start Ollama
ollama serve

# Pull model
ollama pull llama3:8b

# Run services (use Docker for non-GPU services)
docker compose -f docker-compose.dev.yml up -d postgres redis
```

### Full Setup (All Services)

```bash
# Install Ollama natively
brew install ollama

# Start Ollama (native, uses Metal GPU)
ollama serve

# Pull models
ollama pull llama3:8b

# Start Docker services (CPU only)
docker compose -f docker-compose.dev.yml up -d

# Configure Cyrex to use native Ollama
export OLLAMA_BASE_URL=http://host.docker.internal:11434

# Start Cyrex service
docker compose -f docker-compose.dev.yml up cyrex
```

## Environment Variables for MacBook

Create a `.env.macbook` file with these settings:

```bash
# Ollama (native, uses Metal GPU)
OLLAMA_BASE_URL=http://localhost:11434

# Local LLM settings
LOCAL_LLM_BACKEND=ollama
LOCAL_LLM_MODEL=llama3:8b

# PyTorch will automatically use MPS (no config needed)

# Docker services (CPU only)
POSTGRES_HOST=localhost
REDIS_HOST=localhost
```

## Comparison: MacBook vs Linux with NVIDIA GPU

| Feature | MacBook (MPS) | Linux (CUDA) |
|---------|---------------|--------------|
| GPU API | Metal (MPS) | CUDA |
| Ollama acceleration | Metal | CUDA |
| PyTorch device | `mps` | `cuda` |
| Docker GPU | Not available | Available |
| Performance | Good for inference | Best for training |
| Memory | Unified (shared) | Separate VRAM |
| Setup complexity | Simple | Requires NVIDIA drivers |

## Key Takeaways

- **MacBooks use MPS (Metal), not CUDA** - This is automatic and works well
- **Ollama on Mac uses Metal GPU** - No CUDA needed, just install Ollama natively
- **Docker on Mac = CPU only** - Use native services for GPU work
- **Hybrid approach works best** - Docker for services, native for GPU
- **Performance is good** - MPS provides solid acceleration for inference

## Additional Resources

- **Ollama macOS Installation:** https://ollama.ai/download
- **PyTorch MPS Documentation:** https://pytorch.org/docs/stable/notes/mps.html
- **Apple Metal Documentation:** https://developer.apple.com/metal/

## Quick Reference Commands

```bash
# Install Ollama
brew install ollama

# Start Ollama (uses Metal GPU automatically)
ollama serve

# Pull model
ollama pull llama3:8b

# Test Ollama
ollama run llama3:8b "Hello"

# Check GPU usage (Activity Monitor)
open -a "Activity Monitor"

# Verify PyTorch MPS
python -c "import torch; print(torch.backends.mps.is_available())"

# Start Docker services (CPU only)
docker compose -f docker-compose.dev.yml up -d postgres redis

# Connect Docker to native Ollama
export OLLAMA_BASE_URL=http://host.docker.internal:11434
```
