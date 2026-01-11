# MacBook Team Member Setup Guide

This guide is specifically for MacBook team members who need to run the Deepiri development environment with GPU acceleration.

## Important: MacBooks Cannot Use CUDA

**Key Point**: MacBooks (M1/M2/M3/M4) do NOT have NVIDIA GPUs and cannot use CUDA. They use Apple's Metal Performance Shaders (MPS) instead.

- CUDA = NVIDIA GPUs only (Linux/Windows with NVIDIA GPU)
- MPS = Apple Silicon GPUs only (MacBooks)
- These are NOT interchangeable

## Setup Options for MacBook Team Members

### Option 1: Native macOS Development (Recommended)

Run everything natively on macOS for best performance and GPU acceleration.

#### Step 1: Install Ollama for macOS

Ollama on macOS automatically uses Metal (Apple GPU) for acceleration:

```bash
# Install Ollama using Homebrew (recommended)
brew install ollama

# Or download from: https://ollama.ai/download
```

Start Ollama service:

```bash
ollama serve
```

Ollama will automatically use your Mac's GPU (Metal) - no CUDA needed.

#### Step 2: Install PyTorch with MPS Support

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (includes MPS support automatically)
pip install torch torchvision torchaudio

# Verify MPS is available
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

#### Step 3: Install Project Dependencies

```bash
cd deepiri-platform/diri-cyrex
pip install -r requirements.txt
```

#### Step 4: Configure Environment

Set environment variables for native macOS:

```bash
# In your .zshrc or .bash_profile
export OLLAMA_BASE_URL=http://localhost:11434
export LOCAL_LLM_BACKEND=ollama
export LOCAL_LLM_MODEL=llama3:8b
```

#### Step 5: Pull Ollama Models

```bash
# Pull the default model
ollama pull llama3:8b

# Verify it's using GPU
ollama run llama3:8b "test"
# Check Activity Monitor - you should see GPU usage
```

### Option 2: Docker for Services, Native for GPU Work

Use Docker for services that don't need GPU, but run PyTorch/Ollama natively.

#### Docker Setup (CPU-only services)

```bash
# Start services that don't need GPU
docker compose -f docker-compose.dev.yml up -d postgres redis

# Run Ollama natively (outside Docker)
ollama serve
```

#### Connect Native Ollama to Docker Services

Update `docker-compose.dev.yml` to use host Ollama:

```yaml
services:
  cyrex:
    environment:
      OLLAMA_BASE_URL: http://host.docker.internal:11434  # Connect to native Ollama
```

### Option 3: Remote GPU Server (For Heavy Workloads)

If you need CUDA for training or large models, use a remote Linux server with NVIDIA GPU.

#### Setup Remote Server

1. Set up a Linux server with NVIDIA GPU
2. Install CUDA, PyTorch, and Ollama on the server
3. Configure SSH access
4. Connect from MacBook via API or SSH

#### Connect from MacBook

```python
# Example: Connect to remote Ollama
import httpx

OLLAMA_URL = "http://your-gpu-server:11434"
response = httpx.get(f"{OLLAMA_URL}/api/tags")
```

## Docker Limitations on macOS

### Why MPS Doesn't Work in Docker

- Docker on macOS runs Linux containers in a VM
- Metal (MPS) is macOS-specific and doesn't work in Linux containers
- You can only use CPU in Docker containers on macOS

### Workaround: Hybrid Approach

```bash
# Services in Docker (CPU only)
docker compose -f docker-compose.dev.yml up -d postgres redis

# GPU-accelerated services natively
ollama serve  # Uses Metal GPU
python app.py  # Uses MPS for PyTorch
```

## Verification Steps

### 1. Verify Ollama is Using GPU

```bash
# Start Ollama
ollama serve

# In another terminal, run a model
ollama run llama3:8b "Hello"

# Check Activity Monitor:
# - Open Activity Monitor
# - Go to Window > GPU History
# - You should see GPU usage when Ollama is running
```

### 2. Verify PyTorch MPS

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

### 3. Verify Deepiri Integration

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
- Monitor memory usage: `Activity Monitor > Memory`
- Close other apps when running large models
- Consider using smaller models (llama3:8b instead of llama3:70b)

## Troubleshooting

### Ollama Not Using GPU

**Symptoms**: Slow inference, no GPU usage in Activity Monitor

**Solutions**:
1. Verify Ollama version: `ollama --version` (should be recent)
2. Check model size: Smaller models work better on Mac
3. Restart Ollama: `pkill ollama && ollama serve`
4. Verify Metal is available: System Information > Graphics/Displays

### PyTorch MPS Not Available

**Symptoms**: `torch.backends.mps.is_available()` returns `False`

**Solutions**:
1. Update macOS: Must be 12.3+ (Monterey or newer)
2. Reinstall PyTorch: `pip install --upgrade torch torchvision torchaudio`
3. Check Python version: Must be 3.8+
4. Verify architecture: `uname -m` should return `arm64`

### Docker Services Can't Connect to Native Ollama

**Symptoms**: Connection refused when connecting to Ollama from Docker

**Solutions**:
1. Use `host.docker.internal:11434` instead of `localhost:11434`
2. Verify Ollama is running: `curl http://localhost:11434/api/tags`
3. Check firewall settings
4. Use Docker network mode: `network_mode: host` (Linux only, not macOS)

## Recommended Setup for MacBook Team

### Minimal Setup (Fastest to Get Started)

```bash
# 1. Install Ollama
brew install ollama

# 2. Install Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio
pip install -r requirements.txt

# 3. Start Ollama
ollama serve

# 4. Pull model
ollama pull llama3:8b

# 5. Run services (use Docker for non-GPU services)
docker compose -f docker-compose.dev.yml up -d postgres redis
```

### Full Setup (All Services)

```bash
# 1. Install Ollama natively
brew install ollama

# 2. Start Ollama (native, uses Metal GPU)
ollama serve

# 3. Pull models
ollama pull llama3:8b

# 4. Start Docker services (CPU only)
docker compose -f docker-compose.dev.yml up -d

# 5. Configure Cyrex to use native Ollama
export OLLAMA_BASE_URL=http://host.docker.internal:11434

# 6. Start Cyrex service
docker compose -f docker-compose.dev.yml up cyrex
```

## Environment Variables for MacBook

Create a `.env.macbook` file:

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

1. **MacBooks use MPS (Metal), not CUDA** - This is automatic and works well
2. **Ollama on Mac uses Metal GPU** - No CUDA needed, just install Ollama natively
3. **Docker on Mac = CPU only** - Use native services for GPU work
4. **Hybrid approach works best** - Docker for services, native for GPU
5. **Performance is good** - MPS provides solid acceleration for inference

## Additional Resources

- Ollama macOS Installation: https://ollama.ai/download
- PyTorch MPS Documentation: https://pytorch.org/docs/stable/notes/mps.html
- Apple Metal Documentation: https://developer.apple.com/metal/

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

