# GPU Configuration Scripts

Scripts for configuring GPU acceleration for AI workloads.

## Available Scripts

### `install-nvidia-container-toolkit.sh`
Installs NVIDIA Container Toolkit for Docker GPU support on Linux/WSL2.

**Usage:**
```bash
bash scripts/gpu/install-nvidia-container-toolkit.sh
```

**What it does:**
- Detects Linux distribution (Ubuntu/Debian, Fedora/RHEL, Arch, etc.)
- Installs NVIDIA Container Toolkit
- Configures Docker to use NVIDIA runtime
- Tests GPU access in Docker containers

**Requirements:**
- Linux or WSL2
- Docker installed
- NVIDIA drivers installed
- sudo/root access

### `install-nvidia-container-toolkit.ps1`
PowerShell helper for Windows users (redirects to bash script in WSL2).

**Usage:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts/gpu/install-nvidia-container-toolkit.ps1
```

**Note:** For full installation, run the bash script inside WSL2.

### `configure-mps-macos.sh`
Configures MPS (Metal Performance Shaders) for macOS with Ollama, PyTorch, and Milvus.

**Usage:**
```bash
bash scripts/gpu/configure-mps-macos.sh
```

**What it does:**
- Detects Apple Silicon (M1/M2/M3/M4)
- Installs/updates PyTorch with MPS support
- Configures environment variables
- Tests PyTorch MPS functionality
- Checks Ollama and Milvus setup
- Creates `~/.deepiri-mps-env` configuration file

**Requirements:**
- macOS (Apple Silicon recommended)
- Python 3 and pip3
- Homebrew (optional, for Ollama)

## GPU Support by Platform

| Platform | GPU Type | Script | Notes |
|----------|----------|--------|-------|
| Linux/WSL2 | NVIDIA | `install-nvidia-container-toolkit.sh` | Requires NVIDIA drivers |
| macOS | Apple Silicon | `configure-mps-macos.sh` | M1/M2/M3/M4 only |
| macOS | Intel | N/A | CPU only (no GPU acceleration) |
| Windows | NVIDIA | Use WSL2 + bash script | Docker Desktop with WSL2 backend |

## Troubleshooting

### NVIDIA GPU Not Detected
1. Verify drivers: `nvidia-smi`
2. Check Docker GPU access: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi`
3. Restart Docker: `sudo systemctl restart docker`

### MPS Not Available on macOS
1. Verify Apple Silicon: `sysctl -n machdep.cpu.brand_string`
2. Update PyTorch: `pip3 install --upgrade torch`
3. Check MPS: `python3 -c "import torch; print(torch.backends.mps.is_available())"`

## Environment Variables

After running the configuration scripts, these environment variables may be set:

**NVIDIA (Linux):**
- Automatically configured via Docker runtime

**MPS (macOS):**
- `PYTORCH_ENABLE_MPS_FALLBACK=1`
- `OLLAMA_NUM_GPU=1`
- See `~/.deepiri-mps-env` for full list

