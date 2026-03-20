# LLM Management Scripts

Scripts for managing local LLMs, primarily Ollama.

## Available Scripts

### `install-local-llm.sh` / `install-local-llm.ps1`
Installs and configures Ollama for local LLM inference.

**Usage:**
```bash
# Linux/macOS
bash scripts/llm/install-local-llm.sh

# Windows PowerShell
powershell -ExecutionPolicy Bypass -File scripts/llm/install-local-llm.ps1
```

**What it does:**
- Detects operating system and GPU
- Installs Ollama
- Configures Docker for GPU support (if applicable)
- Sets up environment variables

### `check-ollama-models.sh`
Checks Ollama container for models and prompts to pull if none exist.

**Usage:**
```bash
bash scripts/llm/check-ollama-models.sh
```

**What it does:**
- Detects GPU and drivers
- Checks Ollama container status
- Lists available models
- Prompts to pull models if none exist (interactive menu)
- Recommends models based on your hardware
- Configures NVIDIA Container Toolkit if needed

**Note:** This script is useful when you disable model pre-pull during Docker build (`PRE_PULL_MODELS=false`). By default, models are pre-pulled during build for faster first requests.

### `test-ollama-connection.sh`
Quick script to test Ollama connection and list models.

**Usage:**
```bash
bash scripts/llm/test-ollama-connection.sh
```

**What it does:**
- Tests connection to Ollama API
- Lists available models
- Shows model details

### `setup_local_model_in_docker.sh`
Sets up local models in Docker container.

**Usage:**
```bash
bash scripts/llm/setup_local_model_in_docker.sh
```

**What it does:**
- Configures Docker for model storage
- Sets up volume mounts
- Prepares environment for local models

## Common Tasks

### Install Ollama
```bash
bash scripts/llm/install-local-llm.sh
```

### Check Available Models
```bash
bash scripts/llm/check-ollama-models.sh
```

### Test Connection
```bash
bash scripts/llm/test-ollama-connection.sh
```

### Pull a Model

**Option 1: Pre-pull During Docker Build (Recommended)**
```bash
# Default models (mistral:7b, llama3:8b, codellama:7b) are pre-pulled automatically
docker-compose build ollama

# Customize which models to pre-pull
docker-compose build --build-arg MODELS="mistral:7b llama3:8b codellama:7b" ollama
```

**Option 2: Use Installation Script**
```bash
# Interactive script with hardware-based recommendations
bash scripts/llm/check-ollama-models.sh
```

**Option 3: Manual Pull**
```bash
# Pull default model
ollama pull mistral:7b

# Pull alternative models
ollama pull llama3:8b
ollama pull codellama:7b
```

### List Models
```bash
ollama list
```

## Ollama API Endpoints

Once Ollama is running, you can access:

- **List models**: `http://localhost:11434/api/tags`
- **Generate**: `http://localhost:11434/api/generate`
- **Chat**: `http://localhost:11434/api/chat`

## Docker Integration

### Using Docker Compose (Recommended)

The project includes a custom Dockerfile that pre-pulls models during build:

```bash
# Build with default models (mistral:7b, llama3:8b, codellama:7b)
docker-compose build ollama

# Customize models to pre-pull
docker-compose build --build-arg MODELS="mistral:7b llama3:8b" ollama

# Disable pre-pull (use check-ollama-models.sh script instead)
docker-compose build --build-arg PRE_PULL_MODELS=false ollama
```

**Configuration in `docker-compose.dev.yml`:**
```yaml
ollama:
  build:
    context: ./docker/ollama
    dockerfile: Dockerfile
    args:
      PRE_PULL_MODELS: "true"  # Enable/disable pre-pull
      MODELS: "mistral:7b llama3:8b codellama:7b"  # Space-separated list
```

### Manual Docker Run

```bash
# With GPU (Linux)
docker run -d --gpus all -p 11434:11434 --name ollama ollama/ollama

# CPU only
docker run -d -p 11434:11434 --name ollama ollama/ollama
```

### Model Customization

**Default Models:**
- `mistral:7b` (4.1GB) - Default model, efficient and high quality
- `llama3:8b` (4.7GB) - Alternative general-purpose model
- `codellama:7b` (3.8GB) - Specialized for coding tasks

**See `docker/ollama/README.md` for detailed documentation on model customization.**

## Troubleshooting

### Ollama Not Starting
1. Check if port 11434 is available: `lsof -i :11434`
2. Check Ollama logs: `docker logs ollama` (if in Docker)
3. Restart Ollama: `ollama serve`

### Models Not Loading
1. Check disk space: `df -h`
2. Verify model exists: `ollama list`
3. Re-pull model: `ollama pull <model-name>`

### GPU Not Used
1. Verify GPU detection: `nvidia-smi` (NVIDIA) or check MPS (macOS)
2. Check Ollama logs for GPU usage
3. Ensure NVIDIA Container Toolkit is installed (Linux)

## Environment Variables

- `OLLAMA_HOST`: Ollama server address (default: `0.0.0.0:11434`)
- `OLLAMA_NUM_GPU`: Number of GPUs to use (default: `1`)
- `OLLAMA_NUM_PARALLEL`: Number of parallel requests (default: `1`)

