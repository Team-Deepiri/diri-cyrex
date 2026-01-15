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
- Prompts to pull models if none exist
- Configures NVIDIA Container Toolkit if needed

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
```bash
ollama pull llama3:8b
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

Ollama can run in Docker with GPU support:

```bash
# With GPU (Linux)
docker run -d --gpus all -p 11434:11434 --name ollama ollama/ollama

# CPU only
docker run -d -p 11434:11434 --name ollama ollama/ollama
```

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

