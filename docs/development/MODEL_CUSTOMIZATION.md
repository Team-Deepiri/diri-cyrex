# Model Customization Guide

This guide explains how to customize which Ollama models are pre-pulled during Docker build and how to manage models at runtime.

## Overview

The Cyrex project supports two approaches for model installation:

1. **Pre-pull during Docker build** (Recommended) - Models baked into image, fastest first request
2. **Runtime installation** - Flexible, use `check-ollama-models.sh` script or manual pull

## Pre-pull During Docker Build

### Default Configuration

By default, the following models are pre-pulled during Docker build:

- **mistral:7b** (4.1GB) - Default model, efficient and high quality
- **llama3:8b** (4.7GB) - Alternative general-purpose model
- **codellama:7b** (3.8GB) - Specialized for coding tasks

**Total size**: ~12.6GB

### Configuration

Models are configured in `docker-compose.dev.yml`:

```yaml
ollama:
  build:
    context: ./docker/ollama
    dockerfile: Dockerfile
    args:
      PRE_PULL_MODELS: "true"  # Enable/disable pre-pull
      MODELS: "mistral:7b llama3:8b codellama:7b"  # Space-separated list
```

### Customizing Models

**Option 1: Modify `docker-compose.dev.yml`**

Edit the `MODELS` build arg:

```yaml
args:
  MODELS: "mistral:7b llama3:8b gemma2:2b phi3:mini"
```

Then rebuild:
```bash
docker-compose build ollama
```

**Option 2: Override via Command Line**

```bash
# Build with custom models
docker-compose build --build-arg MODELS="mistral:7b llama3:8b" ollama

# Add more models
docker-compose build --build-arg MODELS="mistral:7b llama3:8b codellama:7b gemma2:2b" ollama
```

**Option 3: Disable Pre-pull**

```bash
# Disable pre-pull, use script instead
docker-compose build --build-arg PRE_PULL_MODELS=false ollama
```

### Model Recommendations

**Small & Fast (Good for CPU):**
- `mistral:7b` (4.1GB) - Default, balanced
- `llama3.2:1b` (1.3GB) - Very fast, smaller
- `phi3:mini` (2.3GB) - Fast, efficient

**Balanced (GPU Recommended):**
- `llama3:8b` (4.7GB) - General purpose
- `gemma2:9b` (5.4GB) - Google's model
- `qwen2.5:7b` (4.4GB) - Alibaba's model

**Coding Tasks:**
- `codellama:7b` (3.8GB) - General coding
- `codellama:13b` (7.3GB) - Larger, better quality
- `deepseek-coder:6.7b` (4.1GB) - Advanced coding

**Large & Powerful (Requires 16GB+ VRAM):**
- `mistral-nemo:12b` (7.0GB) - Enhanced Mistral
- `llama3.1:8b` (4.7GB) - Latest Llama
- `gemma2:27b` (16GB) - Very large

## Runtime Installation

### Using Installation Script

The `check-ollama-models.sh` script provides an interactive way to install models based on your hardware:

```bash
./scripts/llm/check-ollama-models.sh
```

**Features:**
- Detects GPU and system RAM
- Recommends models based on hardware
- Interactive menu for model selection
- Handles model installation automatically

### Manual Installation

```bash
# Pull models manually
docker exec deepiri-ollama-dev ollama pull mistral:7b
docker exec deepiri-ollama-dev ollama pull llama3:8b
docker exec deepiri-ollama-dev ollama pull codellama:7b

# Or if running Ollama locally (not in Docker)
ollama pull mistral:7b
ollama pull llama3:8b
ollama pull codellama:7b
```

## Changing Default Model

The default model is configured in `app/settings.py`:

```python
LOCAL_LLM_MODEL: str = "mistral:7b"  # Default model
```

To change the default:

1. **Update settings:**
   ```python
   LOCAL_LLM_MODEL: str = "llama3:8b"  # Your preferred model
   ```

2. **Ensure model is available:**
   ```bash
   # Check if model exists
   docker exec deepiri-ollama-dev ollama list
   
   # Pull if missing
   docker exec deepiri-ollama-dev ollama pull llama3:8b
   ```

3. **Or pre-pull during build:**
   ```bash
   docker-compose build --build-arg MODELS="llama3:8b" ollama
   ```

## Troubleshooting

### Models Not Pre-pulled

**Check if models are in the image:**
```bash
docker exec deepiri-ollama-dev ollama list
```

**If models are missing:**
```bash
# Option 1: Rebuild with pre-pull enabled
docker-compose build --build-arg PRE_PULL_MODELS=true ollama

# Option 2: Install at runtime
./scripts/llm/check-ollama-models.sh

# Option 3: Manual pull
docker exec deepiri-ollama-dev ollama pull mistral:7b
```

### Build Fails with "Model pull failed"

**Possible causes:**
- No internet access during build
- Model name incorrect (check Ollama registry)
- Network timeout

**Solutions:**
- Ensure internet access during build
- Verify model names at https://ollama.com/library
- Use runtime installation instead: `PRE_PULL_MODELS=false`

### Image Too Large

**Reduce number of models:**
```bash
docker-compose build --build-arg MODELS="mistral:7b" ollama
```

**Use smaller models:**
```bash
docker-compose build --build-arg MODELS="llama3.2:1b phi3:mini" ollama
```

**Disable pre-pull:**
```bash
docker-compose build --build-arg PRE_PULL_MODELS=false ollama
```

## Best Practices

1. **For Production:** Pre-pull models during build for fastest first request
2. **For Development:** Use runtime installation for flexibility
3. **For Testing:** Use smaller models (1-3B) for faster iteration
4. **For Production:** Use default models (mistral:7b, llama3:8b, codellama:7b)

## Related Documentation

- `docker/ollama/README.md` - Detailed Docker setup documentation
- `docs/development/LOCAL_MODEL_SETUP.md` - General local model setup
- `scripts/llm/README.md` - LLM management scripts documentation

