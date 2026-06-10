# Local Model Setup Guide

This guide will help you set up Cyrex to use local models instead of OpenAI, saving costs while maintaining functionality.

## Quick Start

### 1. Install Ollama (Recommended - Easiest)

**macOS:**
```bash
brew install ollama
ollama serve
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

**Windows:**
Download from https://ollama.com/download

### 2. Install Models

**Option A: Pre-pull During Docker Build (Recommended - Fastest First Request)**

The Docker setup automatically pre-pulls multiple models during build, eliminating the first-request penalty (1.2-3.3s):

```bash
# Default models (mistral:7b, llama3:8b, codellama:7b) are pre-pulled automatically
docker-compose build ollama

# Or customize which models to pre-pull (space-separated)
docker-compose build --build-arg MODELS="mistral:7b llama3:8b codellama:7b gemma2:2b" ollama

# To disable pre-pull and use script instead
docker-compose build --build-arg PRE_PULL_MODELS=false ollama
```

**Default Models Pre-pulled:**
- **mistral:7b** (4.1GB) - Default model, efficient and high quality
- **llama3:8b** (4.7GB) - Alternative general-purpose model
- **codellama:7b** (3.8GB) - Specialized for coding tasks

**Option B: Use Installation Script (Flexible Runtime Installation)**

```bash
# Interactive script to install models based on your hardware
./scripts/llm/check-ollama-models.sh
```

**Option C: Manual Installation**

```bash
# Recommended models (pick one or more):
ollama pull mistral:7b         # Default, fast, good quality (4.1GB)
ollama pull llama3:8b          # Alternative general-purpose (4.7GB)
ollama pull codellama:7b       # Code-focused (3.8GB)
ollama pull llama3:70b         # Better quality (40GB RAM, requires 48GB+ VRAM)
```

### 3. Configure Environment

Create/update `.env` file:

```env
# Local LLM Configuration
LOCAL_LLM_BACKEND=ollama
LOCAL_LLM_MODEL=mistral:7b  # Default model (changed from llama3:8b)
OLLAMA_BASE_URL=http://localhost:11434

# Milvus (for vector store)
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Redis (for state management and queues)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 4. Start Services

```bash
# Start Milvus (if using Docker)
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest

# Start Redis (if using Docker)
docker run -d --name redis -p 6379:6379 redis:latest

# Start Cyrex
cd deepiri/diri-cyrex
python -m app.main
```

## Architecture Overview

### Core Components

1. **WorkflowOrchestrator** (`app/core/orchestrator.py`)
   - Main orchestration engine
   - Coordinates LLM, RAG, tools, state

2. **LocalLLMProvider** (`app/integrations/local_llm.py`)
   - Supports Ollama, llama.cpp, transformers
   - Unified interface for local models

3. **MilvusVectorStore** (`app/integrations/milvus_store.py`)
   - Production vector database
   - Integrated with existing RAG system

4. **WorkflowStateManager** (`app/core/state_manager.py`)
   - Persistent state tracking
   - Checkpoints and rollback

5. **ToolRegistry** (`app/core/tool_registry.py`)
   - Centralized tool management
   - Dynamic tool registration

6. **SafetyGuardrails** (`app/core/guardrails.py`)
   - Content filtering
   - Prompt injection detection
   - PII detection

7. **TaskQueueManager** (`app/core/queue_manager.py`)
   - Async task queues
   - Priority-based execution
   - Retry logic

8. **TaskExecutionEngine** (`app/core/execution_engine.py`)
   - Workflow execution
   - Step-by-step decomposition
   - Execution trees

9. **SystemMonitor** (`app/core/monitoring.py`)
   - Cost tracking
   - Latency monitoring
   - Drift detection
   - Safety scoring

10. **PromptVersionManager** (`app/core/prompt_manager.py`)
    - Prompt versioning
    - A/B testing
    - Template management

## API Endpoints

### Process Request
```bash
POST /orchestration/process
{
  "user_input": "Generate a summary of my tasks",
  "user_id": "user123",
  "use_rag": true,
  "use_tools": true
}
```

### Execute Workflow
```bash
POST /orchestration/workflow
{
  "workflow_id": "workflow_123",
  "steps": [
    {
      "name": "retrieve_context",
      "tool": "knowledge_retrieval",
      "input": {"query": "user tasks"}
    },
    {
      "name": "generate_summary",
      "input": {"context": "{{retrieve_context.output}}"}
    }
  ]
}
```

### Get Status
```bash
GET /orchestration/status
```

## Model Customization

### Pre-pulling Models During Docker Build

The Docker setup supports pre-pulling multiple models during the build process, which:
- ✅ Eliminates first-request penalty (1.2-3.3s model loading time)
- ✅ Models are baked into the image (no runtime download)
- ✅ Works even on fresh deployments

**Configuration in `docker-compose.dev.yml`:**
```yaml
ollama:
  build:
    context: ./docker/ollama
    dockerfile: Dockerfile
    args:
      PRE_PULL_MODELS: "true"  # Set to "false" to disable
      MODELS: "mistral:7b llama3:8b codellama:7b"  # Space-separated list
```

**Customizing Models:**
```bash
# Build with default models (mistral:7b, llama3:8b, codellama:7b)
docker-compose build ollama

# Build with custom models (space-separated)
docker-compose build --build-arg MODELS="mistral:7b llama3:8b" ollama

# Add more models
docker-compose build --build-arg MODELS="mistral:7b llama3:8b codellama:7b gemma2:2b phi3:mini" ollama

# Disable pre-pull (use check-ollama-models.sh script instead)
docker-compose build --build-arg PRE_PULL_MODELS=false ollama
```

**Default Models:**
- **mistral:7b** (4.1GB) - Default model, efficient and high quality
- **llama3:8b** (4.7GB) - Alternative general-purpose model
- **codellama:7b** (3.8GB) - Specialized for coding tasks

**See `docker/ollama/README.md` for detailed documentation on model customization.**

## Model Comparison

### Ollama (Recommended for Start)
- **Pros**: Easy setup, good performance, many models, pre-pull support
- **Cons**: Requires Ollama service running
- **Best for**: Development and production

### llama.cpp
- **Pros**: Direct model loading, no service needed
- **Cons**: Requires model files (.gguf), more setup
- **Best for**: Embedded systems, offline use

### Transformers (HuggingFace)
- **Pros**: Full control, fine-tuning support
- **Cons**: High memory usage, slower
- **Best for**: Research, custom models

## Cost Savings

Using local models vs OpenAI:

- **OpenAI GPT-4**: ~$0.03 per 1K tokens
- **Local Llama 3 8B**: $0.00 (just compute)
- **Savings**: 100% on API costs

Estimated monthly savings for 1M tokens:
- OpenAI: ~$30
- Local: $0 (just server costs)

## Performance Tips

1. **Use GPU** if available:
   ```bash
   # For llama.cpp with GPU
   CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
   ```

2. **Optimize model size**:
   - 8B models: Fast, good for most tasks
   - 70B models: Better quality, slower

3. **Enable caching**:
   - Redis caching for embeddings
   - Response caching for repeated queries

## Troubleshooting

### Ollama not responding
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Model not found
```bash
# List available models
ollama list

# Pull missing model
ollama pull mistral:7b  # Default model
ollama pull llama3:8b    # Alternative
ollama pull codellama:7b # Coding model

# Or use the installation script
./scripts/llm/check-ollama-models.sh
```

### Models not pre-pulled during build
```bash
# Check if models are in the image
docker exec deepiri-ollama-dev ollama list

# If models are missing, either:
# 1. Rebuild with pre-pull enabled
docker-compose build --build-arg PRE_PULL_MODELS=true ollama

# 2. Or install at runtime using the script
./scripts/llm/check-ollama-models.sh

# 3. Or manually pull
docker exec deepiri-ollama-dev ollama pull mistral:7b
```

### Milvus connection failed
```bash
# Check Milvus status
docker ps | grep milvus

# Start Milvus
docker start milvus
```

### Redis connection failed
```bash
# Check Redis status
redis-cli ping

# Start Redis
docker start redis
```

## Next Steps

1. Test with a simple request:
   ```bash
   curl -X POST http://localhost:8000/orchestration/process \
     -H "Content-Type: application/json" \
     -H "x-api-key: your-key" \
     -d '{"user_input": "Hello, how are you?"}'
   ```

2. Monitor performance:
   ```bash
   curl http://localhost:8000/orchestration/status
   ```

3. Check logs for any issues

4. Start using in production!

## File Structure

```
app/
├── core/                    # Core orchestration
│   ├── orchestrator.py     # Main orchestrator
│   ├── execution_engine.py # Task execution
│   ├── state_manager.py    # State persistence
│   ├── tool_registry.py   # Tool management
│   ├── guardrails.py       # Safety checks
│   ├── queue_manager.py    # Async queues
│   ├── monitoring.py       # Metrics & analytics
│   └── prompt_manager.py   # Prompt versioning
│
├── integrations/           # External integrations
│   ├── local_llm.py       # Local LLM provider
│   ├── milvus_store.py    # Milvus vector store
│   └── rag_bridge.py      # RAG system bridge
│
└── routes/
    └── orchestration_api.py # API endpoints
```

All file names are developer-friendly and follow proper naming conventions!

