# Diri-Cyrex — AI Intelligence Engine

> The AI/ML microservice that powers the Deepiri platform with agent orchestration, document intelligence, RAG, and vendor fraud detection.

## What Is Cyrex?

Cyrex is a **Python/FastAPI microservice** (port 8000) that provides AI capabilities to the rest of the Deepiri ecosystem. Other services call Cyrex to:

- **Chat with AI agents** that remember context and use tools
- **Break down complex tasks** into actionable steps
- **Analyze documents** — invoices, contracts, PDFs, and extract structured data
- **Detect vendor fraud** — inflated billing, phantom work, kickbacks across 6 industries
- **Search knowledge** — upload documents, ask natural language questions, get answers from those documents (RAG)
- **Orchestrate multi-agent workflows** — specialized agents working together in pipelines

Think of Cyrex as the **brain** of the Deepiri platform.

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Git (with submodule support)

### Start with Docker Compose (Recommended)

Cyrex is part of the larger **Deepiri Platform** monorepo. From the platform root:

```bash
# 1. Clone and initialize submodules
git clone git@github.com:Team-Deepiri/deepiri-platform.git
cd deepiri-platform
git submodule update --init --recursive

# 2. Start AI team services
cd team_dev_environments/ai-team
./build.sh && ./start.sh
```

Or start Cyrex directly with docker compose:

```bash
cd deepiri-platform
docker compose -f docker-compose.dev.yml up -d \
  postgres redis influxdb etcd minio milvus \
  cyrex cyrex-interface ollama synapse synapse-sugar-glider
```

### Start Locally (Development)

```bash
# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Start dependencies (PostgreSQL, Redis, Milvus, etc.)
# Then:
python -m app.main
```

Or use the file watcher for auto-reload:

```bash
python cyrex_watcher.py
```

### Access Services

| Service | URL |
|---------|-----|
| Cyrex API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Cyrex Interface (UI) | http://localhost:5175 |
| Health Check | http://localhost:8000/health |
| Metrics (Prometheus) | http://localhost:8000/metrics |

## Architecture

```
User/API Request
    ↓
FastAPI Server (port 8000)
    ↓
Router → Agent(s) → LLM (Ollama / OpenAI)
    ↓
Storage: PostgreSQL + Redis + Milvus (vector DB)
    ↓
Response
```

### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| API Server | `app/main.py` | FastAPI entry point, all routes |
| Agent System | `app/agents/` | Multi-agent framework (orchestrator, task decomposer, QA, etc.) |
| Orchestrator | `app/core/orchestrator.py` | Coordinates all components, tool execution, streaming |
| LLM Integration | `app/integrations/` | Ollama (local) and OpenAI (cloud) support |
| RAG Engine | `app/integrations/universal_rag_engine.py` | Document indexing, retrieval, generation |
| Vendor Fraud | `app/services/vendor_intelligence_service.py` | 6-industry fraud detection |
| Document Indexing | `app/services/document_indexing_service.py` | Multi-format document parsing |
| System Init | `app/core/system_initializer.py` | Bootstrap all services at startup |

### Agent Roles

| Role | Description |
|------|-------------|
| `orchestrator` | Directs workflow and delegates tasks |
| `task_decomposer` | Breaks big tasks into smaller steps |
| `time_optimizer` | Optimizes scheduling |
| `creative_sparker` | Generates creative ideas |
| `quality_assurance` | Reviews outputs for quality |
| `engagement_specialist` | Maintains user engagement |
| `memory_manager` | Manages multi-tier memory system |
| `tool_executor` | Executes registered tools |
| `guardrail_enforcer` | Enforces content guardrails |
| `invoice_analyzer` | Analyzes vendor invoices |
| `vendor_intelligence` | Gathers vendor profiles and history |
| `pricing_benchmark` | Compares prices against benchmarks |
| `fraud_detector` | Detects fraud patterns |
| `document_processor` | Processes invoices/documents |
| `risk_assessor` | Assesses risk levels |

## API Authentication

Most endpoints require an API key header:

```bash
curl -H "x-api-key: change-me" http://localhost:8000/agent/intelligence/generate-ability
```

**Endpoints that DON'T require API keys:**
- `GET /health` — Health check
- `GET /metrics` — Prometheus metrics
- `GET /docs` — Swagger UI

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | OpenAI API key (optional, for cloud LLM) |
| `OPENAI_MODEL` | `gpt-4o-mini` | Default OpenAI model |
| `CYREX_API_KEY` | `change-me` | API key for Cyrex endpoints |
| `REDIS_HOST` | `localhost` | Redis host (use `redis` in Docker) |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host (use `postgres` in Docker) |
| `MILVUS_HOST` | `localhost` | Milvus host (use `milvus` in Docker) |
| `LOCAL_LLM_BACKEND` | `ollama` | Local LLM backend |
| `LOCAL_LLM_MODEL` | `llama3:8b` | Ollama model name (`.env.example` default) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `deepiri` | PostgreSQL database name |
| `POSTGRES_USER` | `deepiri` | PostgreSQL user |
| `POSTGRES_PASSWORD` | `deepiripassword` | PostgreSQL password |
| `LLAMA_CPP_MODEL_PATH` | — | Path to GGUF model file (llama_cpp backend) |
| `MESSAGING_SERVICE_URL` | `http://messaging-service:5009` | Synapse messaging service URL |
| `JWT_SECRET` | `default-secret-change-in-production` | JWT secret key |
| `CORS_ORIGIN` | `http://localhost:5173` | Allowed CORS origin |
| `NODE_BACKEND_URL` | `http://localhost:5000` | Node backend API URL |
| `LANGCHAIN_API_KEY` | — | LangChain API key (optional) |
| `LANGCHAIN_PROJECT` | `deepiri` | LangChain project name |
| `MINIO_ROOT_USER` | `minioadmin` | MinIO access key |
| `MINIO_ROOT_PASSWORD` | `minioadmin` | MinIO secret key |
| `S3_ENDPOINT_URL` | `http://minio:9000` | S3/MinIO endpoint URL |

## Docker Build

Cyrex uses a hybrid Dockerfile that supports both **prebuilt** (CUDA) and **from-scratch** (CPU) builds:

```bash
# Auto-detect GPU and build
cd deepiri-platform
docker compose -f docker-compose.dev.yml build cyrex

# Force CPU build
docker build --build-arg BUILD_TYPE=from-scratch --build-arg BASE_IMAGE=python:3.11-slim -t deepiri-dev-cyrex:latest .

# GPU build (default)
docker build --build-arg BUILD_TYPE=prebuilt --build-arg BASE_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime -t deepiri-dev-cyrex:latest .
```

### Build Args

| Arg | Default | Purpose |
|-----|---------|---------|
| `BUILD_TYPE` | `prebuilt` | `prebuilt` (CUDA) or `from-scratch` (CPU) |
| `BASE_IMAGE` | `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime` | Base Docker image |
| `DEVICE_TYPE` | `auto` | `auto`, `gpu`, `cpu`, `mpsos` |

## Submodules

| Submodule | Path | URL |
|-----------|------|-----|
| diri-agent-testing-utils | `diri-agent-testing-utils/` | `git@github.com:Team-Deepiri/diri-agent-testing-utils.git` |
| deepiri-dataset-processor | `deepiri-dataset-processor/` | `git@github.com:Team-Deepiri/deepiri-dataset-processor.git` |

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/README.md](docs/README.md) | Full documentation index |
| [docs/getting-started/HOW_TO_USE_CYREX.md](docs/getting-started/HOW_TO_USE_CYREX.md) | How to use the Cyrex API |
| [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) | System architecture |
| [docs/architecture/CORE_SYSTEMS.md](docs/architecture/CORE_SYSTEMS.md) | Core system components |
| [docs/features/AGENTS_IMPLEMENTATION.md](docs/features/AGENTS_IMPLEMENTATION.md) | Agent system documentation |
| [docs/features/COMPLETE_RAG_IMPLEMENTATION.md](docs/features/COMPLETE_RAG_IMPLEMENTATION.md) | RAG system documentation |
| [docs/features/CYREX_VENDOR_FRAUD_SYSTEM.md](docs/features/CYREX_VENDOR_FRAUD_SYSTEM.md) | Vendor fraud detection |
| [docs/features/UNDOCUMENTED_ENDPOINTS.md](docs/features/UNDOCUMENTED_ENDPOINTS.md) | Undocumented API endpoints |
| [docs/getting-started/CYREX_VENDOR_FRAUD_QUICK_START.md](docs/getting-started/CYREX_VENDOR_FRAUD_QUICK_START.md) | Fraud detection quick start |
| [cyrex-interface/README.md](cyrex-interface/README.md) | Frontend interface guide |

## Testing

```bash
# Run all tests
pytest

# Run with test runner script
python scripts/dev/run_tests.py --category all

# Run specific test
pytest tests/test_comprehensive.py
```

See [tests/README.md](tests/README.md) for the full test suite guide.

## Project Structure

```
diri-cyrex/
├── app/                          # FastAPI application
│   ├── main.py                   # Entry point
│   ├── settings.py               # Configuration
│   ├── logging_config.py         # Logging setup
│   ├── agents/                   # Agent system
│   │   ├── base_agent.py         # Base agent class
│   │   ├── agent_factory.py      # Agent factory
│   │   ├── implementations/      # Agent implementations
│   │   ├── prompts/              # Prompt templates
│   │   └── tools/                # Agent tools
│   ├── core/                     # Core systems
│   │   ├── orchestrator.py       # Main orchestrator
│   │   ├── system_initializer.py # Bootstrap
│   │   ├── state_manager.py      # State management
│   │   ├── langgraph_workflow.py # LangGraph workflow engine
│   │   ├── realtime_data_pipeline.py # Real-time data pipeline
│   │   └── ...
│   ├── routes/                   # API routes
│   │   ├── intelligence_api.py   # AI intelligence endpoints
│   │   ├── orchestration_api.py  # Orchestration endpoints
│   │   ├── universal_rag_api.py  # Universal RAG API
│   │   ├── document_indexing_api.py
│   │   ├── vendor_fraud_api.py
│   │   ├── cyrex_guard_api.py
│   │   └── ...
│   ├── services/                 # Business logic
│   │   ├── vendor_intelligence_service.py
│   │   ├── document_indexing_service.py
│   │   ├── fraud_detector.py
│   │   ├── pricing_benchmark.py
│   │   └── ...
│   ├── integrations/             # External integrations
│   │   ├── ollama_container.py   # Ollama integration
│   │   ├── universal_rag_engine.py
│   │   ├── enhanced_universal_rag_engine.py
│   │   └── ...
│   ├── database/                 # Database models
│   ├── config/                   # Configuration
│   ├── middleware/               # HTTP middleware
│   ├── ml_models/                # ML model classification
│   ├── pipeline/                 # Pipeline contracts & tools
│   └── utils/                    # Utility functions
├── cyrex-interface/              # React frontend UI
├── tests/                        # Test suite
├── scripts/                      # Utility scripts
├── docker/                       # Docker configuration
├── cyrex_watcher.py              # Development file watcher
├── Dockerfile                    # GPU-enabled build
├── Dockerfile.cpu                # CPU-only build
└── requirements/                 # Dependency files
```

## License

See [LICENSE.md](LICENSE.md) for license information.
