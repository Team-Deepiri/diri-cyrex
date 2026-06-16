# How to Use Cyrex AI Service

Cyrex is the AI/ML microservice that provides natural language processing, agent orchestration, embeddings, challenge generation, and model inference capabilities for the Deepiri platform.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [API Key Authentication](#api-key-authentication)
4. [Deepiri AI Endpoints](#deepiri-ai-endpoints)
5. [General API Endpoints](#general-api-endpoints)
6. [Document Intelligence](#document-intelligence)
7. [Vendor Fraud Detection](#vendor-fraud-detection)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)
10. [Using the Interactive API Docs](#using-the-interactive-api-docs)
11. [Integration Examples](#integration-examples)
12. [Common Use Cases](#common-use-cases)

---

## Quick Start

### 1. Start Cyrex Service

**Option A: Using the AI Team Script (Recommended)**

From the `deepiri-platform` root:

```bash
cd team_dev_environments/ai-team
./build.sh && ./start.sh
```

This starts Cyrex along with all dependencies: PostgreSQL, Redis, Milvus, MinIO, Ollama, and Synapse.

**Option B: Using Docker Compose Directly**

From the `deepiri-platform` root:

```bash
docker compose -f docker-compose.dev.yml up -d \
  postgres redis influxdb etcd minio milvus \
  cyrex cyrex-interface ollama synapse synapse-sugar-glider
```

**Option C: Local Development**

```bash
# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Start with auto-reload (file watcher)
python cyrex_watcher.py
```

### 2. Verify Cyrex is Running

```bash
# Check health
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","version":"3.0.0","timestamp":...}
```

### 3. Access Interactive API Documentation

Open your browser and navigate to:
```
http://localhost:8000/docs
```

This provides **Swagger UI** where you can:
- See all available endpoints
- Test endpoints directly in the browser
- View request/response schemas
- Try out different API calls

---

## Architecture Overview

Cyrex is a **FastAPI microservice** that provides AI capabilities through multiple systems:

### Agent System

Cyrex implements a multi-agent framework with specialized roles:

- **Orchestrator** — Directs workflow and delegates tasks
- **Task Decomposer** — Breaks big tasks into smaller steps
- **Time Optimizer** — Optimizes scheduling
- **Creative Sparker** — Generates creative ideas
- **Quality Assurance** — Reviews outputs
- **Vendor Fraud Agents** — Specialized fraud detection agents

### Three-Tier AI System

1. **Intent Classification** (Tier 1) — Routes user commands to predefined abilities using BERT/DeBERTa
2. **Ability Generation** (Tier 2) — Creates dynamic abilities on-the-fly using LLM + RAG
3. **Workflow Optimization** (Tier 3) — RL-based adaptive learning for productivity optimization

### Knowledge Retrieval (RAG)

Cyrex supports Retrieval-Augmented Generation with multiple knowledge bases:
- `user_patterns` — User behavior patterns
- `project_context` — Project-specific context
- `ability_templates` — Pre-defined ability templates
- `rules_knowledge` — Business rules and constraints
- `historical_abilities` — Previously generated abilities

---

## API Key Authentication

**IMPORTANT**: Most Cyrex endpoints require an API key. The default key is `change-me` (or set via `CYREX_API_KEY` environment variable).

**Endpoints that DON'T require API key:**
- `GET /health` — Health check
- `GET /metrics` — Prometheus metrics
- `GET /docs` — Interactive API documentation

**All other endpoints require the API key header:**
```bash
-H "x-api-key: change-me"
```

### Finding Your API Key

```bash
# Check the API key in your .env file
cat .env | grep CYREX_API_KEY
```

**Default API Key**: `change-me` (if not set in environment)

---

## Deepiri AI Endpoints

### Command Routing (Tier 1)

Route a user command to a predefined ability:

```bash
curl -X POST http://localhost:8000/agent/intelligence/route-command \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "command": "Can you review this code for security issues?",
    "user_role": "software_engineer",
    "min_confidence": 0.7,
    "top_k": 3
  }'
```

**Request Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `command` | string | User's natural language command |
| `user_role` | string | User's role (optional) |
| `context` | object | Additional context (optional) |
| `min_confidence` | float | Minimum confidence threshold (default: 0.7) |
| `top_k` | int | Number of top predictions (default: 3) |

### Contextual Ability Generation (Tier 2)

Generate dynamic ability using LLM + RAG:

```bash
curl -X POST http://localhost:8000/agent/intelligence/generate-ability \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "user_command": "I need to refactor this codebase to use TypeScript",
    "user_profile": {
      "role": "software_engineer",
      "momentum": 450,
      "level": 15
    },
    "project_context": {
      "language": "JavaScript",
      "files": 50
    }
  }'
```

### Workflow Optimization (Tier 3)

Get RL optimizer recommendation:

```bash
curl -X POST http://localhost:8000/agent/intelligence/recommend-action \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "user_data": {
      "momentum": 450,
      "current_level": 15,
      "task_completion_rate": 0.85,
      "daily_streak": 7
    }
  }'
```

### Knowledge Retrieval

Query knowledge bases:

```bash
curl -X POST http://localhost:8000/agent/intelligence/knowledge/query \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are effective focus boost strategies?",
    "knowledge_bases": ["user_patterns", "ability_templates"],
    "top_k": 5
  }'
```

Index content into a knowledge base:

```bash
curl -X POST http://localhost:8000/agent/intelligence/knowledge/index \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "User patterns show that focus boosts are most effective in the afternoon...",
    "metadata": {
      "user_id": "user123",
      "type": "user_pattern"
    },
    "knowledge_base": "user_patterns"
  }'
```

### Additional Intelligence Endpoints

**Ability Feedback:**
```bash
curl -X POST http://localhost:8000/agent/intelligence/ability/feedback \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{"feedback": "very helpful", "ability_id": "..."}'
```

**Optimizer Reward:**
```bash
curl -X POST http://localhost:8000/agent/intelligence/optimizer/reward \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "outcome": {
      "task_completed": true,
      "efficiency": 0.92,
      "user_rating": 5
    }
  }'
```

**Optimizer Update:**
```bash
curl -X POST http://localhost:8000/agent/intelligence/optimizer/update \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{"epochs": 10}'
```

**Formatted Knowledge Query:**
```bash
curl -X POST http://localhost:8000/agent/intelligence/knowledge/query-formatted \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are our vendor policies?",
    "top_k": 5
  }'
```

---

## General API Endpoints

### Text Embeddings

Generate vector embeddings for text:

```bash
curl -X POST http://localhost:8000/api/embeddings \
  -H "Content-Type: application/json" \
  -H "x-api-key: change-me" \
  -d '{
    "text": "Hello world, this is a test",
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  }'
```

### AI Completions

Get AI-generated text completions:

```bash
curl -X POST http://localhost:8000/api/complete \
  -H "Content-Type: application/json" \
  -H "x-api-key: change-me" \
  -d '{
    "prompt": "What is artificial intelligence?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

---

## Document Intelligence

### Document Indexing

Index text content:

```bash
curl -X POST http://localhost:8000/agent/document-indexing/index/text \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Document text content...",
    "metadata": {"title": "Example Document", "type": "text"}
  }'
```

Search indexed documents:

```bash
curl -X POST http://localhost:8000/agent/document-indexing/search \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "search terms",
    "top_k": 5
  }'
```

See [docs/features/COMPLETE_RAG_IMPLEMENTATION.md](../features/COMPLETE_RAG_IMPLEMENTATION.md) for the full RAG documentation.

---

## Vendor Fraud Detection

Cyrex includes a comprehensive vendor fraud detection system for 6 industries:

- Property Management
- Corporate Procurement
- Insurance PC
- General Contractors
- Retail E-Commerce
- Law Firms

See [docs/getting-started/CYREX_VENDOR_FRAUD_QUICK_START.md](./CYREX_VENDOR_FRAUD_QUICK_START.md) for a 5-minute quick start guide.

### Fraud Detection Endpoints

**Process an invoice:**
```bash
curl -X POST http://localhost:8000/agent/vendor-fraud/invoice/process \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{"invoice_data": {...}, "industry": "property_management"}'
```

**Detect fraud:**
```bash
curl -X POST http://localhost:8000/agent/vendor-fraud/fraud/detect \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{"context": {...}}'
```

**Pricing benchmark:**
```bash
curl -X POST http://localhost:8000/agent/vendor-fraud/pricing/benchmark \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{"service": "plumbing repair", "industry": "property_management"}'
```

See [docs/features/CYREX_VENDOR_FRAUD_SYSTEM.md](../features/CYREX_VENDOR_FRAUD_SYSTEM.md) for the full vendor fraud documentation.

---

## Configuration

### Environment Variables

Make sure these are set in your `.env` file or Docker environment:

```bash
# Required for AI features
OPENAI_API_KEY=your-openai-api-key-here

# API Key for Cyrex endpoints (default: "change-me")
CYREX_API_KEY=change-me

# Optional
OPENAI_MODEL=gpt-4o-mini

# Docker Compose defaults (override if needed):
MILVUS_HOST=milvus
MILVUS_PORT=19530
REDIS_HOST=redis
POSTGRES_HOST=postgres-cyrex
POSTGRES_PORT=5434
OLLAMA_BASE_URL=http://ollama:11434
```

### Setting a Custom API Key

You can set a custom API key in your `.env` file:

```bash
CYREX_API_KEY=my-secret-api-key-123
```

Then use it in your requests:
```bash
curl -H "x-api-key: my-secret-api-key-123" http://localhost:8000/agent/intelligence/route-command
```

---

## Troubleshooting

### Cyrex Won't Start

```bash
# Check logs
docker compose -f docker-compose.dev.yml logs -f cyrex

# Check if port 8000 is in use
# Linux/Mac
lsof -i :8000

# Restart Cyrex
docker compose -f docker-compose.dev.yml restart cyrex
```

### API Returns 401 "Invalid API key" Error

**This is the most common error!** It means you're missing the API key header.

**Solution:**
```bash
# Always include the API key header for non-health/metrics endpoints
curl -H "x-api-key: change-me" http://localhost:8000/agent/intelligence/route-command
```

### Milvus Connection Issues

```bash
# Ensure Milvus and dependencies are running
docker compose -f docker-compose.dev.yml up -d etcd minio milvus

# Check Milvus health
curl http://localhost:19530/healthz
```

---

## Using the Interactive API Docs

The easiest way to test Cyrex is through the interactive documentation:

1. **Open Swagger UI:**
   ```
   http://localhost:8000/docs
   ```

2. **Try an Endpoint (No API Key):**
   - Click on `GET /health` (doesn't require API key)
   - Click "Try it out"
   - Click "Execute"
   - See the response

3. **Test POST Endpoints (Requires API Key):**
   - Click on `POST /agent/intelligence/route-command`
   - Click "Try it out"
   - **IMPORTANT**: Click "Authorize" button at the top
   - Enter API key: `change-me`
   - Click "Authorize" then "Close"
   - Edit the JSON request body
   - Click "Execute"
   - View the response

---

## Integration Examples

### From Frontend (JavaScript)

```javascript
// Health check (no API key needed)
const response = await fetch('http://localhost:8000/health');
const data = await response.json();
console.log(data);

// Generate embedding (API key required)
const embeddingResponse = await fetch('http://localhost:8000/api/embeddings', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'x-api-key': 'change-me'  // Required!
  },
  body: JSON.stringify({ text: 'Hello world' })
});
const embedding = await embeddingResponse.json();
console.log(embedding);
```

### From Python

```python
import requests

# Health check (no API key needed)
response = requests.get('http://localhost:8000/health')
print(response.json())

# Generate embedding (API key required)
response = requests.post(
    'http://localhost:8000/api/embeddings',
    json={'text': 'Hello world'},
    headers={'x-api-key': 'change-me'}  # Required!
)
print(response.json())

# Route command (API key required)
response = requests.post(
    'http://localhost:8000/agent/intelligence/route-command',
    json={
        'command': 'Review this code',
        'user_role': 'software_engineer'
    },
    headers={'x-api-key': 'change-me'}
)
print(response.json())
```

---

## Cyrex Interface

Prefer a UI over cURL? Launch the Cyrex Interface that lives in `cyrex-interface`.

### Local run
```bash
cd cyrex-interface
npm install
npm run dev -- --host 0.0.0.0 --port 5175
# visit http://localhost:5175 and plug in your x-api-key
```

### Docker Compose
The `cyrex-interface` service is already wired into `docker-compose.dev.yml`:

```bash
cd deepiri-platform
docker compose -f docker-compose.dev.yml up cyrex cyrex-interface
```

The dashboard exposes:
- Chat UX powered by `/agent/intelligence/generate-ability`
- Forms for `/agent/intelligence/route-command`, `/generate-ability`, `/recommend-action`, `/knowledge/query`
- Document indexing panel
- Vendor fraud detection dashboard
- Workflow playground for LangGraph multi-agent workflows

---

## Common Use Cases

1. **Agent Chat** — Talk to AI agents that remember context and use tools
2. **Task Decomposition** — Break big tasks into smaller, actionable steps
3. **Document Analysis** — Upload invoices/contracts and get structured data
4. **Fraud Detection** — Analyze vendor billing for suspicious patterns
5. **RAG Search** — Upload documents, then ask natural language questions
6. **AI Embeddings** — Generate vectors for semantic search
7. **Multi-Agent Workflows** — Orchestrate specialized agents in pipelines

## Next Steps

- Explore the full API at `http://localhost:8000/docs`
- Check `app/` for the source code
- See [docs/architecture/ARCHITECTURE.md](../architecture/ARCHITECTURE.md) for system architecture
- See [deepiri-platform/team_dev_environments/ai-team/README.md](https://github.com/Team-Deepiri/deepiri-platform/blob/dev/team_dev_environments/ai-team/README.md) for AI team development environment details
