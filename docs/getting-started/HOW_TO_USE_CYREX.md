# How to Test and Use Cyrex AI Service

Cyrex is the AI/ML service that provides natural language processing, embeddings, challenge generation, and model inference capabilities.

##  Deepiri AI System (Three-Tier Architecture)

Cyrex now includes the complete Deepiri AI system with three tiers:

1. **Intent Classification** (Tier 1): BERT/DeBERTa-based intent classification
2. **Ability Generation** (Tier 2): LLM + RAG for dynamic ability creation
3. **Productivity Agent** (Tier 3): RL-based adaptive learning

See `DEEPIRI_AI_SYSTEM.md` for complete documentation.

##  Quick Start

### 1. Start Cyrex Service

**Option A: Using Docker Compose (Recommended)**
```bash
# Start Cyrex with all dependencies
cd deepiri
docker compose -f docker-compose.dev.yml up -d cyrex mongodb influxdb milvus etcd minio

# Or use the AI team script
cd team_dev_environments/ai-team
./start.sh
```

**Option B: Using Team-Specific Compose**
```bash
docker compose -f docker-compose.ai-team.yml up -d
```

### 2. Verify Cyrex is Running

```bash
# Check health
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","service":"cyrex","version":"1.0.0"}

# Check service status
docker compose -f docker-compose.dev.yml ps cyrex

# View logs
docker compose -f docker-compose.dev.yml logs -f cyrex
```

### 3. Access Interactive API Documentation

Open your browser and navigate to:
```
http://localhost:8000/docs
```

This provides a **Swagger UI** where you can:
- See all available endpoints
- Test endpoints directly in the browser
- View request/response schemas
- Try out different API calls

##  API Key Authentication

**IMPORTANT**: Most Cyrex endpoints require an API key. The default key is `change-me` (or set via `CYREX_API_KEY` environment variable).

**Endpoints that DON'T require API key:**
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /docs` - Interactive API documentation

**All other endpoints require the API key header:**
```bash
-H "x-api-key: change-me"
```

### Finding Your API Key

```bash
# Check the API key in the running container
docker compose -f docker-compose.dev.yml exec cyrex env | grep CYREX_API_KEY

# Or check your .env file
cat .env | grep CYREX_API_KEY
```

**Default API Key**: `change-me` (if not set in environment)

##  Deepiri AI Endpoints

### Command Routing (Tier 1)
```bash
# Route user command to predefined ability
curl -X POST http://localhost:8000/agent/intelligence/route-command \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "command": "Can you review this code for security issues?",
    "user_role": "software_engineer",
    "min_confidence": 0.7
  }'
```

### Contextual Ability Generation (Tier 2)
```bash
# Generate dynamic ability using LLM + RAG
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
    }
  }'
```

### Workflow Optimization (Tier 3)
```bash
# Get RL optimizer recommendation
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
```bash
# Query knowledge bases
curl -X POST http://localhost:8000/agent/intelligence/knowledge/query \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are effective focus boost strategies?",
    "knowledge_bases": ["user_patterns", "ability_templates"],
    "top_k": 5
  }'
```

##  API Endpoints

### Health & Info (No API Key Required)

```bash
# Health check - NO API KEY NEEDED
curl http://localhost:8000/health

# Service information - REQUIRES API KEY
curl -H "x-api-key: change-me" http://localhost:8000/api/info

# Prometheus metrics - NO API KEY NEEDED
curl http://localhost:8000/metrics
```

### Text Embeddings (Requires API Key)

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

**Response:**
```json
{
  "embedding": [0.123, -0.456, ...],
  "dimension": 384,
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

### AI Completions (Requires API Key)

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

**Response:**
```json
{
  "completion": "Artificial intelligence (AI) is...",
  "tokens_used": 45,
  "model": "gpt-4o-mini"
}
```

### Challenge Generation (Requires API Key)

Generate AI-powered challenges:

```bash
curl -X POST http://localhost:8000/api/agent/challenge/generate \
  -H "Content-Type: application/json" \
  -H "x-api-key: change-me" \
  -d '{
    "task": {
      "title": "Build a REST API",
      "description": "Create a RESTful API for a todo application"
    },
    "difficulty": "medium",
    "type": "coding"
  }'
```

### Task Processing (Requires API Key)

Process and understand tasks:

```bash
curl -X POST http://localhost:8000/api/agent/task/process \
  -H "Content-Type: application/json" \
  -H "x-api-key: change-me" \
  -d '{
    "task_description": "Implement user authentication",
    "context": "Node.js backend application"
  }'
```

##  Testing Examples

### Example 1: Simple Health Check

```bash
# Windows PowerShell
Invoke-WebRequest -Uri http://localhost:8000/health -Method GET

# Linux/Mac
curl http://localhost:8000/health
```

### Example 2: Generate Embedding (With API Key)

```bash
# Windows PowerShell
$headers = @{
    "Content-Type" = "application/json"
    "x-api-key" = "change-me"
}
$body = @{
    text = "Deepiri is an AI-powered learning platform"
} | ConvertTo-Json

Invoke-WebRequest -Uri http://localhost:8000/api/embeddings `
  -Method POST `
  -Headers $headers `
  -Body $body

# Linux/Mac
curl -X POST http://localhost:8000/api/embeddings \
  -H "Content-Type: application/json" \
  -H "x-api-key: change-me" \
  -d '{"text":"Deepiri is an AI-powered learning platform"}'
```

### Example 3: AI Completion (With API Key)

```bash
# Windows PowerShell
$headers = @{
    "Content-Type" = "application/json"
    "x-api-key" = "change-me"
}
$body = @{
    prompt = "Explain machine learning in simple terms"
    max_tokens = 150
} | ConvertTo-Json

Invoke-WebRequest -Uri http://localhost:8000/api/complete `
  -Method POST `
  -Headers $headers `
  -Body $body

# Linux/Mac
curl -X POST http://localhost:8000/api/complete \
  -H "Content-Type: application/json" \
  -H "x-api-key: change-me" \
  -d '{
    "prompt": "Explain machine learning in simple terms",
    "max_tokens": 150
  }'
```

##  Configuration

### Environment Variables

Make sure these are set in your `.env` file or Docker environment:

```bash
# Required for AI features
OPENAI_API_KEY=your-openai-api-key-here

# API Key for Cyrex endpoints (default: "change-me")
CYREX_API_KEY=change-me

# Optional
OPENAI_MODEL=gpt-4o-mini
MILVUS_HOST=milvus
MILVUS_PORT=19530
INFLUXDB_URL=http://influxdb:8086
LOG_LEVEL=INFO
```

### Setting a Custom API Key

You can set a custom API key in your `.env` file:

```bash
CYREX_API_KEY=my-secret-api-key-123
```

Then use it in your requests:
```bash
curl -H "x-api-key: my-secret-api-key-123" http://localhost:8000/api/info
```

### Check Environment Variables

```bash
# View Cyrex environment
docker compose -f docker-compose.dev.yml exec cyrex env | grep -E "OPENAI|CYREX|MILVUS"
```

##  Troubleshooting

### Cyrex Won't Start

```bash
# Check logs
docker compose -f docker-compose.dev.yml logs cyrex

# Check if port 8000 is in use
# Windows
netstat -ano | findstr :8000
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
curl -H "x-api-key: change-me" http://localhost:8000/api/info

# Or check what API key is configured
docker compose -f docker-compose.dev.yml exec cyrex env | grep CYREX_API_KEY
```

**Note:** `/health` and `/metrics` endpoints don't require API keys.

### API Returns Other Errors

1. **Check OpenAI API Key:**
   ```bash
   docker compose -f docker-compose.dev.yml exec cyrex env | grep OPENAI_API_KEY
   ```

2. **Check Dependencies:**
   ```bash
   # Ensure MongoDB, InfluxDB, and Milvus are running
   docker compose -f docker-compose.dev.yml ps mongodb influxdb milvus
   ```

3. **Check Service Logs:**
   ```bash
   docker compose -f docker-compose.dev.yml logs -f cyrex
   ```

### Milvus Connection Issues

If you see Milvus connection errors:

```bash
# Ensure Milvus and dependencies are running
docker compose -f docker-compose.dev.yml up -d etcd minio milvus

# Check Milvus health
curl http://localhost:19530/healthz

# Check Cyrex can reach Milvus
docker compose -f docker-compose.dev.yml exec cyrex ping milvus
```

##  Using the Interactive API Docs

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
   - Click on `POST /api/embeddings`
   - Click "Try it out"
   - **IMPORTANT**: Click "Authorize" button at the top
   - Enter API key: `change-me`
   - Click "Authorize" then "Close"
   - Edit the JSON request body
   - Click "Execute"
   - View the response

**Note:** The Swagger UI has an "Authorize" button at the top where you can enter your API key once, and it will be used for all requests.

##  Integration Examples

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

### From Backend (Node.js)

```javascript
const axios = require('axios');

// Health check (no API key needed)
const health = await axios.get('http://cyrex:8000/health');
console.log(health.data);

// Generate completion (API key required)
const completion = await axios.post('http://cyrex:8000/api/complete', {
  prompt: 'What is AI?',
  max_tokens: 100
}, {
  headers: {
    'x-api-key': 'change-me'  // Required!
  }
});
console.log(completion.data);
```

### From Python

```python
import requests

# Health check (no API key needed)
response = requests.get('http://localhost:8000/health')
print(response.json())

# Generate embedding (API key required)
response = requests.post('http://localhost:8000/api/embeddings', 
  json={'text': 'Hello world'},
  headers={'x-api-key': 'change-me'})  # Required!
print(response.json())
```

## Cyrex Interface

Prefer a UI over cURL? Launch the Cyrex Interface that lives in `diri-cyrex/cyrex-interface`.

### Local run
```bash
cd diri-cyrex/cyrex-interface
npm install
npm run dev -- --host 0.0.0.0 --port 5175
# visit http://localhost:5175 and plug in your x-api-key
```

### Docker Compose
Use the `cyrex-interface` service that is already wired into `docker-compose.dev.yml`:

```bash
cd deepiri
docker compose -f docker-compose.dev.yml up cyrex cyrex-interface
```

The dashboard exposes:
- Chat UX powered by `/agent/intelligence/generate-ability`
- Forms for `/route-command`, `/generate-ability`, `/recommend-action`, `/knowledge/query`
- Inline reminders for `pytest`, `mypy`, and `npm run lint`

##  Common Use Cases

1. **Challenge Generation**: Generate coding challenges based on task descriptions
2. **Text Embeddings**: Create vector embeddings for semantic search
3. **Task Understanding**: Process and understand user tasks
4. **RAG (Retrieval Augmented Generation)**: Generate content using retrieved context
5. **Model Inference**: Run ML model predictions

##  Next Steps

- Explore the full API at `http://localhost:8000/docs`
- Check `diri-cyrex/app/` for the source code
- Review `diri-cyrex/RUN_WITHOUT_DOCKER.md` for local development
- See `docs/SERVICES_OVERVIEW.md` for more service details

##  Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics
- **Service Logs**: `docker compose -f docker-compose.dev.yml logs -f cyrex`

