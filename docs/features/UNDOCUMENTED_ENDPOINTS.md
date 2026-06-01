# Additional API Endpoints

This document covers API endpoints that are not yet documented in the main guides. All endpoints require `x-api-key: change-me` header unless noted.

## Agent Playground

### `/agent/playground/*`

Interactive testing interface for multi-agent workflows.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/agent/playground/create-agent` | Create a playground agent |
| POST | `/agent/playground/invoke` | Invoke an agent with input |
| GET | `/agent/playground/agents` | List all agents |
| DELETE | `/agent/playground/agent/{id}` | Delete an agent |

## Company Automation

### `/agent/company-automation/*`

Automate business processes with AI agents.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/agent/company-automation/process` | Process a company automation request |
| POST | `/agent/company-automation/train-adapter` | Train a LoRA adapter |
| GET | `/agent/company-automation/adapters` | List available adapters |
| POST | `/agent/company-automation/register-tools` | Register custom tools |

## Collection Management

### `/agent/collections/*`

Manage vector store collections (Milvus).

| Method | Path | Description |
|--------|------|-------------|
| POST | `/agent/collections/create` | Create a new collection |
| GET | `/agent/collections/list` | List all collections |
| DELETE | `/agent/collections/{name}` | Delete a collection |

## Language Intelligence

### `/agent/language-intelligence/*`

NLP and language processing capabilities.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/agent/language-intelligence/analyze` | Analyze text for language features |
| POST | `/agent/language-intelligence/similarity` | Compare text similarity |
| GET | `/agent/language-intelligence/languages` | List supported languages |

## Document Extraction

### `/agent/documents/extract/*`

Extract structured data from documents.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/agent/documents/extract/text` | Extract from raw text |
| POST | `/agent/documents/extract/file` | Extract from uploaded file |
| POST | `/agent/documents/extract/url` | Extract from URL |

## Testing API

### `/agent/testing/*`

Testing utilities and mock endpoints.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/agent/testing/mock` | Run a mock test |
| POST | `/agent/testing/benchmark` | Run benchmarks |
| GET | `/agent/testing/results` | Get test results |

## Workflow API

### `/agent/workflow/*`

LangGraph multi-agent workflow management.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/agent/workflow/execute` | Execute a workflow |
| GET | `/agent/workflow/{id}/status` | Get workflow status |
| GET | `/agent/workflow/{id}/result` | Get workflow result |
| DELETE | `/agent/workflow/{id}` | Cancel a workflow |

## Cyrex Guard

### `/agent/guard/*`

Universal vendor fraud detection platform for 6 industries.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/agent/guard/analyze` | Analyze a vendor for fraud |
| GET | `/agent/guard/report/{id}` | Get a guard report |
| POST | `/agent/guard/configure` | Configure guard rules |

## Documents

### `/agent/documents/*`

Document management endpoints.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/agent/documents/list` | List all documents |
| GET | `/agent/documents/{id}` | Get document details |
| DELETE | `/agent/documents/{id}` | Delete a document |
| POST | `/agent/documents/upload` | Upload a document |
| POST | `/agent/documents/analyze` | Analyze a document |

## Agent Management

### `/agent/*`

Core agent endpoints.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/agent/list` | List all agents |
| POST | `/agent/create` | Create a new agent |
| GET | `/agent/{id}` | Get agent details |
| POST | `/agent/{id}/invoke` | Invoke an agent |
| POST | `/agent/{id}/stop` | Stop an agent |

## Challenge

### `/agent/challenge/*`

Challenge generation and management.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/agent/challenge/generate` | Generate a challenge |
| GET | `/agent/challenge/{id}` | Get challenge details |
| POST | `/agent/challenge/submit` | Submit challenge answer |

## Session

### `/agent/session/*`

Session management.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/agent/sessions` | List sessions |
| GET | `/agent/sessions/{id}` | Get session details |
| DELETE | `/agent/sessions/{id}` | Delete a session |

## Monitoring

### `/agent/monitoring/*`

System monitoring and metrics.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/agent/monitoring/health` | System health |
| GET | `/agent/monitoring/stats` | System statistics |
| GET | `/agent/monitoring/metrics` | Prometheus metrics |

---

## Base URL

All endpoints are relative to:
```
http://localhost:8000
```

## Authentication

All endpoints (except `/health`, `/metrics`, `/docs`) require:
```
x-api-key: change-me
```

## Swagger UI

For interactive testing of all endpoints, visit:
```
http://localhost:8000/docs
```
