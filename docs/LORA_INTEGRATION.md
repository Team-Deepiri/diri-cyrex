# LoRA/QLoRA Integration Guide

## Overview

Complete LoRA/QLoRA adapter integration system for company data automation. Integrates with Synapse broker, Redis, and ModelKit for seamless adapter training and deployment.

## Architecture

```
Company Data → Automation Service → Agent (with LoRA) → Tools → Results
                    ↓
            LoRA Adapter Service
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
    Synapse Broker          ModelKit
    (Redis Streams)         (Registry)
        ↓                       ↓
    Helox Training      Model Storage
```

## Components

### 1. LoRA Adapter Service (`integrations/lora_adapter_service.py`)

**Purpose**: Manages LoRA/QLoRA adapters for company-specific fine-tuning

**Features**:
- Request adapter training via Synapse
- Receive adapter ready notifications
- Load adapters for inference
- Integrate with ModelKit registry
- Cache adapters locally

**Key Methods**:
- `request_adapter_training()` - Request training from Helox
- `load_adapter()` - Load adapter for company
- `get_adapter_for_company()` - Get loaded adapter
- `list_adapters()` - List all adapters

### 2. Company Data Automation (`integrations/company_data_automation.py`)

**Purpose**: Main service for processing company data and automating tools

**Features**:
- Process company data with agents
- Use company-specific LoRA adapters
- Execute automation tools
- Track automation jobs
- Store results in memory

**Key Methods**:
- `process_company_data()` - Process data with automation
- `train_company_adapter()` - Request adapter training
- `register_company_tools()` - Register company-specific tools

### 3. API Routes (`routes/company_automation_api.py`)

**Endpoints**:
- `POST /company-automation/process` - Process company data
- `POST /company-automation/train-adapter` - Request adapter training
- `GET /company-automation/adapters` - List adapters
- `POST /company-automation/register-tools` - Register tools

## Integration Flow

### Training Flow (Cyrex → Helox)

1. **Request Training**
   ```python
   lora_service = await get_lora_service()
   request_id = await lora_service.request_adapter_training(
       company_id="company123",
       training_data=[...],
       use_qlora=True
   )
   ```

2. **Publish to Synapse**
   - Message published to `lora_training_requests` channel
   - Forwarded to ModelKit streaming
   - Helox receives and processes

3. **Helox Trains Adapter**
   - Uses existing LoRA training pipeline
   - Trains with company data
   - Saves to registry

4. **Adapter Ready Notification**
   - Helox publishes `lora_adapter_ready` event
   - Cyrex receives via Synapse
   - Downloads and caches adapter
   - Loads for immediate use

### Inference Flow (Company Data → Automation)

1. **Receive Company Data**
   ```python
   automation = await get_automation_service()
   result = await automation.process_company_data(
       company_id="company123",
       data={"task": "analyze sales data"},
       use_adapter=True
   )
   ```

2. **Get Company Agent**
   - Creates/gets agent for company
   - Loads company-specific LoRA adapter
   - Configures with company context

3. **Process with Agent**
   - Agent uses LoRA adapter for inference
   - Processes company data
   - Calls automation tools

4. **Execute Tools**
   - Agent makes tool calls
   - Tools execute via API bridge
   - Results returned

## Synapse Channels

- `lora_training_requests` - Training requests from Cyrex
- `lora_adapter_ready` - Adapter ready notifications from Helox
- `lora_adapter_requests` - Adapter loading requests
- `company_automation_jobs` - Automation job updates

## ModelKit Integration

- **Registry**: Stores trained adapters
- **Streaming**: Receives adapter ready events
- **Download**: Downloads adapters from registry

## Database Tables

### `lora_adapters`
- Tracks all adapters
- Stores adapter paths and metadata
- Links to companies

### `company_automation_jobs`
- Tracks automation jobs
- Stores input/output data
- Links to adapters and agents

## Usage Examples

### Request Adapter Training

```python
from app.integrations.lora_adapter_service import get_lora_service

lora_service = await get_lora_service()
request_id = await lora_service.request_adapter_training(
    company_id="acme_corp",
    training_data=[
        {"input": "Analyze Q4 sales", "output": "Sales increased 15%"},
        {"input": "Process invoices", "output": "Processed 500 invoices"},
    ],
    config={
        "lora_rank": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
    },
    use_qlora=True
)
```

### Process Company Data

```python
from app.integrations.company_data_automation import get_automation_service

automation = await get_automation_service()
result = await automation.process_company_data(
    company_id="acme_corp",
    data={
        "task": "analyze_customer_feedback",
        "data": ["Great product!", "Needs improvement"],
    },
    task_type="sentiment_analysis",
    use_adapter=True
)
```

### Register Company Tools

```python
await automation.register_company_tools(
    company_id="acme_corp",
    tools=[
        {
            "name": "crm_lookup",
            "endpoint": "https://crm.acme.com/api/lookup",
            "method": "POST",
            "auth": {"type": "bearer", "token": "..."},
            "description": "Lookup customer in CRM",
        },
    ]
)
```

## Configuration

### Environment Variables

```env
# Base Model
BASE_MODEL_PATH=mistralai/Mistral-7B-v0.1

# Adapter Cache
ADAPTER_CACHE_DIR=/app/adapters

# ModelKit
MODEL_REGISTRY_TYPE=mlflow
MLFLOW_TRACKING_URI=http://mlflow:5000
S3_ENDPOINT_URL=http://minio:9000
S3_BUCKET=mlflow-artifacts

# Redis/Synapse
REDIS_URL=redis://redis:6379
```

## Alignment with Company Data Automation Goal

All components are designed for:
1. **Company Data Processing**: Direct support for company-specific data
2. **Automation**: Tool execution and workflow automation
3. **Personalization**: Company-specific adapters
4. **Scalability**: Multi-company support
5. **Integration**: Seamless tool and API integration

