# Company Automation System

## Complete Implementation

### LoRA/QLoRA Integration System

**Files Created:**
1. `integrations/lora_adapter_service.py` - Complete LoRA adapter management
2. `integrations/company_data_automation.py` - Company data automation service
3. `routes/company_automation_api.py` - REST API endpoints

**Integration Points:**
- Synapse Broker (Redis Streams)
- ModelKit (Registry & Streaming)
- PostgreSQL (Persistence)
- Agents (Inference with adapters)
- API Bridge (Tool automation)

## System Architecture

```
Company Data
    v
Company Data Automation Service
    v
Company Agent (with LoRA Adapter)
    v
Process Data -> Call Tools -> Return Results
    v
Store in Memory & Database
```

## Adapter Training Flow

```
Cyrex: Request Training
    v
Synapse: lora_training_requests channel
    v
ModelKit: Streaming to Helox
    v
Helox: Train LoRA/QLoRA Adapter
    v
Helox: Publish lora_adapter_ready
    v
Cyrex: Download & Load Adapter
    v
Ready for Company Automation
```

## Key Features

### 1. LoRA Adapter Service
- Request adapter training via Synapse
- Receive adapter ready notifications
- Load adapters for company-specific inference
- Cache adapters locally
- Integrate with ModelKit registry

### 2. Company Data Automation
- Process company data with agents
- Use company-specific LoRA adapters
- Execute automation tools
- Track automation jobs
- Store company context in memory

### 3. Tool Automation
- Register company-specific tools
- Agents automatically call tools
- Tool results integrated into responses
- Full tool execution tracking

## API Endpoints

### Company Automation
- `POST /company-automation/process` - Process company data
- `POST /company-automation/train-adapter` - Request adapter training
- `GET /company-automation/adapters` - List adapters
- `POST /company-automation/register-tools` - Register company tools

## Database Tables

### `lora_adapters`
- Adapter metadata and paths
- Company associations
- Status tracking

### `company_automation_jobs`
- Job tracking
- Input/output data
- Status and results

### `company_tools`
- Registered company tools
- Tool configurations
- Company associations

## Synapse Channels

- `lora_training_requests` - Training requests to Helox
- `lora_adapter_ready` - Adapter ready from Helox
- `lora_adapter_requests` - Adapter loading requests
- `company_automation_jobs` - Job updates

## ModelKit Integration

- **Registry**: Store trained adapters
- **Streaming**: Receive adapter events
- **Download**: Automatic adapter retrieval

## All Systems Aligned for Company Data Automation

- **Agents** - Process company data with LoRA
- **Tools** - Automate company APIs
- **Memory** - Store company context
- **Sessions** - Track company sessions
- **API Bridge** - Connect company APIs
- **Synapse** - Coordinate adapter training
- **ModelKit** - Manage adapter lifecycle
- **Database** - Persist company data

## Ready to Use

### Request Adapter Training
```python
lora_service = await get_lora_service()
request_id = await lora_service.request_adapter_training(
    company_id="acme_corp",
    training_data=[...],
    use_qlora=True
)
```

### Process Company Data
```python
automation = await get_automation_service()
result = await automation.process_company_data(
    company_id="acme_corp",
    data={"task": "analyze_data", "data": [...]},
    use_adapter=True
)
```

### Register Company Tools
```python
await automation.register_company_tools(
    company_id="acme_corp",
    tools=[{
        "name": "crm_api",
        "endpoint": "https://crm.acme.com/api",
        "method": "POST",
        "auth": {"type": "bearer", "token": "..."}
    }]
)
```

## Package Dependencies

All packages verified in `requirements.txt`:
- `peft>=0.7.0` - LoRA/QLoRA
- `bitsandbytes>=0.41.0` - Quantization
- `asyncpg>=0.29.0` - PostgreSQL
- `httpx>=0.27.2` - API calls
- `transformers>=4.35.0` - Model loading
- `torch>=2.0.0` - PyTorch

## Integration Status

- **LoRA/QLoRA Integration** - Complete
- **Synapse Broker** - Complete
- **ModelKit** - Complete
- **Redis** - Complete
- **PostgreSQL** - Complete
- **Agent Integration** - Complete
- **Tool Automation** - Complete
- **Company Data Processing** - Complete

## Next Steps for ML Team (Helox)

1. Subscribe to `lora_training_requests` channel
2. Receive training requests with company data
3. Use existing LoRA training pipeline
4. Train QLoRA adapters
5. Publish `lora_adapter_ready` events
6. Store in ModelKit registry

## System Status

**Goal**: Take in company data and automate tools
**Status**: Fully implemented and ready

All components are aligned and ready for production use.

