# LoRA/QLoRA System Integration

## LoRA/QLoRA Integration - Complete

### Files Created

1. **`integrations/lora_adapter_service.py`**
   - LoRA/QLoRA adapter management
   - Synapse broker integration
   - ModelKit integration
   - Redis streaming support
   - Adapter training requests
   - Adapter loading and caching

2. **`integrations/company_data_automation.py`**
   - Main automation service
   - Company data processing
   - Tool automation
   - Agent integration with LoRA
   - Job tracking

3. **`routes/company_automation_api.py`**
   - REST API endpoints
   - Process company data
   - Request adapter training
   - List adapters
   - Register tools

## Integration Points

### Synapse Broker
- **Channels**:
  - `lora_training_requests` - Training requests to Helox
  - `lora_adapter_ready` - Adapter ready notifications
  - `lora_adapter_requests` - Adapter loading requests
- **Integration**: Full pub/sub support

### ModelKit
- **Registry**: Adapter storage and retrieval
- **Streaming**: Event subscription for adapter ready
- **Download**: Automatic adapter download from registry

### Redis
- **State**: Session and adapter state
- **Caching**: Adapter caching
- **Queues**: Message queuing

### PostgreSQL
- **Tables**:
  - `lora_adapters` - Adapter metadata
  - `company_automation_jobs` - Job tracking
  - `company_tools` - Tool registrations

## System Alignment for Company Data Automation

### All Systems Aligned

1. **Agents** -> Process company data with LoRA adapters
2. **Tools** -> Automate company APIs and workflows
3. **Memory** -> Store company-specific context
4. **Sessions** -> Track company sessions
5. **API Bridge** -> Connect to company APIs
6. **Synapse** -> Coordinate adapter training
7. **ModelKit** -> Manage adapter lifecycle
8. **Database** -> Persist company data and jobs

## Usage Flow

### 1. Onboard Company
```python
automation = await get_automation_service()

# Register company tools
await automation.register_company_tools(
    company_id="acme_corp",
    tools=[{
        "name": "crm_lookup",
        "endpoint": "https://crm.acme.com/api",
        "method": "POST",
        "auth": {"type": "bearer", "token": "..."},
    }]
)

# Request adapter training
await automation.train_company_adapter(
    company_id="acme_corp",
    training_data=[...]
)
```

### 2. Process Company Data
```python
result = await automation.process_company_data(
    company_id="acme_corp",
    data={
        "task": "analyze_customer_feedback",
        "data": ["Great product!", "Needs improvement"],
    },
    use_adapter=True
)
```

### 3. Agent Automates
- Agent uses company LoRA adapter
- Processes company data
- Calls company tools automatically
- Returns automation results

## Ready to Send/Receive

### Sending Adapter Training Requests
- Via Synapse broker
- Via ModelKit streaming
- Via REST API

### Receiving Adapter Ready Notifications
- Via Synapse broker subscription
- Via ModelKit streaming
- Automatic download and loading

## Package Dependencies

All required packages in `requirements.txt`:
- `asyncpg` - PostgreSQL
- `httpx` - API calls
- `peft` - LoRA/QLoRA
- `bitsandbytes` - Quantization
- `transformers` - Model loading
- `torch` - PyTorch

## Next Steps for ML Team (Helox)

The ML team can now:
1. Subscribe to `lora_training_requests` channel
2. Receive training requests with company data
3. Train LoRA/QLoRA adapters using existing pipeline
4. Publish `lora_adapter_ready` events
5. Store adapters in ModelKit registry

## System Status

- LoRA/QLoRA integration complete
- Synapse broker integration complete
- ModelKit integration complete
- Redis integration complete
- Company data automation ready
- Tool automation ready
- All systems aligned with goal

