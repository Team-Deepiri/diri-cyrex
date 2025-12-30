# Company Data Automation - System Alignment

## Overall Goal

**Take in company data and automate tools** - All systems aligned to support this goal.

## System Components Aligned

### 1. LoRA/QLoRA Integration
**Purpose**: Company-specific model fine-tuning
- **File**: `integrations/lora_adapter_service.py`
- **Alignment**: Enables company-specific adapters for better automation
- **Integration**: Synapse broker, ModelKit, Redis
- **Use Case**: Train adapters on company data for personalized automation

### 2. Company Data Automation Service
**Purpose**: Main service for processing company data
- **File**: `integrations/company_data_automation.py`
- **Alignment**: Directly processes company data and automates tools
- **Features**:
  - Process company data with agents
  - Use company-specific LoRA adapters
  - Execute automation tools
  - Track automation jobs

### 3. Agent System
**Purpose**: AI agents for automation
- **Files**: `agents/base_agent.py`, `agents/agent_factory.py`
- **Alignment**: Agents process company data and call tools
- **Enhancement**: Company-specific agents with LoRA adapters
- **Tool Integration**: Full tool support for automation

### 4. API Bridge
**Purpose**: External API integration
- **File**: `integrations/api_bridge.py`
- **Alignment**: Enables tool automation via external APIs
- **Enhancement**: Company-specific tool registration
- **Use Case**: Connect to company APIs for automation

### 5. Synapse Broker
**Purpose**: Message broker for adapter training
- **File**: `integrations/synapse_broker.py`
- **Alignment**: Handles adapter training requests and notifications
- **Channels**: `lora_training_requests`, `lora_adapter_ready`
- **Use Case**: Coordinate between Cyrex and Helox for adapter training

### 6. Memory System
**Purpose**: Context and memory management
- **File**: `core/memory_manager.py`
- **Alignment**: Stores company-specific context and automation history
- **Enhancement**: Company-scoped memories
- **Use Case**: Remember company preferences and past automations

### 7. Session Management
**Purpose**: Session tracking
- **File**: `core/session_manager.py`
- **Alignment**: Tracks company sessions for automation
- **Enhancement**: Company-specific sessions
- **Use Case**: Maintain context across automation jobs

### 8. Tool Registry
**Purpose**: Tool management
- **File**: `core/tool_registry.py`
- **Alignment**: Manages automation tools
- **Enhancement**: Company-specific tool registration
- **Use Case**: Register and execute company tools

## Data Flow for Company Automation

```
Company Data Input
    ↓
Company Data Automation Service
    ↓
Get/Create Company Agent (with LoRA adapter)
    ↓
Agent Processes Data
    ↓
Agent Calls Tools (via API Bridge)
    ↓
Tools Execute (company APIs)
    ↓
Results Stored (Memory + Database)
    ↓
Return Automation Results
```

## Adapter Training Flow

```
Company Data + Training Request
    ↓
LoRA Adapter Service
    ↓
Publish to Synapse (lora_training_requests)
    ↓
Helox Receives & Trains
    ↓
Helox Publishes (lora_adapter_ready)
    ↓
Cyrex Downloads & Loads Adapter
    ↓
Adapter Ready for Company Automation
```

## API Endpoints

### Company Automation
- `POST /company-automation/process` - Process company data
- `POST /company-automation/train-adapter` - Request adapter training
- `GET /company-automation/adapters` - List adapters
- `POST /company-automation/register-tools` - Register company tools

## Database Schema

### Company Automation Tables
- `company_automation_jobs` - Tracks automation jobs
- `company_tools` - Registered company tools
- `lora_adapters` - LoRA adapter metadata

## Key Features for Company Automation

1. **Company-Specific Adapters**: Each company gets personalized LoRA adapter
2. **Tool Automation**: Agents automatically call company tools
3. **Data Processing**: Structured processing of company data
4. **Job Tracking**: Full tracking of automation jobs
5. **Memory**: Company-specific context and history
6. **Scalability**: Multi-company support

## Integration Points

### Synapse Channels
- `lora_training_requests` - Training requests
- `lora_adapter_ready` - Adapter ready notifications
- `company_automation_jobs` - Job updates

### ModelKit
- Registry for adapter storage
- Streaming for adapter events
- Download for adapter retrieval

### Redis
- Session state
- Message queuing
- Caching

## Example Workflow

1. **Company Onboarding**
   ```python
   # Register company tools
   await automation.register_company_tools(
       company_id="acme_corp",
       tools=[...]
   )
   
   # Request adapter training
   await automation.train_company_adapter(
       company_id="acme_corp",
       training_data=[...]
   )
   ```

2. **Process Company Data**
   ```python
   result = await automation.process_company_data(
       company_id="acme_corp",
       data={"task": "process_invoices", "invoices": [...]},
       use_adapter=True
   )
   ```

3. **Agent Automates Tools**
   - Agent uses company LoRA adapter
   - Processes company data
   - Calls company tools automatically
   - Returns results

## All Files Aligned

All core systems support company data automation
All integrations support tool automation
All agents support company-specific processing
All tools support company APIs
All memory/session systems support company context
All messaging supports adapter training workflow

