# Data Storage Architecture

## 🎯 **CRITICAL: Universal Training Data Store**

**This is the UNIVERSAL place where all real-time data gets stored for training later.**

### Location
- **Service**: `app/core/training_data_store.py`
- **Storage**: CSV/JSONL files in `data/training/` directory
- **Real-time**: Synapse (Redis Streams) for streaming
- **Export**: JSONL format for ML training

### What Goes Here

✅ **Agent Events** → `training_data_store.store_agent_event()`
- All agent lifecycle events
- Event types, payloads, severity
- Used for training event prediction models

✅ **Agent Tasks** → `training_data_store.store_agent_task()`
- Task execution data
- Input/output, status, retry counts
- Used for training task completion models

✅ **Tool Executions** → `training_data_store.store_tool_execution()`
- Tool usage patterns
- Input parameters, outputs, execution times
- Used for training tool selection models

✅ **Workflow Data** → `training_data_store.store_workflow_data()`
- Workflow state transitions
- Step results, errors
- Used for training workflow orchestration models

### Data Flow

```
Real-time Event → Synapse (Redis Streams) → Training Data Store (CSV/JSONL)
                                              ↓
                                    Export for Training (JSONL)
```

---

## PostgreSQL: Only Essential Data

### What Stays in PostgreSQL

✅ **agent_playground_messages**
- Conversation history for agent playground
- Needs persistence for chat continuity
- Links to `instance_id` for conversation threads

### What Does NOT Go to PostgreSQL

❌ **Agent Configs** → In-memory or config files
❌ **Agent Instances** → In-memory only (ephemeral)
❌ **Agent Tasks** → Training Data Store
❌ **Tool Executions** → Training Data Store
❌ **Agent Events** → Training Data Store
❌ **Workflows** → Training Data Store
❌ **Prompt Templates** → Code/config files (not DB)
❌ **Metrics** → Monitoring system (InfluxDB/Prometheus)

---

## Storage Decision Tree

```
Is it conversation history?
├─ YES → PostgreSQL (agent_playground_messages)
└─ NO → Is it real-time training data?
    ├─ YES → Training Data Store (CSV/JSONL + Synapse)
    └─ NO → Is it configuration?
        ├─ YES → Config files or in-memory
        └─ NO → Is it metrics?
            ├─ YES → Monitoring system (InfluxDB/Prometheus)
            └─ NO → Evaluate case by case
```

---

## Usage Examples

### Store Agent Event
```python
from app.core.training_data_store import get_training_data_store

store = get_training_data_store()
await store.store_agent_event(
    event_type="agent_started",
    agent_id="agent-123",
    payload={"model": "llama3:8b"},
    severity="info"
)
```

### Store Tool Execution
```python
await store.store_tool_execution(
    execution_id="exec-456",
    agent_id="agent-123",
    tool_name="calculate",
    input_params={"expression": "2+2"},
    output_result={"result": 4},
    execution_time_ms=12.5
)
```

### Export for Training
```python
# Export all events from last week
export_path = store.export_for_training(
    data_type="events",
    start_date="2024-01-01",
    end_date="2024-01-07"
)
# Returns: data/training/events_training.jsonl
```

---

## File Structure

```
data/training/
├── events/
│   ├── events_2024-01-01.csv
│   ├── events_2024-01-02.csv
│   └── ...
├── tasks/
│   ├── tasks_2024-01-01.csv
│   └── ...
├── tool_executions/
│   ├── tool_executions_2024-01-01.csv
│   └── ...
└── workflows/
    ├── workflows_2024-01-01.csv
    └── ...
```

---

## Key Principles

1. **PostgreSQL = Persistence** (conversation history only)
2. **Training Data Store = Real-time training data** (events, tasks, tools, workflows)
3. **Synapse = Real-time streaming** (all training data also streams through Synapse)
4. **Config Files = Configuration** (prompts, agent configs)
5. **In-Memory = Ephemeral** (agent instances, runtime state)
6. **Monitoring = Metrics** (InfluxDB/Prometheus for observability)

---

## Migration Notes

If you see code storing to old PostgreSQL tables:
- `agent_events` → Use `training_data_store.store_agent_event()`
- `agent_tasks` → Use `training_data_store.store_agent_task()`
- `tool_executions` → Use `training_data_store.store_tool_execution()`
- `workflows` → Use `training_data_store.store_workflow_data()`
- `agent_configs` → Remove (use config files)
- `agent_instances` → Remove (in-memory only)
- `prompt_templates` → Remove (use code/config files)
- `agent_metrics` → Remove (use monitoring system)

