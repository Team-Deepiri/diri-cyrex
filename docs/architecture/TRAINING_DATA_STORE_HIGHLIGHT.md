# 🎯 **UNIVERSAL TRAINING DATA STORE - HIGHLIGHTED**

## ⚠️ **THIS IS THE UNIVERSAL PLACE FOR ALL REAL-TIME TRAINING DATA**

**Location**: `app/core/training_data_store.py`

**Purpose**: Store ALL real-time data that will be used for training models later.

---

## 📍 What Goes Here

### ✅ Agent Events
- All agent lifecycle events
- Event types, payloads, severity levels
- **Usage**: Train event prediction models

### ✅ Agent Tasks  
- Task execution data
- Input/output, status, retry counts
- **Usage**: Train task completion models

### ✅ Tool Executions
- Tool usage patterns
- Input parameters, outputs, execution times
- **Usage**: Train tool selection models

### ✅ Workflow Data
- Workflow state transitions
- Step results, errors
- **Usage**: Train workflow orchestration models

---

## 🔄 Data Flow

```
Real-time Event
    ↓
Synapse (Redis Streams) ← Real-time streaming
    ↓
Training Data Store (CSV/JSONL) ← Persistent storage
    ↓
Export for Training (JSONL) ← ML training format
```

---

## 💾 Storage Format

- **Real-time**: Synapse (Redis Streams) for streaming
- **Persistent**: CSV files organized by date (`events_2024-01-01.csv`)
- **Export**: JSONL format for ML training

---

## 📂 File Structure

```
data/training/
├── events/
│   ├── events_2024-01-01.csv
│   └── events_2024-01-02.csv
├── tasks/
│   └── tasks_2024-01-01.csv
├── tool_executions/
│   └── tool_executions_2024-01-01.csv
└── workflows/
    └── workflows_2024-01-01.csv
```

---

## 🚀 Usage

```python
from app.core.training_data_store import get_training_data_store

store = get_training_data_store()

# Store agent event
await store.store_agent_event(
    event_type="agent_started",
    agent_id="agent-123",
    payload={"model": "llama3:8b"}
)

# Store tool execution
await store.store_tool_execution(
    execution_id="exec-456",
    tool_name="calculate",
    input_params={"expression": "2+2"},
    output_result={"result": 4}
)

# Export for training
export_path = store.export_for_training("events")
# Returns: data/training/events_training.jsonl
```

---

## ⚡ Key Features

1. **Real-time Streaming**: All data streams through Synapse (Redis Streams)
2. **Persistent Storage**: CSV files organized by date
3. **Training Ready**: Export to JSONL format for ML training
4. **Automatic Organization**: Files organized by data type and date
5. **No PostgreSQL**: This is NOT in PostgreSQL - it's file-based for training

---

## 🎯 **THIS IS WHERE ALL REAL-TIME TRAINING DATA GOES**

**Remember**: 
- ✅ Events → Training Data Store
- ✅ Tasks → Training Data Store  
- ✅ Tool Executions → Training Data Store
- ✅ Workflows → Training Data Store
- ❌ NOT PostgreSQL
- ❌ NOT in-memory only

