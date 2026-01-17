# Training Data Store Implementation

## ✅ Implementation Complete

The training data store has been migrated from CSV/JSONL files to **PostgreSQL with `training_data` schema**.

---

## 📊 Data Storage Architecture

### PostgreSQL Schema Organization

```
PostgreSQL Database
├── public schema
│   └── agent_playground_messages (conversation history)
└── training_data schema  ← NEW
    ├── training_events
    ├── training_tasks
    ├── training_tool_executions
    └── training_workflows
```

### Data Type → Storage Location Mapping

| Data Type | Storage Location | Notes |
|-----------|----------------|-------|
| **Conversation Messages** | PostgreSQL (`public.agent_playground_messages`) | Needs persistence for chat history |
| **Agent Events** | PostgreSQL (`training_data.training_events`) + Synapse | Real-time streaming + queryable storage |
| **Agent Tasks** | PostgreSQL (`training_data.training_tasks`) + Synapse | Real-time streaming + queryable storage |
| **Tool Executions** | PostgreSQL (`training_data.training_tool_executions`) + Synapse | Real-time streaming + queryable storage |
| **Workflows** | PostgreSQL (`training_data.training_workflows`) + Synapse | Real-time streaming + queryable storage |
| **Agent Configs** | Config files (not DB) | In-memory or config files |
| **Agent Instances** | In-memory only | Ephemeral, no persistence needed |
| **Prompt Templates** | Code/config files | Not in database |
| **Metrics** | Monitoring system (InfluxDB/Prometheus) | Not in PostgreSQL |

---

## 🔄 Data Flow

```
Real-time Event
    ↓
Synapse (Redis Streams) ← Real-time streaming
    ↓
PostgreSQL (training_data schema) ← Queryable storage
    ↓
Export to CSV/JSONL ← For ML training pipelines
```

---

## 📁 Files Modified

### 1. `app/database/agent_tables.py`
- ✅ Added `create_training_data_tables()` function
- ✅ Creates `training_data` schema
- ✅ Creates 4 tables: `training_events`, `training_tasks`, `training_tool_executions`, `training_workflows`
- ✅ Updated `initialize_agent_database()` to also create training data tables

### 2. `app/core/training_data_store.py`
- ✅ Updated to use PostgreSQL instead of CSV/JSONL
- ✅ All `store_*` methods now write to PostgreSQL
- ✅ Still publishes to Synapse for real-time streaming
- ✅ Added `export_for_training()` - exports from PostgreSQL to JSONL
- ✅ Added `export_to_csv()` - exports from PostgreSQL to CSV
- ✅ Supports filtering by date range and custom filters

---

## 🚀 Usage

### Storing Training Data

```python
from app.core.training_data_store import get_training_data_store

store = get_training_data_store()

# Store agent event
await store.store_agent_event(
    event_type="task_completed",
    agent_id="agent-123",
    payload={"result": "success"}
)

# Store agent task
await store.store_agent_task(
    task_id="task-456",
    agent_id="agent-123",
    task_type="code_generation",
    status="completed"
)

# Store tool execution
await store.store_tool_execution(
    execution_id="exec-789",
    agent_id="agent-123",
    tool_name="python_executor",
    input_params={"code": "print('hello')"},
    output_result={"stdout": "hello"}
)

# Store workflow data
await store.store_workflow_data(
    workflow_id="workflow-101",
    workflow_type="multi_agent",
    phase="executing"
)
```

### Exporting for Training

```python
# Export to JSONL (standard for ML training)
await store.export_for_training(
    data_type="events",
    output_path="data/training/events_export.jsonl",
    start_date="2024-01-01",
    end_date="2024-01-31",
    filters={"agent_id": "agent-123"}
)

# Export to CSV
await store.export_to_csv(
    data_type="tasks",
    output_path="data/training/tasks_export.csv",
    start_date="2024-01-01"
)
```

---

## 🔍 Querying Training Data

You can query training data directly from PostgreSQL:

```sql
-- Get all events from last week
SELECT * FROM training_data.training_events 
WHERE created_at >= NOW() - INTERVAL '7 days'
ORDER BY created_at DESC;

-- Get failed tasks
SELECT * FROM training_data.training_tasks 
WHERE status = 'failed'
ORDER BY created_at DESC;

-- Get tool executions by tool name
SELECT * FROM training_data.training_tool_executions 
WHERE tool_name = 'python_executor'
AND created_at >= '2024-01-01';
```

---

## 🎯 Benefits

1. **Queryable**: Fast SQL queries with filtering, aggregation, joins
2. **Indexed**: All tables have proper indexes for performance
3. **Real-time**: Still streams to Synapse for real-time processing
4. **Exportable**: Easy export to CSV/JSONL for ML training
5. **Organized**: Clear schema separation (public vs training_data)
6. **No Redundancy**: Uses existing PostgreSQL, no new infrastructure

---

## 🔧 Initialization

Training data tables are automatically created when:
- `initialize_agent_database()` is called (via `agent_playground_api.py`)
- System starts up (via `SystemInitializer`)

No manual migration needed - tables are created on first use.

---

## 📝 Next Steps (Optional)

1. **RAG Integration**: Index training data to Milvus for semantic search (opt-in)
2. **Analytics**: Add views/aggregations for common queries
3. **Retention**: Add data retention policies (archive old data)
4. **Backup**: Set up automated backups for training_data schema

---

## ✅ Status

- ✅ PostgreSQL schema created
- ✅ Training data tables created
- ✅ Store methods updated to use PostgreSQL
- ✅ Export methods added (JSONL + CSV)
- ✅ Synapse streaming still works
- ✅ Initialization integrated
- ✅ No linter errors

**Ready for use!** 🎉

