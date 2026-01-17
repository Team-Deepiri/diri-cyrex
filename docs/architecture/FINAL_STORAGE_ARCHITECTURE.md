# Final Storage Architecture

## ✅ Confirmed Architecture

| Data Type       | Storage                              | Reason                |
| --------------- | ------------------------------------ | --------------------- |
| Workflow state  | PostgreSQL                           | ACID, real-time       |
| Task executions | PostgreSQL                           | Monitoring, retries   |
| Events (audit)  | PostgreSQL                           | Debugging, compliance |
| Conversations   | PostgreSQL                           | Runtime correctness   |
| Training data   | CSV / JSONL (→ object storage later) | Batch ML              |
| Metrics         | Prometheus / InfluxDB                | Time-series           |

---

## 📊 PostgreSQL: Operational Tables

### 1. **Workflows** (`workflows`)
**Purpose:** Operational workflow state for real-time querying

**Schema:**
```sql
CREATE TABLE workflows (
    workflow_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255),
    description TEXT,
    workflow_type VARCHAR(100),
    status VARCHAR(50) DEFAULT 'pending',  -- pending, running, completed, failed
    current_step VARCHAR(255),
    total_steps INTEGER DEFAULT 0,
    completed_steps INTEGER DEFAULT 0,
    state_data JSONB DEFAULT '{}'::jsonb,
    step_results JSONB DEFAULT '{}'::jsonb,
    assigned_agents JSONB DEFAULT '[]'::jsonb,
    checkpoints JSONB DEFAULT '[]'::jsonb,
    error TEXT,
    error_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,
    tags JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    deadline TIMESTAMP
);
```

**Use Cases:**
- Real-time workflow status queries
- Workflow management UI
- State consistency (ACID)

---

### 2. **Task Executions** (`task_executions`)
**Purpose:** Operational task execution state for monitoring and retries

**Schema:**
```sql
CREATE TABLE task_executions (
    execution_id VARCHAR(255) PRIMARY KEY,
    workflow_id VARCHAR(255) REFERENCES workflows(workflow_id),
    agent_id VARCHAR(255),
    task_name VARCHAR(255) NOT NULL,
    task_type VARCHAR(100),
    priority VARCHAR(20) DEFAULT 'normal',
    status VARCHAR(50) DEFAULT 'pending',  -- pending, running, completed, failed
    input_data JSONB DEFAULT '{}'::jsonb,
    output_data JSONB,
    error TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    timeout_seconds INTEGER DEFAULT 300,
    execution_time_ms FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

**Use Cases:**
- Task monitoring dashboard
- Retry logic
- Performance tracking
- Error debugging

---

### 3. **Events** (`events`)
**Purpose:** Append-only audit log for debugging and compliance

**Schema:**
```sql
CREATE TABLE events (
    event_id VARCHAR(255) PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100),  -- workflow, task, agent
    entity_id VARCHAR(255),
    workflow_id VARCHAR(255),
    agent_id VARCHAR(255),
    session_id VARCHAR(255),
    source VARCHAR(100) DEFAULT 'cyrex',
    payload JSONB DEFAULT '{}'::jsonb,
    severity VARCHAR(20) DEFAULT 'info',
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Use Cases:**
- Audit trail
- Debugging (what happened when)
- Compliance logging
- State reconstruction

---

### 4. **Conversations** (`agent_playground_messages`)
**Purpose:** Conversation history for runtime correctness

**Schema:**
```sql
CREATE TABLE agent_playground_messages (
    message_id VARCHAR(255) PRIMARY KEY,
    instance_id VARCHAR(255) NOT NULL,
    agent_id VARCHAR(255),
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    tool_calls JSONB,
    is_error BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Use Cases:**
- Chat history continuity
- Conversation context
- User experience

---

## 📁 Training Data Store: CSV/JSONL

### Location
- **Files:** `data/training/{type}/{type}_YYYY-MM-DD.csv`
- **Real-time:** Synapse (Redis Streams)
- **Export:** JSONL for ML training

### Data Types
- Agent events
- Agent tasks
- Tool executions
- Workflow data

### Why CSV/JSONL?
- ✅ Standard format for ML training
- ✅ Batch processing friendly
- ✅ Easy export/import
- ✅ Can move to object storage (S3/GCS) later

---

## 📈 Metrics: Time-Series DB

### Storage
- **Prometheus** - Metrics collection
- **InfluxDB** - Time-series analytics

### What Goes Here
- Performance metrics (latency, throughput)
- Resource usage (CPU, memory)
- Business metrics (success rate, error rate)

---

## 🔄 Data Flow

### Operational Data Flow:
```
Workflow/Task Execution
    ↓
PostgreSQL (operational state) ← Real-time querying
    ↓
Events Table (append-only) ← Audit trail
    ↓
Training Data Store (CSV/JSONL) ← For ML training (optional copy)
```

### Training Data Flow:
```
Real-time Event
    ↓
Synapse (Redis Streams) ← Real-time streaming
    ↓
CSV/JSONL Files ← Training data storage
    ↓
Export to JSONL ← For ML training pipelines
```

---

## 🎯 Key Principles

1. **PostgreSQL = Operational State**
   - Real-time querying
   - ACID transactions
   - State consistency

2. **CSV/JSONL = Training Data**
   - Batch processing
   - ML training format
   - Can scale to object storage

3. **Time-Series DB = Metrics**
   - Optimized for time-series queries
   - Efficient compression
   - Built-in aggregations

4. **Separation of Concerns**
   - Operational ≠ Training
   - Different access patterns
   - Different storage needs

---

## ✅ Implementation Status

- ✅ PostgreSQL tables created (`workflows`, `task_executions`, `events`, `agent_playground_messages`)
- ✅ Training data store using CSV/JSONL
- ✅ Synapse streaming for real-time
- ✅ Metrics in Prometheus/InfluxDB

**Architecture is production-ready!** 🎉

