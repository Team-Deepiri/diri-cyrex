# Cyrex AI System Architecture

## Overview

Comprehensive, enterprise-grade AI agent system with full state management, memory, messaging, and tool integration.

## Core Components

### 1. Database Layer (`database/postgres.py`)
- **PostgreSQL Connection Manager**: Async connection pooling with health checks
- **Connection Management**: Automatic reconnection, connection lifecycle
- **Query Execution**: Safe parameterized queries with async support

### 2. Type System (`core/types.py`)
- **Comprehensive Type Definitions**: All data structures for the system
- **Agent Types**: AgentRole, AgentStatus, AgentConfig
- **Memory Types**: MemoryType enum, Memory dataclass
- **Message Types**: Message, MessagePriority
- **Protocol Definitions**: MemoryStore, MessageBroker interfaces

### 3. Session Management (`core/session_manager.py`)
- **Session Lifecycle**: Create, update, delete, expiration
- **PostgreSQL Persistence**: Full session state stored in database
- **Automatic Cleanup**: Background task for expired sessions
- **Multi-User Support**: User and agent session tracking

### 4. Memory Management (`core/memory_manager.py`)
- **Multi-Tier Memory**:
  - Short-term: Session-based, in-memory
  - Long-term: Persistent, database-backed
  - Episodic: Event-based memories
  - Semantic: Factual knowledge
  - Working: Current context window
- **Vector Search**: Integration with Milvus for semantic search
- **Context Building**: Automatic context assembly from relevant memories

### 5. API Bridge (`integrations/api_bridge.py`)
- **Tool Registration**: Register external APIs as tools
- **Rate Limiting**: Per-tool rate limit enforcement
- **Authentication**: Bearer tokens, API keys
- **Retry Logic**: Automatic retries with exponential backoff
- **Error Handling**: Comprehensive error management

### 6. Synapse Message Broker (`integrations/synapse_broker.py`)
- **Pub/Sub System**: Channel-based messaging
- **Queue Management**: Persistent message queues
- **Priority Handling**: Message priority levels
- **Persistence**: PostgreSQL-backed message storage
- **Subscriptions**: Event-driven message delivery

### 7. LangGraph Integration (`core/langgraph_integration.py`)
- **State Machine**: Workflow state management
- **Node-Based Workflows**: Define agent workflows as graphs
- **State Persistence**: Workflow state stored in database
- **Fallback Mode**: Works without LangGraph library

### 8. Event Handler (`core/event_handler.py`)
- **Event Routing**: Type-based event routing
- **Middleware Support**: Event processing pipeline
- **Async Processing**: Background event processing
- **Event Persistence**: All events logged to database
- **Subscriptions**: Subscribe to specific event types

### 9. Agent Initializer (`core/agent_initializer.py`)
- **Agent Registration**: Register and configure agents
- **Configuration Management**: Persistent agent configs
- **State Tracking**: Agent status and activity tracking
- **Role-Based**: Support for multiple agent roles

### 10. Enhanced Guardrails (`core/enhanced_guardrails.py`)
- **Rule-Based Filtering**: Regex-based content filtering
- **Custom Validators**: Extensible validation system
- **Violation Logging**: All violations logged to database
- **Action Types**: Block, warn, modify, log
- **Severity Levels**: Low, medium, high, critical

### 11. System Initializer (`core/system_initializer.py`)
- **Unified Initialization**: Initialize all systems in correct order
- **Health Checks**: System-wide health monitoring
- **Graceful Shutdown**: Clean shutdown of all components

## Data Flow

```
User Request
    ↓
Session Manager (create/get session)
    ↓
Memory Manager (build context from memories)
    ↓
Guardrails (safety checks)
    ↓
Agent Initializer (get agent config)
    ↓
LangGraph (execute workflow)
    ↓
API Bridge (call external tools if needed)
    ↓
Synapse Broker (publish events/messages)
    ↓
Event Handler (process events)
    ↓
Memory Manager (store new memories)
    ↓
Response
```

## Database Schema

### Sessions Table
- `session_id` (PK)
- `user_id`, `agent_id`
- `status`, `context` (JSONB)
- `metadata` (JSONB)
- `created_at`, `updated_at`, `expires_at`, `last_activity`

### Memories Table
- `memory_id` (PK)
- `session_id`, `user_id`
- `memory_type`, `content`
- `metadata` (JSONB), `importance`
- `access_count`, `last_accessed`
- `created_at`, `expires_at`

### Agents Table
- `agent_id` (PK)
- `role`, `name`, `description`
- `capabilities` (JSONB), `tools` (JSONB)
- `model_config` (JSONB)
- `temperature`, `max_tokens`
- `system_prompt`, `guardrails` (JSONB)
- `metadata` (JSONB)
- `created_at`, `updated_at`

### Messages Table (Synapse)
- `message_id` (PK)
- `channel`, `sender`, `recipient`
- `priority`, `payload` (JSONB)
- `headers` (JSONB)
- `timestamp`, `expires_at`
- `retry_count`, `max_retries`, `processed`

### Events Table
- `event_id` (PK)
- `event_type`, `source`, `target`
- `payload` (JSONB), `metadata` (JSONB)
- `timestamp`, `processed`

### LangGraph States Table
- `state_id` (PK)
- `workflow_id`, `current_node`
- `next_nodes` (JSONB)
- `data` (JSONB), `history` (JSONB)
- `status`, `created_at`, `updated_at`

### Guardrail Rules Table
- `rule_id` (PK)
- `name`, `pattern`, `action`, `severity`
- `description`, `enabled`
- `created_at`, `updated_at`

### Guardrail Violations Table
- `violation_id` (PK)
- `rule_id`, `content`
- `action_taken`, `severity`
- `metadata` (JSONB), `timestamp`

## Usage Examples

### Initialize System
```python
from app.core.system_initializer import get_system_initializer

initializer = await get_system_initializer()
await initializer.initialize_all()
```

### Create Session
```python
from app.core.session_manager import get_session_manager

session_mgr = await get_session_manager()
session = await session_mgr.create_session(
    user_id="user123",
    agent_id="agent456",
    context={"task": "analyze data"},
    ttl=3600
)
```

### Store Memory
```python
from app.core.memory_manager import get_memory_manager
from app.core.types import MemoryType

memory_mgr = await get_memory_manager()
memory_id = await memory_mgr.store_memory(
    content="User prefers dark mode",
    memory_type=MemoryType.LONG_TERM,
    user_id="user123",
    importance=0.8
)
```

### Register API Tool
```python
from app.integrations.api_bridge import get_api_bridge

api_bridge = await get_api_bridge()
await api_bridge.register_tool(
    tool_name="weather_api",
    api_endpoint="https://api.weather.com/v1/forecast",
    method="GET",
    headers={"X-API-Key": "key123"},
    rate_limit={"requests": 100, "window": 60}
)
```

### Publish Message
```python
from app.integrations.synapse_broker import get_synapse_broker
from app.core.types import MessagePriority

broker = await get_synapse_broker()
message_id = await broker.publish(
    channel="agent_communication",
    payload={"task": "process_request", "data": {...}},
    sender="agent1",
    recipient="agent2",
    priority=MessagePriority.HIGH
)
```

### Create LangGraph Workflow
```python
from app.core.langgraph_integration import get_langgraph_manager

langgraph = await get_langgraph_manager()
workflow = langgraph.create_workflow("my_workflow")

# Add nodes
workflow.add_node("start", start_handler)
workflow.add_node("process", process_handler)
workflow.add_node("end", end_handler)

# Add edges
workflow.add_edge("START", "start")
workflow.add_edge("start", "process")
workflow.add_edge("process", "end")
workflow.add_edge("end", "END")

# Build and execute
workflow.build_graph()
result = await workflow.execute(initial_state={"input": "data"})
```

### Emit Event
```python
from app.core.event_handler import get_event_handler

event_handler = await get_event_handler()
event_id = await event_handler.emit(
    event_type="task_completed",
    payload={"task_id": "task123", "result": "success"},
    source="agent1",
    target="orchestrator"
)
```

### Register Agent
```python
from app.core.agent_initializer import get_agent_initializer
from app.core.types import AgentRole

agent_init = await get_agent_initializer()
agent = await agent_init.register_agent(
    role=AgentRole.TASK_DECOMPOSER,
    name="Task Decomposer Agent",
    description="Breaks down complex tasks",
    capabilities=["task_analysis", "decomposition"],
    tools=["api_tool1", "api_tool2"],
    temperature=0.7,
    system_prompt="You are a task decomposition expert..."
)
```

## Environment Variables

Required PostgreSQL configuration:
- `POSTGRES_HOST` (default: "postgres")
- `POSTGRES_PORT` (default: 5432)
- `POSTGRES_DB` (default: "deepiri")
- `POSTGRES_USER` (default: "deepiri")
- `POSTGRES_PASSWORD` (default: "deepiripassword")

## Scalability Considerations

1. **Connection Pooling**: PostgreSQL uses asyncpg with configurable pool sizes
2. **Caching**: In-memory caches for frequently accessed data
3. **Background Tasks**: Async processing for non-blocking operations
4. **Database Indexing**: All tables have appropriate indexes
5. **Rate Limiting**: Built-in rate limiting for API tools
6. **Message Queues**: Persistent queues handle high message volumes

## Security

1. **Guardrails**: Multi-layer content filtering
2. **PII Detection**: Automatic detection and handling
3. **Input Validation**: All inputs validated before processing
4. **SQL Injection Prevention**: Parameterized queries only
5. **Rate Limiting**: Prevents abuse of external APIs

## Future Enhancements

- [ ] Distributed message broker (Redis/RabbitMQ)
- [ ] GraphQL API layer
- [ ] Real-time WebSocket support
- [ ] Advanced ML-based content moderation
- [ ] Multi-tenant isolation
- [ ] Audit logging system
- [ ] Performance metrics and monitoring

