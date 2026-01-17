"""
Agent Database Tables and Migrations
Database schema for agent system components
"""
from typing import Optional
from ..database.postgres import get_postgres_manager
from ..logging_config import get_logger

logger = get_logger("cyrex.database.agent_tables")


async def create_agent_tables():
    """
    Create operational database tables for agent system in cyrex schema.
    
    All AI/Agent tables are in the 'cyrex' schema to separate from backend/user data.
    
    PostgreSQL Storage (Operational) - cyrex schema:
    - agent_playground_messages: Conversation history (runtime correctness)
    - workflows: Workflow state (ACID, real-time querying)
    - task_executions: Task execution state (monitoring, retries)
    - events: Event audit log (debugging, compliance)
    
    Training Data Store (CSV/JSONL + Synapse):
    - Agent events → training_data_store.store_agent_event()
    - Agent tasks → training_data_store.store_agent_task()
    - Tool executions → training_data_store.store_tool_execution()
    - Workflow data → training_data_store.store_workflow_data()
    
    Not in PostgreSQL:
    - Agent configs: In-memory or config files
    - Agent instances: In-memory only (ephemeral)
    - Prompt templates: Code/config files
    - Metrics: Prometheus/InfluxDB (time-series)
    """
    postgres = await get_postgres_manager()
    
    # Create cyrex schema
    await postgres.execute("CREATE SCHEMA IF NOT EXISTS cyrex")
    
    # Conversations: Runtime correctness
    await postgres.execute("""
        CREATE TABLE IF NOT EXISTS cyrex.agent_playground_messages (
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
        CREATE INDEX IF NOT EXISTS idx_playground_messages_instance ON cyrex.agent_playground_messages(instance_id);
        CREATE INDEX IF NOT EXISTS idx_playground_messages_agent ON cyrex.agent_playground_messages(agent_id);
        CREATE INDEX IF NOT EXISTS idx_playground_messages_created ON cyrex.agent_playground_messages(created_at);
    """)
    
    # Workflows: Operational state (ACID, real-time)
    await postgres.execute("""
        CREATE TABLE IF NOT EXISTS cyrex.workflows (
            workflow_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255),
            description TEXT,
            workflow_type VARCHAR(100),
            status VARCHAR(50) DEFAULT 'pending',
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
        CREATE INDEX IF NOT EXISTS idx_workflows_status ON cyrex.workflows(status);
        CREATE INDEX IF NOT EXISTS idx_workflows_type ON cyrex.workflows(workflow_type);
        CREATE INDEX IF NOT EXISTS idx_workflows_created ON cyrex.workflows(created_at);
        CREATE INDEX IF NOT EXISTS idx_workflows_updated ON cyrex.workflows(updated_at);
    """)
    
    # Task Executions: Operational state (monitoring, retries)
    await postgres.execute("""
        CREATE TABLE IF NOT EXISTS cyrex.task_executions (
            execution_id VARCHAR(255) PRIMARY KEY,
            workflow_id VARCHAR(255) REFERENCES cyrex.workflows(workflow_id) ON DELETE SET NULL,
            agent_id VARCHAR(255),
            task_name VARCHAR(255) NOT NULL,
            task_type VARCHAR(100),
            priority VARCHAR(20) DEFAULT 'normal',
            status VARCHAR(50) DEFAULT 'pending',
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
        CREATE INDEX IF NOT EXISTS idx_task_executions_workflow ON cyrex.task_executions(workflow_id);
        CREATE INDEX IF NOT EXISTS idx_task_executions_agent ON cyrex.task_executions(agent_id);
        CREATE INDEX IF NOT EXISTS idx_task_executions_status ON cyrex.task_executions(status);
        CREATE INDEX IF NOT EXISTS idx_task_executions_created ON cyrex.task_executions(created_at);
    """)
    
    # Events: Audit log (debugging, compliance)
    await postgres.execute("""
        CREATE TABLE IF NOT EXISTS cyrex.events (
            event_id VARCHAR(255) PRIMARY KEY,
            event_type VARCHAR(100) NOT NULL,
            entity_type VARCHAR(100),
            entity_id VARCHAR(255),
            workflow_id VARCHAR(255),
            agent_id VARCHAR(255),
            session_id VARCHAR(255),
            source VARCHAR(100) DEFAULT 'cyrex',
            payload JSONB DEFAULT '{}'::jsonb,
            severity VARCHAR(20) DEFAULT 'info',
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_events_entity ON cyrex.events(entity_type, entity_id);
        CREATE INDEX IF NOT EXISTS idx_events_type ON cyrex.events(event_type);
        CREATE INDEX IF NOT EXISTS idx_events_workflow ON cyrex.events(workflow_id);
        CREATE INDEX IF NOT EXISTS idx_events_agent ON cyrex.events(agent_id);
        CREATE INDEX IF NOT EXISTS idx_events_created ON cyrex.events(created_at);
        CREATE INDEX IF NOT EXISTS idx_events_severity ON cyrex.events(severity);
    """)
    
    logger.info("Operational database tables created in cyrex schema (workflows, task_executions, events, conversations)")


async def drop_agent_tables():
    """Drop agent-related tables in cyrex schema (CAUTION: destructive)"""
    postgres = await get_postgres_manager()
    
    tables = [
        "cyrex.events",  # Drop first (no dependencies)
        "cyrex.task_executions",  # Drop second (depends on workflows)
        "cyrex.workflows",  # Drop third
        "cyrex.agent_playground_messages",  # Drop last
    ]
    
    for table in tables:
        await postgres.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    
    logger.info("Operational database tables dropped from cyrex schema")


async def check_tables_exist() -> bool:
    """Check if agent tables exist in cyrex schema"""
    postgres = await get_postgres_manager()
    
    result = await postgres.fetchval("""
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_schema = 'cyrex' 
        AND table_name = 'agent_playground_messages'
    """)
    
    return result > 0


async def initialize_agent_database():
    """Initialize agent database (create tables if not exist)"""
    if not await check_tables_exist():
        await create_agent_tables()
        logger.info("Agent database initialized")
    else:
        logger.info("Agent database tables already exist")
    
    # Note: Training data tables are NOT created here
    # Training data goes to CSV/JSONL files via training_data_store.py
    # and streams to Synapse for real-time processing

