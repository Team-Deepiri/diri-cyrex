"""
Agent Database Tables and Migrations
Database schema for agent system components
"""
from typing import Optional
from ..database.postgres import get_postgres_manager
from ..logging_config import get_logger

logger = get_logger("cyrex.database.agent_tables")


async def create_agent_tables():
    """Create all agent-related database tables"""
    postgres = await get_postgres_manager()
    
    # Agent configurations table
    await postgres.execute("""
        CREATE TABLE IF NOT EXISTS agent_configs (
            agent_id VARCHAR(255) PRIMARY KEY,
            agent_type VARCHAR(100) NOT NULL,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            role VARCHAR(100),
            capabilities JSONB DEFAULT '[]'::jsonb,
            tools JSONB DEFAULT '[]'::jsonb,
            model_config JSONB DEFAULT '{}'::jsonb,
            system_prompt TEXT,
            temperature FLOAT DEFAULT 0.7,
            max_tokens INTEGER DEFAULT 2000,
            guardrails JSONB DEFAULT '[]'::jsonb,
            metadata JSONB DEFAULT '{}'::jsonb,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_agent_configs_type ON agent_configs(agent_type);
        CREATE INDEX IF NOT EXISTS idx_agent_configs_role ON agent_configs(role);
    """)
    
    # Agent instances (running agents)
    await postgres.execute("""
        CREATE TABLE IF NOT EXISTS agent_instances (
            instance_id VARCHAR(255) PRIMARY KEY,
            agent_id VARCHAR(255) REFERENCES agent_configs(agent_id),
            session_id VARCHAR(255),
            status VARCHAR(50) DEFAULT 'idle',
            current_task_id VARCHAR(255),
            state_data JSONB DEFAULT '{}'::jsonb,
            metrics JSONB DEFAULT '{}'::jsonb,
            started_at TIMESTAMP DEFAULT NOW(),
            last_activity TIMESTAMP DEFAULT NOW(),
            stopped_at TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_agent_instances_agent ON agent_instances(agent_id);
        CREATE INDEX IF NOT EXISTS idx_agent_instances_session ON agent_instances(session_id);
        CREATE INDEX IF NOT EXISTS idx_agent_instances_status ON agent_instances(status);
    """)
    
    # Agent tasks
    await postgres.execute("""
        CREATE TABLE IF NOT EXISTS agent_tasks (
            task_id VARCHAR(255) PRIMARY KEY,
            agent_id VARCHAR(255),
            instance_id VARCHAR(255),
            task_type VARCHAR(100) NOT NULL,
            priority VARCHAR(20) DEFAULT 'normal',
            status VARCHAR(50) DEFAULT 'pending',
            input_data JSONB DEFAULT '{}'::jsonb,
            output_data JSONB,
            error TEXT,
            retry_count INTEGER DEFAULT 0,
            max_retries INTEGER DEFAULT 3,
            timeout_seconds INTEGER DEFAULT 300,
            created_at TIMESTAMP DEFAULT NOW(),
            started_at TIMESTAMP,
            completed_at TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_agent_tasks_agent ON agent_tasks(agent_id);
        CREATE INDEX IF NOT EXISTS idx_agent_tasks_status ON agent_tasks(status);
        CREATE INDEX IF NOT EXISTS idx_agent_tasks_type ON agent_tasks(task_type);
        CREATE INDEX IF NOT EXISTS idx_agent_tasks_created ON agent_tasks(created_at);
    """)
    
    # Agent interactions/conversations
    await postgres.execute("""
        CREATE TABLE IF NOT EXISTS agent_interactions (
            interaction_id VARCHAR(255) PRIMARY KEY,
            session_id VARCHAR(255),
            agent_id VARCHAR(255),
            role VARCHAR(20) NOT NULL,
            content TEXT NOT NULL,
            tool_calls JSONB,
            token_count INTEGER,
            confidence FLOAT,
            metadata JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_agent_interactions_session ON agent_interactions(session_id);
        CREATE INDEX IF NOT EXISTS idx_agent_interactions_agent ON agent_interactions(agent_id);
        CREATE INDEX IF NOT EXISTS idx_agent_interactions_created ON agent_interactions(created_at);
    """)
    
    # Tool executions
    await postgres.execute("""
        CREATE TABLE IF NOT EXISTS tool_executions (
            execution_id VARCHAR(255) PRIMARY KEY,
            agent_id VARCHAR(255),
            task_id VARCHAR(255),
            tool_name VARCHAR(255) NOT NULL,
            tool_category VARCHAR(100),
            input_params JSONB DEFAULT '{}'::jsonb,
            output_result JSONB,
            status VARCHAR(50) DEFAULT 'pending',
            error TEXT,
            execution_time_ms FLOAT,
            created_at TIMESTAMP DEFAULT NOW(),
            completed_at TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_tool_executions_agent ON tool_executions(agent_id);
        CREATE INDEX IF NOT EXISTS idx_tool_executions_tool ON tool_executions(tool_name);
        CREATE INDEX IF NOT EXISTS idx_tool_executions_status ON tool_executions(status);
    """)
    
    # Workflows
    await postgres.execute("""
        CREATE TABLE IF NOT EXISTS workflows (
            workflow_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            workflow_type VARCHAR(100),
            phase VARCHAR(50) DEFAULT 'initializing',
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
        CREATE INDEX IF NOT EXISTS idx_workflows_phase ON workflows(phase);
        CREATE INDEX IF NOT EXISTS idx_workflows_type ON workflows(workflow_type);
        CREATE INDEX IF NOT EXISTS idx_workflows_created ON workflows(created_at);
    """)
    
    # Prompt templates
    await postgres.execute("""
        CREATE TABLE IF NOT EXISTS prompt_templates (
            template_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            category VARCHAR(100),
            version VARCHAR(20) DEFAULT '1.0.0',
            system_template TEXT NOT NULL,
            user_template TEXT NOT NULL,
            variables JSONB DEFAULT '[]'::jsonb,
            tools JSONB DEFAULT '[]'::jsonb,
            examples JSONB DEFAULT '[]'::jsonb,
            temperature FLOAT DEFAULT 0.7,
            max_tokens INTEGER DEFAULT 2000,
            top_p FLOAT DEFAULT 0.9,
            tags JSONB DEFAULT '[]'::jsonb,
            author VARCHAR(255),
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_prompt_templates_category ON prompt_templates(category);
        CREATE INDEX IF NOT EXISTS idx_prompt_templates_name ON prompt_templates(name);
    """)
    
    # Agent events for monitoring
    await postgres.execute("""
        CREATE TABLE IF NOT EXISTS agent_events (
            event_id VARCHAR(255) PRIMARY KEY,
            event_type VARCHAR(100) NOT NULL,
            agent_id VARCHAR(255),
            workflow_id VARCHAR(255),
            session_id VARCHAR(255),
            source VARCHAR(100),
            payload JSONB DEFAULT '{}'::jsonb,
            severity VARCHAR(20) DEFAULT 'info',
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_agent_events_type ON agent_events(event_type);
        CREATE INDEX IF NOT EXISTS idx_agent_events_agent ON agent_events(agent_id);
        CREATE INDEX IF NOT EXISTS idx_agent_events_created ON agent_events(created_at);
        CREATE INDEX IF NOT EXISTS idx_agent_events_severity ON agent_events(severity);
    """)
    
    # Performance metrics
    await postgres.execute("""
        CREATE TABLE IF NOT EXISTS agent_metrics (
            metric_id VARCHAR(255) PRIMARY KEY,
            agent_id VARCHAR(255),
            metric_name VARCHAR(100) NOT NULL,
            metric_value FLOAT NOT NULL,
            unit VARCHAR(50),
            tags JSONB DEFAULT '{}'::jsonb,
            recorded_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_agent_metrics_agent ON agent_metrics(agent_id);
        CREATE INDEX IF NOT EXISTS idx_agent_metrics_name ON agent_metrics(metric_name);
        CREATE INDEX IF NOT EXISTS idx_agent_metrics_recorded ON agent_metrics(recorded_at);
    """)
    
    logger.info("All agent database tables created successfully")


async def drop_agent_tables():
    """Drop all agent-related tables (CAUTION: destructive)"""
    postgres = await get_postgres_manager()
    
    tables = [
        "agent_metrics",
        "agent_events",
        "prompt_templates",
        "workflows",
        "tool_executions",
        "agent_interactions",
        "agent_tasks",
        "agent_instances",
        "agent_configs",
    ]
    
    for table in tables:
        await postgres.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    
    logger.info("All agent database tables dropped")


async def check_tables_exist() -> bool:
    """Check if agent tables exist"""
    postgres = await get_postgres_manager()
    
    result = await postgres.fetchval("""
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'agent_configs'
    """)
    
    return result > 0


async def initialize_agent_database():
    """Initialize agent database (create tables if not exist)"""
    if not await check_tables_exist():
        await create_agent_tables()
        logger.info("Agent database initialized")
    else:
        logger.info("Agent database tables already exist")

