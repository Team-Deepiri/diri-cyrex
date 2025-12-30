"""
Agent Initialization System
Data structures and initialization logic for agent setup
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from ..core.types import AgentConfig, AgentRole, AgentStatus
from ..database.postgres import get_postgres_manager
from ..logging_config import get_logger
import json

logger = get_logger("cyrex.agent_initializer")


class AgentInitializer:
    """
    Manages agent initialization, configuration, and lifecycle
    Handles agent registration, configuration persistence, and state management
    """
    
    def __init__(self):
        self._agents: Dict[str, AgentConfig] = {}
        self._agent_states: Dict[str, AgentStatus] = {}
        self.logger = logger
    
    async def initialize(self):
        """Initialize agent initializer and create database tables"""
        # Create agents table
        postgres = await get_postgres_manager()
        await postgres.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id VARCHAR(255) PRIMARY KEY,
                role VARCHAR(100) NOT NULL,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                capabilities JSONB,
                tools JSONB,
                model_config JSONB,
                temperature FLOAT DEFAULT 0.7,
                max_tokens INTEGER DEFAULT 2000,
                system_prompt TEXT,
                guardrails JSONB,
                metadata JSONB,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_agents_role ON agents(role);
            CREATE INDEX IF NOT EXISTS idx_agents_name ON agents(name);
        """)
        
        # Create agent states table
        await postgres.execute("""
            CREATE TABLE IF NOT EXISTS agent_states (
                agent_id VARCHAR(255) PRIMARY KEY,
                status VARCHAR(50) NOT NULL,
                current_task VARCHAR(255),
                metadata JSONB,
                last_activity TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            );
        """)
        
        self.logger.info("Agent initializer initialized")
    
    async def register_agent(
        self,
        role: AgentRole,
        name: str,
        description: str = "",
        capabilities: Optional[List[str]] = None,
        tools: Optional[List[str]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
        guardrails: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentConfig:
        """Register a new agent"""
        agent = AgentConfig(
            role=role,
            name=name,
            description=description,
            capabilities=capabilities or [],
            tools=tools or [],
            model_config=model_config or {},
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            guardrails=guardrails or [],
            metadata=metadata or {},
        )
        
        # Store in database
        postgres = await get_postgres_manager()
        await postgres.execute("""
            INSERT INTO agents (agent_id, role, name, description, capabilities, tools,
                              model_config, temperature, max_tokens, system_prompt, guardrails, metadata, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            ON CONFLICT (agent_id) DO UPDATE SET
                role = EXCLUDED.role,
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                capabilities = EXCLUDED.capabilities,
                tools = EXCLUDED.tools,
                model_config = EXCLUDED.model_config,
                temperature = EXCLUDED.temperature,
                max_tokens = EXCLUDED.max_tokens,
                system_prompt = EXCLUDED.system_prompt,
                guardrails = EXCLUDED.guardrails,
                metadata = EXCLUDED.metadata,
                updated_at = EXCLUDED.updated_at
        """, agent.agent_id, agent.role.value, agent.name, agent.description,
            json.dumps(agent.capabilities), json.dumps(agent.tools),
            json.dumps(agent.model_config), agent.temperature, agent.max_tokens,
            agent.system_prompt, json.dumps(agent.guardrails), json.dumps(agent.metadata),
            agent.created_at, agent.updated_at)
        
        # Cache in memory
        self._agents[agent.agent_id] = agent
        self._agent_states[agent.agent_id] = AgentStatus.IDLE
        
        # Initialize agent state
        await postgres.execute("""
            INSERT INTO agent_states (agent_id, status, last_activity, updated_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (agent_id) DO NOTHING
        """, agent.agent_id, AgentStatus.IDLE.value, datetime.utcnow(), datetime.utcnow())
        
        self.logger.info(f"Agent registered: {agent.agent_id}", role=role.value, name=name)
        return agent
    
    async def get_agent(self, agent_id: str) -> Optional[AgentConfig]:
        """Get agent configuration by ID"""
        # Check cache first
        if agent_id in self._agents:
            return self._agents[agent_id]
        
        # Load from database
        postgres = await get_postgres_manager()
        row = await postgres.fetchrow("SELECT * FROM agents WHERE agent_id = $1", agent_id)
        
        if row:
            agent = AgentConfig(
                agent_id=row['agent_id'],
                role=AgentRole(row['role']),
                name=row['name'],
                description=row['description'] or "",
                capabilities=json.loads(row['capabilities']) if row['capabilities'] else [],
                tools=json.loads(row['tools']) if row['tools'] else [],
                model_config=json.loads(row['model_config']) if row['model_config'] else {},
                temperature=row['temperature'],
                max_tokens=row['max_tokens'],
                system_prompt=row['system_prompt'],
                guardrails=json.loads(row['guardrails']) if row['guardrails'] else [],
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                created_at=row['created_at'],
                updated_at=row['updated_at'],
            )
            self._agents[agent_id] = agent
            return agent
        
        return None
    
    async def list_agents(
        self,
        role: Optional[AgentRole] = None,
        limit: int = 100,
    ) -> List[AgentConfig]:
        """List agents with optional filtering"""
        postgres = await get_postgres_manager()
        
        query = "SELECT * FROM agents WHERE 1=1"
        params = []
        
        if role:
            query += " AND role = $1"
            params.append(role.value)
            query += f" ORDER BY created_at DESC LIMIT $2"
            params.append(limit)
        else:
            query += f" ORDER BY created_at DESC LIMIT $1"
            params.append(limit)
        
        rows = await postgres.fetch(query, *params)
        
        agents = []
        for row in rows:
            agent = AgentConfig(
                agent_id=row['agent_id'],
                role=AgentRole(row['role']),
                name=row['name'],
                description=row['description'] or "",
                capabilities=json.loads(row['capabilities']) if row['capabilities'] else [],
                tools=json.loads(row['tools']) if row['tools'] else [],
                model_config=json.loads(row['model_config']) if row['model_config'] else {},
                temperature=row['temperature'],
                max_tokens=row['max_tokens'],
                system_prompt=row['system_prompt'],
                guardrails=json.loads(row['guardrails']) if row['guardrails'] else [],
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                created_at=row['created_at'],
                updated_at=row['updated_at'],
            )
            agents.append(agent)
        
        return agents
    
    async def update_agent_status(
        self,
        agent_id: str,
        status: AgentStatus,
        current_task: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Update agent status"""
        postgres = await get_postgres_manager()
        await postgres.execute("""
            UPDATE agent_states SET
                status = $1,
                current_task = $2,
                metadata = $3,
                last_activity = $4,
                updated_at = $5
            WHERE agent_id = $6
        """, status.value, current_task, json.dumps(metadata or {}),
            datetime.utcnow(), datetime.utcnow(), agent_id)
        
        self._agent_states[agent_id] = status
        self.logger.debug(f"Agent status updated: {agent_id}", status=status.value)


# Global agent initializer
_agent_initializer: Optional[AgentInitializer] = None


async def get_agent_initializer() -> AgentInitializer:
    """Get or create agent initializer singleton"""
    global _agent_initializer
    if _agent_initializer is None:
        _agent_initializer = AgentInitializer()
        await _agent_initializer.initialize()
    return _agent_initializer

