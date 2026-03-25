"""
Company Data Automation Service
Main service for processing company data and automating tools
Integrates with LoRA adapters, agents, and tool systems
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from ..core.types import AgentRole
from ..agents import AgentFactory
from ..integrations.lora_adapter_service import get_lora_service
from ..integrations.api_bridge import get_api_bridge
from ..core.memory_manager import get_memory_manager
from ..core.session_manager import get_session_manager
from ..integrations.synapse_broker import get_synapse_broker
from ..database.postgres import get_postgres_manager
from ..logging_config import get_logger
from datetime import datetime
import json

logger = get_logger("cyrex.company_automation")


class CompanyDataAutomation:
    """
    Main service for company data automation
    Processes company data, trains adapters, and automates tools
    """
    
    def __init__(self):
        self.logger = logger
        self._agents: Dict[str, Any] = {}
        self._company_sessions: Dict[str, str] = {}
    
    async def initialize(self):
        """Initialize automation service"""
        # Create database tables
        postgres = await get_postgres_manager()
        await postgres.execute("""
            CREATE TABLE IF NOT EXISTS company_automation_jobs (
                job_id VARCHAR(255) PRIMARY KEY,
                company_id VARCHAR(255) NOT NULL,
                job_type VARCHAR(100) NOT NULL,
                input_data JSONB,
                output_data JSONB,
                status VARCHAR(50) NOT NULL,
                adapter_id VARCHAR(255),
                agent_id VARCHAR(255),
                metadata JSONB,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_company_jobs_company_id ON company_automation_jobs(company_id);
            CREATE INDEX IF NOT EXISTS idx_company_jobs_status ON company_automation_jobs(status);
            
            CREATE TABLE IF NOT EXISTS company_tools (
                company_id VARCHAR(255) NOT NULL,
                tool_name VARCHAR(255) NOT NULL,
                tool_config JSONB NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                PRIMARY KEY (company_id, tool_name)
            );
            CREATE INDEX IF NOT EXISTS idx_company_tools_company_id ON company_tools(company_id);
        """)
        
        logger.info("Company data automation service initialized")
    
    async def process_company_data(
        self,
        company_id: str,
        data: Dict[str, Any],
        task_type: str = "automation",
        use_adapter: bool = True,
    ) -> Dict[str, Any]:
        """
        Process company data and automate tools
        
        Args:
            company_id: Company identifier
            data: Company data to process
            task_type: Type of automation task
            use_adapter: Whether to use company-specific LoRA adapter
        
        Returns:
            Processing results
        """
        job_id = f"job_{company_id}_{datetime.utcnow().isoformat()}"
        
        try:
            # Store job in database
            postgres = await get_postgres_manager()
            await postgres.execute("""
                INSERT INTO company_automation_jobs (job_id, company_id, job_type, input_data, status, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, job_id, company_id, task_type, json.dumps(data), "processing",
                datetime.utcnow(), datetime.utcnow())
            
            # Get or create company session
            session_id = await self._get_company_session(company_id)
            
            # Get company-specific agent with adapter
            agent = await self._get_company_agent(company_id, session_id, use_adapter)
            
            # Process with agent
            result = await agent.invoke(
                input_text=f"Process this {task_type} task: {json.dumps(data, indent=2)}",
                context={
                    "company_id": company_id,
                    "task_type": task_type,
                    "data": data,
                },
            )
            
            # Execute any tool calls
            tool_results = []
            for tool_call in result.tool_calls:
                tool_result = await self._execute_automation_tool(tool_call, company_id)
                tool_results.append(tool_result)
            
            # Update job status
            await postgres.execute("""
                UPDATE company_automation_jobs SET
                    output_data = $1,
                    status = $2,
                    agent_id = $3,
                    updated_at = $4,
                    completed_at = $5
                WHERE job_id = $6
            """, json.dumps({
                "result": result.content,
                "tool_results": tool_results,
                "metadata": result.metadata,
            }), "completed", agent.agent_id, datetime.utcnow(), datetime.utcnow(), job_id)
            
            # Store in memory for future reference
            memory_mgr = await get_memory_manager()
            await memory_mgr.store_memory(
                content=f"Company {company_id} processed {task_type}: {result.content}",
                memory_type="episodic",
                user_id=company_id,
                importance=0.8,
                metadata={"job_id": job_id, "task_type": task_type},
            )
            
            return {
                "job_id": job_id,
                "status": "completed",
                "result": result.content,
                "tool_results": tool_results,
                "confidence": result.confidence,
            }
            
        except Exception as e:
            logger.error(f"Company data processing failed: {e}", exc_info=True)
            
            # Update job status
            postgres = await get_postgres_manager()
            await postgres.execute("""
                UPDATE company_automation_jobs SET
                    status = $1,
                    metadata = $2,
                    updated_at = $3
                WHERE job_id = $4
            """, "failed", json.dumps({"error": str(e)}), datetime.utcnow(), job_id)
            
            raise
    
    async def train_company_adapter(
        self,
        company_id: str,
        training_data: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Request LoRA adapter training for company data
        
        Args:
            company_id: Company identifier
            training_data: Company-specific training data
            config: LoRA configuration
        
        Returns:
            Request ID
        """
        lora_service = await get_lora_service()
        request_id = await lora_service.request_adapter_training(
            company_id=company_id,
            training_data=training_data,
            config=config,
            use_qlora=True,  # Use QLoRA for efficiency
        )
        
        logger.info(f"Company adapter training requested: {request_id}", company_id=company_id)
        return request_id
    
    async def _get_company_session(self, company_id: str) -> str:
        """Get or create session for company"""
        if company_id in self._company_sessions:
            return self._company_sessions[company_id]
        
        session_mgr = await get_session_manager()
        session = await session_mgr.create_session(
            user_id=company_id,
            metadata={"type": "company", "company_id": company_id},
            ttl=86400,  # 24 hours
        )
        
        self._company_sessions[company_id] = session.session_id
        return session.session_id
    
    async def _get_company_agent(
        self,
        company_id: str,
        session_id: str,
        use_adapter: bool = True,
    ):
        """Get or create agent for company"""
        if company_id in self._agents:
            return self._agents[company_id]
        
        # Create orchestrator agent for company
        agent = await AgentFactory.create_agent(
            role=AgentRole.ORCHESTRATOR,
            session_id=session_id,
            model_name="llama3:8b",
            metadata={"company_id": company_id},
        )
        
        # Load company adapter if available
        if use_adapter:
            lora_service = await get_lora_service()
            adapter = await lora_service.get_adapter_for_company(company_id)
            if adapter:
                # Store adapter reference in agent metadata
                agent.config.metadata["adapter_id"] = company_id
                logger.info(f"Company adapter loaded for agent: {company_id}")
        
        self._agents[company_id] = agent
        return agent
    
    async def _execute_automation_tool(
        self,
        tool_call: Dict[str, Any],
        company_id: str,
    ) -> Dict[str, Any]:
        """Execute automation tool for company"""
        tool_name = tool_call.get("tool")
        parameters = tool_call.get("parameters", {})
        
        # Get API bridge
        api_bridge = await get_api_bridge()
        
        # Add company context to parameters
        parameters["company_id"] = company_id
        
        try:
            # Call tool via API bridge
            result = await api_bridge.call_tool(tool_name, parameters)
            return {
                "tool": tool_name,
                "status": result.status,
                "result": result.result,
                "error": result.error,
            }
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {
                "tool": tool_name,
                "status": "error",
                "error": str(e),
            }
    
    async def register_company_tools(
        self,
        company_id: str,
        tools: List[Dict[str, Any]],
    ):
        """Register company-specific tools for automation"""
        api_bridge = await get_api_bridge()
        
        registered_tools = []
        for tool in tools:
            tool_name = f"{company_id}_{tool['name']}"
            await api_bridge.register_tool(
                tool_name=tool_name,
                api_endpoint=tool.get("endpoint"),
                method=tool.get("method", "POST"),
                headers=tool.get("headers", {}),
                auth=tool.get("auth"),
                description=tool.get("description", f"Company tool: {tool.get('name', '')}"),
                rate_limit=tool.get("rate_limit"),
            )
            registered_tools.append(tool_name)
        
        # Store tool registrations in database
        postgres = await get_postgres_manager()
        for tool in tools:
            tool_name = f"{company_id}_{tool['name']}"
            await postgres.execute("""
                INSERT INTO company_tools (company_id, tool_name, tool_config, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (company_id, tool_name) DO UPDATE SET
                    tool_config = EXCLUDED.tool_config,
                    updated_at = EXCLUDED.updated_at
            """, company_id, tool_name, json.dumps(tool), datetime.utcnow(), datetime.utcnow())
        
        logger.info(f"Company tools registered: {company_id}", count=len(registered_tools))


# Global service instance
_automation_service: Optional[CompanyDataAutomation] = None


async def get_automation_service() -> CompanyDataAutomation:
    """Get or create automation service singleton"""
    global _automation_service
    if _automation_service is None:
        _automation_service = CompanyDataAutomation()
        await _automation_service.initialize()
    return _automation_service

