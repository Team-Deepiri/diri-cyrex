"""
Agent Factory
Factory for creating and initializing agents with Ollama integration
"""
from typing import Optional, Dict, Any
from ..core.types import AgentConfig, AgentRole, IndustryNiche
from ..core.agent_initializer import get_agent_initializer
from ..integrations.local_llm import get_local_llm, LocalLLMProvider
from ..logging_config import get_logger
from .base_agent import BaseAgent
from .implementations import (
    TaskDecomposerAgent,
    TimeOptimizerAgent,
    CreativeSparkerAgent,
    QualityAssuranceAgent,
    EngagementSpecialistAgent,
    VendorFraudAgent,
)

logger = get_logger("cyrex.agent.factory")


class AgentFactory:
    """Factory for creating agents with proper initialization"""
    
    @staticmethod
    async def create_agent(
        role: AgentRole,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        model_name: str = "llama3:8b",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> BaseAgent:
        """
        Create an agent with Ollama integration
        
        Args:
            role: Agent role
            agent_id: Optional agent ID (will be generated if not provided)
            session_id: Optional session ID
            model_name: Ollama model name
            temperature: LLM temperature
            max_tokens: Max tokens for responses
            **kwargs: Additional agent configuration
        
        Returns:
            Initialized agent instance
        """
        # Get or create agent config
        agent_init = await get_agent_initializer()
        
        if agent_id:
            agent_config = await agent_init.get_agent(agent_id)
            if not agent_config:
                # Create new agent
                agent_config = await agent_init.register_agent(
                    role=role,
                    name=kwargs.get("name", f"{role.value.replace('_', ' ').title()} Agent"),
                    description=kwargs.get("description", ""),
                    capabilities=kwargs.get("capabilities", []),
                    tools=kwargs.get("tools", []),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model_config={"model": model_name},
                )
        else:
            # Create new agent
            agent_config = await agent_init.register_agent(
                role=role,
                name=kwargs.get("name", f"{role.value.replace('_', ' ').title()} Agent"),
                description=kwargs.get("description", ""),
                capabilities=kwargs.get("capabilities", []),
                tools=kwargs.get("tools", []),
                temperature=temperature,
                max_tokens=max_tokens,
                model_config={"model": model_name},
            )
        
        # Initialize Ollama LLM
        llm_provider = get_local_llm(
            backend="ollama",
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        if not llm_provider:
            logger.warning(f"Failed to initialize Ollama, agent may not work properly")
        
        # Create agent instance based on role
        agent_class = AgentFactory._get_agent_class(role)
        
        agent = agent_class(
            agent_config=agent_config,
            llm_provider=llm_provider,
            session_id=session_id,
        )
        
        # Register tools
        await AgentFactory._register_agent_tools(agent)
        
        logger.info(f"Agent created: {agent.agent_id}", role=role.value, model=model_name)
        return agent
    
    @staticmethod
    def _get_agent_class(role: AgentRole) -> type:
        """Get agent class for role"""
        role_map = {
            AgentRole.TASK_DECOMPOSER: TaskDecomposerAgent,
            AgentRole.TIME_OPTIMIZER: TimeOptimizerAgent,
            AgentRole.CREATIVE_SPARKER: CreativeSparkerAgent,
            AgentRole.QUALITY_ASSURANCE: QualityAssuranceAgent,
            AgentRole.ENGAGEMENT_SPECIALIST: EngagementSpecialistAgent,
            # Vendor Fraud Detection Agents
            AgentRole.INVOICE_ANALYZER: VendorFraudAgent,
            AgentRole.VENDOR_INTELLIGENCE: VendorFraudAgent,
            AgentRole.PRICING_BENCHMARK: VendorFraudAgent,
            AgentRole.FRAUD_DETECTOR: VendorFraudAgent,
            AgentRole.DOCUMENT_PROCESSOR: VendorFraudAgent,
            AgentRole.RISK_ASSESSOR: VendorFraudAgent,
        }
        
        agent_class = role_map.get(role)
        if not agent_class:
            # Default to base agent
            from .base_agent import BaseAgent
            return BaseAgent
        
        return agent_class
    
    @staticmethod
    async def _register_agent_tools(agent: BaseAgent):
        """Register tools with agent"""
        from .tools.api_tools import register_api_tools
        from .tools.memory_tools import register_memory_tools
        from .tools.utility_tools import register_utility_tools
        
        await register_memory_tools(agent)
        await register_api_tools(agent)
        await register_utility_tools(agent)


# Convenience function wrapper for easier imports
async def create_agent(
    role: AgentRole,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    model_name: str = "llama3:8b",
    temperature: float = 0.7,
    max_tokens: int = 2000,
    **kwargs
) -> BaseAgent:
    """
    Create an agent with Ollama integration (convenience wrapper)
    
    This is a convenience function that wraps AgentFactory.create_agent()
    for easier imports and usage.
    
    Args:
        role: Agent role
        agent_id: Optional agent ID (will be generated if not provided)
        session_id: Optional session ID
        model_name: Ollama model name
        temperature: LLM temperature
        max_tokens: Max tokens for responses
        **kwargs: Additional agent configuration
    
    Returns:
        Initialized agent instance
    """
    return await AgentFactory.create_agent(
        role=role,
        agent_id=agent_id,
        session_id=session_id,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
