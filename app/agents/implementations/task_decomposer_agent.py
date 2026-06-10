"""
Task Decomposer Agent Implementation
"""
from typing import Dict, Any
from ..base_agent import BaseAgent
from ..prompts.task_decomposer_prompts import TASK_DECOMPOSER_PROMPT
from ...core.types import AgentConfig, AgentRole
from ...core.agent_initializer import get_agent_initializer
from ...integrations.local_llm import get_local_llm
from ...logging_config import get_logger

logger = get_logger("cyrex.agent.task_decomposer")


class TaskDecomposerAgent(BaseAgent):
    """Agent specialized in breaking down complex tasks"""
    
    def __init__(self, agent_config: AgentConfig = None, llm_provider = None, session_id: str = None):
        if not agent_config:
            # Create default config
            agent_config = AgentConfig(
                role=AgentRole.TASK_DECOMPOSER,
                name="Task Decomposer Agent",
                description="Breaks down complex tasks into manageable subtasks",
                capabilities=["task_analysis", "decomposition", "dependency_analysis"],
            )
        
        # Set prompt template
        agent_config.system_prompt = TASK_DECOMPOSER_PROMPT
        
        super().__init__(agent_config, llm_provider=llm_provider, session_id=session_id)
    
    async def process(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task decomposition request"""
        task_description = task.get("description", task.get("task", ""))
        
        response = await self.invoke(
            input_text=f"Decompose this task: {task_description}",
            context=context,
        )
        
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "decomposition": response.content,
            "confidence": response.confidence,
            "metadata": response.metadata,
        }

