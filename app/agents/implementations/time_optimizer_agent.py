"""
Time Optimizer Agent Implementation
"""
from typing import Dict, Any
from ..base_agent import BaseAgent
from ..prompts.time_optimizer_prompts import TIME_OPTIMIZER_PROMPT
from ...core.types import AgentConfig, AgentRole

class TimeOptimizerAgent(BaseAgent):
    """Agent specialized in time optimization"""
    
    def __init__(self, agent_config: AgentConfig = None, llm_provider = None, session_id: str = None):
        if not agent_config:
            agent_config = AgentConfig(
                role=AgentRole.TIME_OPTIMIZER,
                name="Time Optimizer Agent",
                description="Optimizes task scheduling and time management",
                capabilities=["time_optimization", "scheduling", "efficiency_analysis"],
            )
        
        agent_config.system_prompt = TIME_OPTIMIZER_PROMPT
        super().__init__(agent_config, llm_provider=llm_provider, session_id=session_id)
    
    async def process(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a time optimization request"""
        task_description = task.get("description", task.get("task", ""))
        
        response = await self.invoke(
            input_text=f"Optimize time for: {task_description}",
            context=context,
        )
        
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "optimization": response.content,
            "confidence": response.confidence,
        }

