"""
Engagement Specialist Agent Implementation
"""
from typing import Dict, Any
from ..base_agent import BaseAgent
from ..prompts.engagement_specialist_prompts import ENGAGEMENT_SPECIALIST_PROMPT
from ...core.types import AgentConfig, AgentRole

class EngagementSpecialistAgent(BaseAgent):
    """Agent specialized in user engagement and motivation"""
    
    def __init__(self, agent_config: AgentConfig = None, llm_provider = None, session_id: str = None):
        if not agent_config:
            agent_config = AgentConfig(
                role=AgentRole.ENGAGEMENT_SPECIALIST,
                name="Engagement Specialist Agent",
                description="Maintains user motivation and engagement",
                capabilities=["motivation", "engagement", "gamification"],
            )
        
        agent_config.system_prompt = ENGAGEMENT_SPECIALIST_PROMPT
        super().__init__(agent_config, llm_provider=llm_provider, session_id=session_id)
    
    async def process(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process an engagement request"""
        task_description = task.get("description", task.get("task", ""))
        
        response = await self.invoke(
            input_text=f"Suggest engagement strategies for: {task_description}",
            context=context,
        )
        
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "strategies": response.content,
            "confidence": response.confidence,
        }

