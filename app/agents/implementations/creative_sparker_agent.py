"""
Creative Sparker Agent Implementation
"""
from typing import Dict, Any
from ..base_agent import BaseAgent
from ..prompts.creative_sparker_prompts import CREATIVE_SPARKER_PROMPT
from ...core.types import AgentConfig, AgentRole

class CreativeSparkerAgent(BaseAgent):
    """Agent specialized in creative idea generation"""
    
    def __init__(self, agent_config: AgentConfig = None, llm_provider = None, session_id: str = None):
        if not agent_config:
            agent_config = AgentConfig(
                role=AgentRole.CREATIVE_SPARKER,
                name="Creative Sparker Agent",
                description="Generates innovative ideas and creative solutions",
                capabilities=["idea_generation", "creative_thinking", "innovation"],
            )
        
        agent_config.system_prompt = CREATIVE_SPARKER_PROMPT
        super().__init__(agent_config, llm_provider=llm_provider, session_id=session_id)
    
    async def process(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a creative ideation request"""
        task_description = task.get("description", task.get("task", ""))
        
        response = await self.invoke(
            input_text=f"Generate creative ideas for: {task_description}",
            context=context,
        )
        
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "ideas": response.content,
            "confidence": response.confidence,
        }

