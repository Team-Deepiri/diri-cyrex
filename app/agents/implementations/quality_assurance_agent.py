"""
Quality Assurance Agent Implementation
"""
from typing import Dict, Any
from ..base_agent import BaseAgent
from ..prompts.quality_assurance_prompts import QUALITY_ASSURANCE_PROMPT
from ...core.types import AgentConfig, AgentRole

class QualityAssuranceAgent(BaseAgent):
    """Agent specialized in quality assurance"""
    
    def __init__(self, agent_config: AgentConfig = None, llm_provider = None, session_id: str = None):
        if not agent_config:
            agent_config = AgentConfig(
                role=AgentRole.QUALITY_ASSURANCE,
                name="Quality Assurance Agent",
                description="Ensures high-quality outputs and catches errors",
                capabilities=["quality_review", "error_detection", "validation"],
            )
        
        agent_config.system_prompt = QUALITY_ASSURANCE_PROMPT
        super().__init__(agent_config, llm_provider=llm_provider, session_id=session_id)
    
    async def process(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a quality assurance request"""
        content = task.get("content", task.get("output", ""))
        
        response = await self.invoke(
            input_text=f"Review this for quality: {content}",
            context=context,
        )
        
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "review": response.content,
            "confidence": response.confidence,
        }

