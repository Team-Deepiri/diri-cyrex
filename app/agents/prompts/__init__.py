"""
Agent Prompt Templates
Pre-defined prompt templates for different agent roles
"""
from .task_decomposer_prompts import TASK_DECOMPOSER_PROMPT
from .time_optimizer_prompts import TIME_OPTIMIZER_PROMPT
from .creative_sparker_prompts import CREATIVE_SPARKER_PROMPT
from .quality_assurance_prompts import QUALITY_ASSURANCE_PROMPT
from .engagement_specialist_prompts import ENGAGEMENT_SPECIALIST_PROMPT

__all__ = [
    'TASK_DECOMPOSER_PROMPT',
    'TIME_OPTIMIZER_PROMPT',
    'CREATIVE_SPARKER_PROMPT',
    'QUALITY_ASSURANCE_PROMPT',
    'ENGAGEMENT_SPECIALIST_PROMPT',
]

