"""
Agent Prompt Templates
Pre-defined prompt templates for different agent roles
"""
from .task_decomposer_prompts import TASK_DECOMPOSER_PROMPT
from .time_optimizer_prompts import TIME_OPTIMIZER_PROMPT
from .creative_sparker_prompts import CREATIVE_SPARKER_PROMPT
from .quality_assurance_prompts import QUALITY_ASSURANCE_PROMPT
from .engagement_specialist_prompts import ENGAGEMENT_SPECIALIST_PROMPT

# Vendor Fraud Detection Prompts
from .vendor_fraud_prompts import (
    VENDOR_FRAUD_SYSTEM_PROMPT,
    PROPERTY_MANAGEMENT_PROMPT,
    CORPORATE_PROCUREMENT_PROMPT,
    INSURANCE_PC_PROMPT,
    GENERAL_CONTRACTORS_PROMPT,
    RETAIL_ECOMMERCE_PROMPT,
    LAW_FIRMS_PROMPT,
    INVOICE_ANALYSIS_PROMPT,
    VENDOR_INTELLIGENCE_PROMPT,
    PRICING_COMPARISON_PROMPT,
    INDUSTRY_PROMPTS,
    get_industry_prompt,
    get_invoice_analysis_prompt,
    get_vendor_intelligence_prompt,
    get_pricing_comparison_prompt,
)

# ReAct Agent Prompts
from .react_agent_prompts import (
    REACT_AGENT_SYSTEM_PROMPT,
    REACT_CONVERSATIONAL_PROMPT,
    REACT_MINIMAL_PROMPT,
)

__all__ = [
    # General Agent Prompts
    'TASK_DECOMPOSER_PROMPT',
    'TIME_OPTIMIZER_PROMPT',
    'CREATIVE_SPARKER_PROMPT',
    'QUALITY_ASSURANCE_PROMPT',
    'ENGAGEMENT_SPECIALIST_PROMPT',
    # Vendor Fraud Detection Prompts
    'VENDOR_FRAUD_SYSTEM_PROMPT',
    'PROPERTY_MANAGEMENT_PROMPT',
    'CORPORATE_PROCUREMENT_PROMPT',
    'INSURANCE_PC_PROMPT',
    'GENERAL_CONTRACTORS_PROMPT',
    'RETAIL_ECOMMERCE_PROMPT',
    'LAW_FIRMS_PROMPT',
    'INVOICE_ANALYSIS_PROMPT',
    'VENDOR_INTELLIGENCE_PROMPT',
    'PRICING_COMPARISON_PROMPT',
    'INDUSTRY_PROMPTS',
    'get_industry_prompt',
    'get_invoice_analysis_prompt',
    'get_vendor_intelligence_prompt',
    'get_pricing_comparison_prompt',
    # ReAct Agent Prompts
    'REACT_AGENT_SYSTEM_PROMPT',
    'REACT_CONVERSATIONAL_PROMPT',
    'REACT_MINIMAL_PROMPT',
]

