"""
Utility Tools for Agents
General utility tools for agents
"""
import json
from typing import Dict, Any

from diri_agent_toolbox.data import safe_calculate

from ...logging_config import get_logger

logger = get_logger("cyrex.agent.tools.utility")


async def register_utility_tools(agent):
    """Register utility tools with an agent"""
    
    def format_json(data: Dict[str, Any]) -> str:
        """Format data as JSON (aligned with diri-agent-toolbox json_format)."""
        return json.dumps(data, indent=2, default=str)
    
    def parse_json(json_str: str) -> Dict[str, Any]:
        """Parse JSON string"""
        return json.loads(json_str)
    
    def calculate(expression: str) -> float:
        """Safe math via diri-agent-toolbox (AST-based; not eval)."""
        try:
            return float(safe_calculate(expression))
        except Exception:
            return 0.0
    
    agent.register_tool("format_json", format_json, "Format data as JSON string")
    agent.register_tool("parse_json", parse_json, "Parse JSON string to dictionary")
    agent.register_tool("calculate", calculate, "Calculate a mathematical expression")
