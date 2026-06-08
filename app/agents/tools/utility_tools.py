"""
Utility Tools for Agents
General utility tools for agents, delegated to diri-agent-toolbox so behavior
matches ComprehensiveAPITools (no duplicated implementations).
"""
from typing import Any, Dict

from diri_agent_toolbox.data import json_format, json_parse, safe_calculate

from ...logging_config import get_logger

logger = get_logger("cyrex.agent.tools.utility")


async def register_utility_tools(agent):
    """Register utility tools with an agent (delegated to diri-agent-toolbox)."""

    async def format_json(data: Dict[str, Any]) -> str:
        """Format data as JSON (diri-agent-toolbox json_format)."""
        result = await json_format(data)
        if not result.success:
            return f"json_format error: {result.error}"
        return result.result

    async def parse_json(json_str: str) -> Any:
        """Parse JSON string (diri-agent-toolbox json_parse)."""
        result = await json_parse(json_str)
        if not result.success:
            raise ValueError(f"json_parse error: {result.error}")
        return result.result

    def calculate(expression: str):
        """Safe math via diri-agent-toolbox (AST-based; not eval).

        Returns a clear error string on invalid input rather than a misleading
        0, so callers can distinguish a real result from a failure.
        """
        try:
            return float(safe_calculate(expression))
        except Exception as e:
            return f"calculation error: {e}"

    agent.register_tool("format_json", format_json, "Format data as JSON string")
    agent.register_tool("parse_json", parse_json, "Parse JSON string to dictionary")
    agent.register_tool("calculate", calculate, "Calculate a mathematical expression")
