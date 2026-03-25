"""
Tool Schema Optimizer

Reduces tool schema payload size sent to Ollama by:
- Removing verbose descriptions
- Simplifying parameter schemas
- Stripping unnecessary metadata
"""
from typing import List, Any, Dict
from ..logging_config import get_logger

logger = get_logger("cyrex.tool_schema_optimizer")


def optimize_tool_schema(tool: Any) -> Any:
    """
    Optimize a single tool's schema to reduce payload size.
    
    For StructuredTool, we can't modify the schema directly, but we can
    create a lightweight wrapper that reduces the description length.
    """
    # For now, we'll create minimal tool descriptions
    # The actual schema optimization happens at the LangChain level
    # by using shorter descriptions
    
    if hasattr(tool, 'description'):
        # Truncate long descriptions (keep first 100 chars)
        original_desc = tool.description
        if len(original_desc) > 100:
            tool.description = original_desc[:97] + "..."
            logger.debug(f"Truncated tool {tool.name} description: {len(original_desc)} -> {len(tool.description)}")
    
    return tool


def optimize_tool_list(tools: List[Any]) -> List[Any]:
    """Optimize a list of tools"""
    optimized = []
    for tool in tools:
        optimized.append(optimize_tool_schema(tool))
    return optimized

