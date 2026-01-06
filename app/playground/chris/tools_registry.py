from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from app.core.tool_registry import get_tool_registry
from .tools import register_chris_tools

REQ_TOOL_NAMES: List[str] = [
    "help_get_product",
]

_registered = False
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str

def ensure_tools_registered() -> None:
    """
    Ensure tools are registered into Cyrex's global tool registry.
    """
    register_chris_tools()
    _registered = True

def get_required_tool_specs() -> List[ToolSpec]:
    ensure_tools_registered()
    registry = get_tool_registry()

    specs: List[ToolSpec] = []
    for tool_name in REQ_TOOL_NAMES:
        tool = registry.get_tool(tool_name)
        if tool is None:
            raise RuntimeError(
                f"Tool '{tool_name}' is not registered. "
                f"Make sure register_chris_tools() registers it."
            )
        specs.append(ToolSpec(name=tool.name, description=tool.description))
    return specs

def tools_prompt_block() -> str:
    """
    Text block you inject into ROUTER_PROMPT so the model can choose a tool.
    """
    tools = get_required_tool_specs()
    return "\n".join(f"- {t.name}: {t.description}" for t in tools)

def execute_tool(tool_name: str, user_input: str) -> Any:
    """
    Execute a registered tool by name.
    Tries string input first (common for Tool(func=...)),
    falls back to {"input": ...} if needed.
    """
    ensure_tools_registered()

    if tool_name not in REQ_TOOL_NAMES:
        raise ValueError(f"Tool '{tool_name}' is not allowed for this agent.")
    
    registry = get_tool_registry()
    return registry.execute_tool(tool_name, user_input)
