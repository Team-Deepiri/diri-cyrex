from __future__ import annotations
from typing import Any, Dict
from .tools_registry import ensure_tools_registered
from app.integrations.local_llm import get_local_llm

_INITIALIZED = False
_AGENT_INFO: Dict[str, Any] = {}


def init_agent() -> Dict[str, Any]:
    """
    One-time init for the playground agent.
    - registers tools into Cyrex tool registry
    - warms up / verifies LLM provider config (Ollama via container)
    - returns agent metadata (useful for UI / debugging)
    """
    global _INITIALIZED, _AGENT_INFO
    if _INITIALIZED:
        return _AGENT_INFO

    ensure_tools_registered()

    provider = get_local_llm()  # uses Cyrex settings & routes to ollama container

    _AGENT_INFO = {
        "name": "chris_playground_agent",
        "description": "Routes buy-intent to help_get_product tool; otherwise answers normally.",
        "model": getattr(provider, "llm", None) or "local_llm",
        "tools": ["help_get_product"],
        "state": "AgentState (playground/chris/state.py)",
    }

    _INITIALIZED = True
    return _AGENT_INFO