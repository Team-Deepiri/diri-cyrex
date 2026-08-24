"""LangChain tools exposing Elkedel eyes + memory to Cyrex agents."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.integrations.elkedel import get_elkedel_client


class _Empty(BaseModel):
    pass


class _Query(BaseModel):
    q: str = Field(default="", description="Text query for visual recall")
    top_k: int = Field(default=5, ge=1, le=20)


class _Since(BaseModel):
    since_ms: int = Field(default=0, ge=0)


def build_elkedel_agent_tools() -> List[StructuredTool]:
    """Namespaced elkedel.* tools for WorkflowOrchestrator / playground."""

    async def _eyes_scene(top_k: int = 20) -> Dict[str, Any]:
        return await get_elkedel_client().eyes_scene(top_k=top_k)

    async def _eyes_where(q: str = "", top_k: int = 5) -> Dict[str, Any]:
        return await get_elkedel_client().eyes_where(query=q, top_k=top_k)

    async def _eyes_status() -> Dict[str, Any]:
        return await get_elkedel_client().eyes_status()

    async def _eyes_start() -> Dict[str, Any]:
        return await get_elkedel_client().eyes_start()

    async def _what_changed(since_ms: int = 0) -> Dict[str, Any]:
        return await get_elkedel_client().what_changed(since_ms)

    async def _stats() -> Dict[str, Any]:
        return await get_elkedel_client().stats()

    return [
        StructuredTool.from_function(
            coroutine=_eyes_scene,
            name="elkedel.eyes_scene",
            description="Active visual identities Elkedel eyes see right now.",
        ),
        StructuredTool.from_function(
            coroutine=_eyes_where,
            name="elkedel.eyes_where",
            description="Text query against episodic visual memory (where is X?).",
            args_schema=_Query,
        ),
        StructuredTool.from_function(
            coroutine=_eyes_status,
            name="elkedel.eyes_status",
            description="Pipeline status: camera, FPS, active traces.",
        ),
        StructuredTool.from_function(
            coroutine=_eyes_start,
            name="elkedel.eyes_start",
            description="Start live camera → detect → memory loop.",
        ),
        StructuredTool.from_function(
            coroutine=_what_changed,
            name="elkedel.what_changed",
            description="Scene diff since timestamp (new/vanished/moved identities).",
            args_schema=_Since,
        ),
        StructuredTool.from_function(
            coroutine=_stats,
            name="elkedel.stats",
            description="Memory store stats (traces, observations).",
        ),
    ]


async def register_elkedel_tools(tool_registry) -> int:
    from app.core.tool_registry import ToolCategory

    n = 0
    for tool in build_elkedel_agent_tools():
        tool_registry.register_tool(tool, category=ToolCategory.CUSTOM)
        n += 1
    return n
