from __future__ import annotations

import pytest

from app.mcp.host import McpToolHost
from app.mcp.registry import McpToolRegistry
from app.mcp.server import create_server
from app.mcp.tools.pressure import register_pressure_tool


class EmptyPressureStore:
    async def get_pressure(self, document_id: str | None = None):
        return []


def make_host() -> McpToolHost:
    registry = McpToolRegistry()
    register_pressure_tool(registry, EmptyPressureStore())
    return McpToolHost(registry)


def test_fastmcp_server_registers_namespaced_tools() -> None:
    server = create_server()

    assert sorted(server._tool_manager._tools) == [
        "cyrex.artifacts.get",
        "cyrex.artifacts.list",
        "cyrex.pressure.get_map",
        "cyrex.reckoning.get",
    ]
    assert "document_id" in server._tool_manager._tools[
        "cyrex.pressure.get_map"
    ].parameters["properties"]


@pytest.mark.asyncio
async def test_fastmcp_wrapper_returns_structured_host_errors() -> None:
    server = create_server(make_host())
    result = await server._tool_manager.call_tool(
        "cyrex.pressure.get_map",
        {"document_id": ""},
    )

    # The FastMCP tool manager preserves the host's JSON-shaped error result.
    assert result["error"]["code"] == "invalid_input"
