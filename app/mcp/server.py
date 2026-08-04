"""FastMCP entry point for the Cyrex MCP host."""

from __future__ import annotations

import os
from typing import Any

from .composition import create_default_host
from .errors import McpToolError
from .host import McpToolHost


async def _invoke_for_mcp(
    host: McpToolHost,
    tool_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    try:
        return await host.invoke(tool_name, payload)
    except McpToolError as exc:
        return exc.as_dict()


def create_server(host: McpToolHost | None = None) -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The MCP host requires the 'mcp' package. Install project dependencies first."
        ) from exc

    tool_host = host or create_default_host()
    server = FastMCP("cyrex")

    @server.tool(name="cyrex.artifacts.get")
    async def artifact_get(artifact_id: str) -> dict[str, Any]:
        return await _invoke_for_mcp(
            tool_host,
            "cyrex.artifacts.get",
            {"artifact_id": artifact_id},
        )

    @server.tool(name="cyrex.artifacts.list")
    async def artifact_list(document_id: str) -> dict[str, Any]:
        return await _invoke_for_mcp(
            tool_host,
            "cyrex.artifacts.list",
            {"document_id": document_id},
        )

    # Voice and RAG are optional until their application services are wired
    # into composition.py. An injected host exposes them without making the
    # default read-only host depend on route globals or AI initialization.
    if "cyrex.voice.query" in tool_host.registry.names():

        @server.tool(name="cyrex.voice.query")
        async def voice_query(
            document_id: str,
            question: str,
        ) -> dict[str, Any]:
            return await _invoke_for_mcp(
                tool_host,
                "cyrex.voice.query",
                {"document_id": document_id, "question": question},
            )

    if "cyrex.rag.query" in tool_host.registry.names():

        @server.tool(name="cyrex.rag.query")
        async def rag_query(
            query: str,
            top_k: int = 10,
            task_type: str | None = None,
            rerank: bool = True,
        ) -> dict[str, Any]:
            return await _invoke_for_mcp(
                tool_host,
                "cyrex.rag.query",
                {
                    "query": query,
                    "top_k": top_k,
                    "task_type": task_type,
                    "rerank": rerank,
                },
            )

    @server.tool(name="cyrex.pressure.get_map")
    async def pressure_get_map(document_id: str | None = None) -> dict[str, Any]:
        return await _invoke_for_mcp(
            tool_host,
            "cyrex.pressure.get_map",
            {"document_id": document_id},
        )

    @server.tool(name="cyrex.reckoning.get")
    async def reckoning_get(document_id: str) -> dict[str, Any]:
        return await _invoke_for_mcp(
            tool_host,
            "cyrex.reckoning.get",
            {"document_id": document_id},
        )

    return server


def main() -> None:
    create_server().run(transport=os.getenv("CYREX_MCP_TRANSPORT", "stdio"))


if __name__ == "__main__":
    main()
