"""FastMCP entry point for the Cyrex MCP host."""

from __future__ import annotations

import os
from functools import wraps
from typing import Any, Callable, ParamSpec

from mcp.server.fastmcp import FastMCP

from .composition import create_default_host
from .errors import McpToolError
from .host import McpToolHost

P = ParamSpec("P")


def _bind_host_tool(
    server: Any,
    host: McpToolHost,
    *,
    name: str,
    payload_builder: Callable[P, dict[str, Any]],
) -> None:
    """Expose a typed payload builder while centralizing host invocation."""

    @wraps(payload_builder)
    async def bound_tool(*args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
        return await _invoke_for_mcp(
            host,
            name,
            payload_builder(*args, **kwargs),
        )

    server.tool(name=name)(bound_tool)


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
    tool_host = host or create_default_host()
    server = FastMCP("cyrex")

    def artifact_get(artifact_id: str) -> dict[str, Any]:
        return {"artifact_id": artifact_id}

    def artifact_list(document_id: str) -> dict[str, Any]:
        return {"document_id": document_id}

    def pressure_get_map(document_id: str | None = None) -> dict[str, Any]:
        return {"document_id": document_id}

    def reckoning_get(document_id: str) -> dict[str, Any]:
        return {"document_id": document_id}

    bindings = [
        ("cyrex.artifacts.get", artifact_get),
        ("cyrex.artifacts.list", artifact_list),
        ("cyrex.pressure.get_map", pressure_get_map),
        ("cyrex.reckoning.get", reckoning_get),
    ]

    # Voice and RAG are optional until their application services are wired
    # into composition.py. An injected host exposes them without making the
    # default read-only host depend on route globals or AI initialization.
    if "cyrex.voice.query" in tool_host.registry.names():

        def voice_query(document_id: str, question: str) -> dict[str, Any]:
            return {"document_id": document_id, "question": question}

        bindings.append(("cyrex.voice.query", voice_query))

    if "cyrex.rag.query" in tool_host.registry.names():

        def rag_query(
            query: str,
            top_k: int = 10,
            task_type: str | None = None,
            rerank: bool = True,
        ) -> dict[str, Any]:
            return {
                "query": query,
                "top_k": top_k,
                "task_type": task_type,
                "rerank": rerank,
            }

        bindings.append(("cyrex.rag.query", rag_query))

    for name, payload_builder in bindings:
        _bind_host_tool(
            server,
            tool_host,
            name=name,
            payload_builder=payload_builder,
        )

    return server


def main() -> None:
    create_server().run(transport=os.getenv("CYREX_MCP_TRANSPORT", "stdio"))


if __name__ == "__main__":
    main()
