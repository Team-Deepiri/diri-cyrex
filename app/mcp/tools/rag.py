"""MCP adapter for the Cyrex retrieval service."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from ..registry import McpToolDefinition, McpToolRegistry


class RagQueryRequest(BaseModel):
    query: str = Field(min_length=1)
    task_type: str | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    rerank: bool = True


class RagQueryResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]


class RagQueryPort(Protocol):
    async def query(
        self,
        query: str,
        *,
        task_type: str | None,
        top_k: int,
        rerank: bool,
    ) -> list[dict[str, Any]]:
        ...


def register_rag_tool(
    registry: McpToolRegistry,
    service: RagQueryPort,
    *,
    timeout_seconds: float = 30.0,
) -> None:
    async def query_rag(
        request: RagQueryRequest,
        _context: Any,
    ) -> RagQueryResponse:
        results = await service.query(
            request.query,
            task_type=request.task_type,
            top_k=request.top_k,
            rerank=request.rerank,
        )
        return RagQueryResponse(query=request.query, results=results)

    registry.register(
        McpToolDefinition(
            name="cyrex.rag.query",
            description="Retrieve relevant Cyrex knowledge for a query.",
            input_model=RagQueryRequest,
            output_model=RagQueryResponse,
            handler=query_rag,
            timeout_seconds=timeout_seconds,
        )
    )
