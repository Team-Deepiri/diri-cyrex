"""MCP adapter for the Voice of the Document service."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from ...pipeline.contracts.models import PersonaScope
from ..registry import McpToolDefinition, McpToolRegistry


class VoiceQueryResponse(BaseModel):
    """Transport-neutral voice response returned by the service port."""

    confessed: bool
    spans: list[dict[str, Any]] = Field(default_factory=list)
    gaps: list[dict[str, Any]] | None = None


class VoiceQueryPort(Protocol):
    async def query(
        self,
        document_id: str,
        question: str,
        persona_scope: PersonaScope,
    ) -> VoiceQueryResponse:
        ...


class VoiceQueryRequest(BaseModel):
    document_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    persona_scope: PersonaScope = Field(default_factory=PersonaScope)


def register_voice_tool(
    registry: McpToolRegistry,
    service: VoiceQueryPort,
    *,
    timeout_seconds: float = 30.0,
) -> None:
    async def query_voice(
        request: VoiceQueryRequest,
        _context: Any,
    ) -> VoiceQueryResponse:
        return await service.query(
            request.document_id,
            request.question,
            request.persona_scope,
        )

    registry.register(
        McpToolDefinition(
            name="cyrex.voice.query",
            description="Answer a document question using cited witness spans.",
            input_model=VoiceQueryRequest,
            output_model=VoiceQueryResponse,
            handler=query_voice,
            timeout_seconds=timeout_seconds,
        )
    )
