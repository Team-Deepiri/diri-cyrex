"""Composition helpers for the runnable Cyrex MCP read-model slice."""

from __future__ import annotations

from ..pipeline.registry.postgres_store import PostgresArtifactStore
from ..pipeline.registry.pressure_store import PostgresPressureStore
from ..pipeline.registry.reckoning_store import PostgresReckoningStore
from .host import McpToolHost
from .registry import McpToolRegistry
from .tools.artifacts import register_artifact_tools
from .tools.pressure import register_pressure_tool
from .tools.rag import RagQueryPort, register_rag_tool
from .tools.reckoning import register_reckoning_tool
from .tools.voice import VoiceQueryPort, register_voice_tool


def create_default_host(
    *,
    voice_service: VoiceQueryPort | None = None,
    rag_service: RagQueryPort | None = None,
) -> McpToolHost:
    """Create a production-backed host with optional AI service adapters.

    Voice and RAG are explicitly injected because their current application
    implementations are not yet stable service ports. This keeps MCP from
    importing route globals or silently exposing the existing voice stub.
    """
    registry = McpToolRegistry()
    register_artifact_tools(registry, PostgresArtifactStore())
    register_pressure_tool(registry, PostgresPressureStore())
    register_reckoning_tool(registry, PostgresReckoningStore())
    if voice_service is not None:
        register_voice_tool(registry, voice_service)
    if rag_service is not None:
        register_rag_tool(registry, rag_service)
    return McpToolHost(registry)
