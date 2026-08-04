"""Composition helpers for the runnable Cyrex MCP read-model slice."""

from __future__ import annotations

from ..pipeline.contracts.ports import (
    ArtifactStorePort,
    PressureReadModelPort,
    ReckoningReadPort,
)
from .host import McpToolHost
from .registry import McpToolRegistry
from .tools.artifacts import register_artifact_tools
from .tools.pressure import register_pressure_tool
from .tools.rag import RagQueryPort, register_rag_tool
from .tools.reckoning import register_reckoning_tool
from .tools.voice import VoiceQueryPort, register_voice_tool


def create_default_host(
    *,
    artifact_store: ArtifactStorePort | None = None,
    pressure_store: PressureReadModelPort | None = None,
    reckoning_store: ReckoningReadPort | None = None,
    voice_service: VoiceQueryPort | None = None,
    rag_service: RagQueryPort | None = None,
) -> McpToolHost:
    """Create a host from injected ports and optional AI service adapters.

    Concrete stores are supplied by the application composition root. Keeping
    them out of Track D prevents MCP from importing another track's
    implementation; tests can provide in-memory fakes through these ports.
    """
    registry = McpToolRegistry()
    if artifact_store is not None:
        register_artifact_tools(registry, artifact_store)
    if pressure_store is not None:
        register_pressure_tool(registry, pressure_store)
    if reckoning_store is not None:
        register_reckoning_tool(registry, reckoning_store)
    if voice_service is not None:
        register_voice_tool(registry, voice_service)
    if rag_service is not None:
        register_rag_tool(registry, rag_service)
    return McpToolHost(registry)
