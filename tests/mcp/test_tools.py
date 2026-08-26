from __future__ import annotations

import pytest

from app.mcp.errors import McpToolError
from app.mcp.host import McpToolHost
from app.mcp.registry import McpToolRegistry
from app.mcp.tools.artifacts import register_artifact_tools
from app.mcp.tools.pressure import register_pressure_tool
from app.mcp.tools.rag import register_rag_tool
from app.mcp.tools.reckoning import register_reckoning_tool
from app.mcp.tools.voice import VoiceQueryResponse, register_voice_tool
from app.pipeline.contracts.models import PredictionRecord, PressureCell


class ArtifactStore:
    async def get(self, artifact_id):
        return None

    async def list_by_document(self, document_id):
        return []


class PressureStore:
    async def get_pressure(self, document_id=None):
        return [
            PressureCell(
                document_id=document_id or "doc-1",
                section_id="section-1",
                score=0.8,
                is_fault_zone=True,
            ),
            PressureCell(
                document_id=document_id or "doc-1",
                section_id="section-2",
                score=0.2,
                is_fault_zone=False,
            ),
        ]


class ReckoningStore:
    async def get_reckoning(self, document_id):
        return [PredictionRecord(field_name="rent", predicted_mean=4500)]


class VoiceService:
    async def query(self, document_id, question, persona_scope):
        return VoiceQueryResponse(
            confessed=False,
            spans=[{"citation_id": "cit-1", "quote": "Rent is due monthly."}],
        )


class RagService:
    async def query(self, query, *, task_type, top_k, rerank):
        return [{"text": "Rent is due monthly.", "score": 0.91}]


def host_with_all_read_tools():
    registry = McpToolRegistry()
    register_artifact_tools(registry, ArtifactStore())
    register_pressure_tool(registry, PressureStore())
    register_reckoning_tool(registry, ReckoningStore())
    register_voice_tool(registry, VoiceService())
    register_rag_tool(registry, RagService())
    return McpToolHost(registry)


@pytest.mark.asyncio
async def test_pressure_tool_returns_cells_and_aggregates():
    result = await host_with_all_read_tools().invoke(
        "cyrex.pressure.get_map",
        {"document_id": "doc-1"},
    )

    assert result["fault_zone_count"] == 1
    assert result["max_score"] == 0.8
    assert len(result["cells"]) == 2


@pytest.mark.asyncio
async def test_reckoning_tool_delegates_to_read_port():
    result = await host_with_all_read_tools().invoke(
        "cyrex.reckoning.get",
        {"document_id": "doc-1"},
    )

    assert result["document_id"] == "doc-1"
    assert result["records"][0]["field_name"] == "rent"


@pytest.mark.asyncio
async def test_artifact_tool_returns_not_found_error():
    with pytest.raises(McpToolError) as error:
        await host_with_all_read_tools().invoke(
            "cyrex.artifacts.get",
            {"artifact_id": "missing"},
        )

    assert error.value.code == "not_found"


@pytest.mark.asyncio
async def test_voice_tool_preserves_citation_response_shape():
    result = await host_with_all_read_tools().invoke(
        "cyrex.voice.query",
        {"document_id": "doc-1", "question": "What is the rent?"},
    )

    assert result["confessed"] is False
    assert result["spans"][0]["citation_id"] == "cit-1"


@pytest.mark.asyncio
async def test_rag_tool_passes_query_options_to_service():
    result = await host_with_all_read_tools().invoke(
        "cyrex.rag.query",
        {"query": "rent", "top_k": 5, "rerank": True},
    )

    assert result["query"] == "rent"
    assert result["results"][0]["score"] == 0.91
