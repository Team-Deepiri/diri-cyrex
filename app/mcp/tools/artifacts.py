"""MCP artifact tools over the shared ArtifactStorePort seam."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...pipeline.contracts.models import ArtifactBundle
from ...pipeline.contracts.ports import ArtifactStorePort
from ..errors import McpToolError
from ..registry import McpToolDefinition, McpToolRegistry


class ArtifactGetRequest(BaseModel):
    artifact_id: str = Field(min_length=1)


class ArtifactGetResponse(BaseModel):
    artifact: ArtifactBundle


class ArtifactListRequest(BaseModel):
    document_id: str = Field(min_length=1)


class ArtifactListResponse(BaseModel):
    document_id: str
    artifacts: list[ArtifactBundle]


def register_artifact_tools(
    registry: McpToolRegistry,
    store: ArtifactStorePort,
    *,
    timeout_seconds: float = 10.0,
) -> None:
    async def get_artifact(request: ArtifactGetRequest, _context: Any) -> ArtifactGetResponse:
        artifact = await store.get(request.artifact_id)
        if artifact is None:
            raise McpToolError(
                "not_found",
                "Artifact not found",
                details={"artifact_id": request.artifact_id},
            )
        return ArtifactGetResponse(artifact=artifact)

    async def list_artifacts(
        request: ArtifactListRequest,
        _context: Any,
    ) -> ArtifactListResponse:
        artifacts = await store.list_by_document(request.document_id)
        return ArtifactListResponse(
            document_id=request.document_id,
            artifacts=artifacts,
        )

    registry.register(
        McpToolDefinition(
            name="cyrex.artifacts.get",
            description="Fetch one non-deleted artifact bundle by ID.",
            input_model=ArtifactGetRequest,
            output_model=ArtifactGetResponse,
            handler=get_artifact,
            timeout_seconds=timeout_seconds,
        )
    )
    registry.register(
        McpToolDefinition(
            name="cyrex.artifacts.list",
            description="List non-deleted artifact bundles for a document.",
            input_model=ArtifactListRequest,
            output_model=ArtifactListResponse,
            handler=list_artifacts,
            timeout_seconds=timeout_seconds,
        )
    )
