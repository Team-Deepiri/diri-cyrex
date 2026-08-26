"""MCP reckoning tool backed by the shared reckoning read port."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...pipeline.contracts.models import PredictionRecord
from ...pipeline.contracts.ports import ReckoningReadPort
from ..registry import McpToolDefinition, McpToolRegistry


class ReckoningRequest(BaseModel):
    document_id: str = Field(min_length=1)


class ReckoningToolResponse(BaseModel):
    document_id: str
    records: list[PredictionRecord]


def register_reckoning_tool(
    registry: McpToolRegistry,
    store: ReckoningReadPort,
    *,
    timeout_seconds: float = 10.0,
) -> None:
    async def get_reckoning(
        request: ReckoningRequest,
        _context: Any,
    ) -> ReckoningToolResponse:
        records = await store.get_reckoning(request.document_id)
        return ReckoningToolResponse(
            document_id=request.document_id,
            records=records,
        )

    registry.register(
        McpToolDefinition(
            name="cyrex.reckoning.get",
            description="Return dead-reckoning prediction records for a document.",
            input_model=ReckoningRequest,
            output_model=ReckoningToolResponse,
            handler=get_reckoning,
            timeout_seconds=timeout_seconds,
        )
    )
