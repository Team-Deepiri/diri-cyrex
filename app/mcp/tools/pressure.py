"""MCP pressure-map tool backed by the shared pressure read port."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...pipeline.contracts.models import PressureCell
from ...pipeline.contracts.ports import PressureReadModelPort
from ..registry import McpToolDefinition, McpToolRegistry


class PressureMapRequest(BaseModel):
    document_id: str | None = Field(default=None, min_length=1)


class PressureMapToolResponse(BaseModel):
    document_id: str | None
    cells: list[PressureCell]
    fault_zone_count: int
    max_score: float


def register_pressure_tool(
    registry: McpToolRegistry,
    store: PressureReadModelPort,
    *,
    timeout_seconds: float = 10.0,
) -> None:
    async def get_pressure(
        request: PressureMapRequest,
        _context: Any,
    ) -> PressureMapToolResponse:
        cells = await store.get_pressure(request.document_id)
        return PressureMapToolResponse(
            document_id=request.document_id,
            cells=cells,
            fault_zone_count=sum(cell.is_fault_zone for cell in cells),
            max_score=max((cell.score for cell in cells), default=0.0),
        )

    registry.register(
        McpToolDefinition(
            name="cyrex.pressure.get_map",
            description="Return epistemic pressure cells for one document or the corpus.",
            input_model=PressureMapRequest,
            output_model=PressureMapToolResponse,
            handler=get_pressure,
            timeout_seconds=timeout_seconds,
        )
    )
