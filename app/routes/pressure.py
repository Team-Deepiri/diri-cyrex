"""Pressure map read API."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.pipeline.contracts.models import PressureCell
from app.pipeline.contracts.ports import PressureReadModelPort


class PressureMapResponse(BaseModel):
    document_id: str
    cells: list[PressureCell]
    fault_zone_count: int
    max_score: float


async def get_pressure_read_model() -> PressureReadModelPort:
    """Application-composition hook for the pressure read-model adapter."""
    raise RuntimeError("Pressure read model dependency is not configured")


router = APIRouter(prefix="/api/v1/pressure", tags=["pressure"])


@router.get("/{document_id}", response_model=PressureMapResponse)
async def get_document_pressure(
    document_id: str,
    store: Annotated[PressureReadModelPort, Depends(get_pressure_read_model)],
) -> PressureMapResponse:
    cells = await store.get_pressure(document_id)
    return PressureMapResponse(
        document_id=document_id,
        cells=cells,
        fault_zone_count=sum(cell.is_fault_zone for cell in cells),
        max_score=max((cell.score for cell in cells), default=0.0),
    )
