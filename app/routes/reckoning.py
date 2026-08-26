"""Reckoning read API — dead reckoning records per document."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.pipeline.contracts.models import PredictionRecord, PredictionStatus
from app.pipeline.contracts.ports import ReckoningReadPort


async def get_reckoning_read_model() -> ReckoningReadPort:
    raise RuntimeError("Reckoning read model dependency is not configured")


class ReckoningResponse(BaseModel):
    document_id: str
    records: list[PredictionRecord]
    anomalous_count: int
    novel_count: int


router = APIRouter(prefix="/api/v1/reckoning", tags=["reckoning"])


@router.get("/{document_id}", response_model=ReckoningResponse)
async def get_document_reckoning(
    document_id: str,
    store: Annotated[ReckoningReadPort, Depends(get_reckoning_read_model)],
) -> ReckoningResponse:
    records = await store.get_reckoning(document_id)
    anomalous = sum(1 for r in records if r.status == PredictionStatus.ANOMALOUS)
    novel = sum(1 for r in records if r.status == PredictionStatus.NOVEL)
    return ReckoningResponse(
        document_id=document_id,
        records=records,
        anomalous_count=anomalous,
        novel_count=novel,
    )
