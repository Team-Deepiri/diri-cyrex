"""Dead reckoning read API."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.pipeline.contracts.models import PredictionRecord
from app.pipeline.contracts.ports import ReckoningReadPort


class ReckoningResponse(BaseModel):
    document_id: str
    records: list[PredictionRecord]
    anomalous_count: int
    novel_count: int


async def get_reckoning_read_model() -> ReckoningReadPort:
    raise RuntimeError("Reckoning read model dependency is not configured")


router = APIRouter(prefix="/api/v1/reckoning", tags=["reckoning"])


@router.get("/{document_id}", response_model=ReckoningResponse)
async def get_document_reckoning(
    document_id: str,
    store: Annotated[ReckoningReadPort, Depends(get_reckoning_read_model)],
) -> ReckoningResponse:
    records = await store.get_reckoning(document_id)
    return ReckoningResponse(
        document_id=document_id,
        records=records,
        anomalous_count=sum(r.status == "anomalous" for r in records),
        novel_count=sum(r.status == "novel" for r in records),
    )