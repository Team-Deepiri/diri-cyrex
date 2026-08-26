"""Duel arena read API — latest DuelState for a document."""

from __future__ import annotations

from typing import Annotated, Any, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.pipeline.contracts.models import (
    ArtifactType,
    DuelResolutionStatus,
    DuelState,
    FieldDiscrepancy,
)
from app.pipeline.contracts.ports import ArtifactStorePort
from app.pipeline.stages.duel import to_arena_rows
from app.routes.artifacts import get_artifact_store

router = APIRouter(prefix="/api/v1/duel", tags=["duel"])


class DuelFieldRow(BaseModel):
    field_name: str
    agent_a_value: Any = None
    agent_b_value: Any = None
    agent_a_confidence: float | None = None
    agent_b_confidence: float | None = None
    is_disagreement: bool = False


class DuelArenaResponse(BaseModel):
    document_id: str
    artifact_id: str | None = None
    agent_a_id: str
    agent_b_id: str
    fields: List[DuelFieldRow]
    disagreements: List[FieldDiscrepancy]
    resolution_status: DuelResolutionStatus


def _duel_state_from_bundle(payload: dict[str, Any]) -> DuelState | None:
    raw = payload.get("duel_state")
    if raw is None:
        return None
    if isinstance(raw, DuelState):
        return raw
    return DuelState.model_validate(raw)


@router.get("/{document_id}", response_model=DuelArenaResponse)
async def get_document_duel(
    document_id: str,
    store: Annotated[ArtifactStorePort, Depends(get_artifact_store)],
) -> DuelArenaResponse:
    bundles = await store.list_by_document(document_id)
    duel_bundle = None
    state: DuelState | None = None
    for bundle in reversed(bundles):
        if bundle.artifact_type != ArtifactType.REASONING:
            continue
        state = _duel_state_from_bundle(bundle.payload or {})
        if state is not None:
            duel_bundle = bundle
            break

    if state is None or duel_bundle is None:
        raise HTTPException(
            status_code=404,
            detail="No duel artifact for document — upload and run pipeline first",
        )

    rows = to_arena_rows(state)
    return DuelArenaResponse(
        document_id=document_id,
        artifact_id=duel_bundle.artifact_id,
        agent_a_id=state.agent_a_id,
        agent_b_id=state.agent_b_id,
        fields=[DuelFieldRow.model_validate(row) for row in rows],
        disagreements=state.disagreements,
        resolution_status=state.resolution_status,
    )
