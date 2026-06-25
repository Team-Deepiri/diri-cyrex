"""REST API for Cyrex agent training — corrections, jobs, and status."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.logging_config import get_logger
from app.pipeline.contracts.models import Citation, LearningArtifact
from app.training.agent_training_service import get_agent_training_service

logger = get_logger("cyrex.training_api")

router = APIRouter(prefix="/training", tags=["training"])


class CorrectionInput(BaseModel):
    document_id: str
    field_name: str
    original_value: Any
    corrected_value: Any
    corrected_citation: Citation
    actor_id: str


class TrainingRunInput(BaseModel):
    experiment_id: str
    model_name: str
    fingerprint: str
    dataset_id: str
    dataset_version: str
    dataset_path: str
    content_hash: str
    row_count: int
    schema: Dict[str, Any] = Field(default_factory=dict)
    produced_by: str = "cyrex"
    priority: str = "batch"
    hyperparameters: Optional[Dict[str, Any]] = None
    tags: Optional[Dict[str, str]] = None


class AgentTrainingInput(BaseModel):
    correlation_id: str
    user_id: str
    learning_artifact_ids: List[str]
    adapter_target: str
    training_run: Optional[TrainingRunInput] = None


class FeedbackLoopInput(BaseModel):
    artifact_ids: List[str] = Field(..., min_length=1)


@router.post("/corrections", response_model=LearningArtifact)
async def submit_correction(body: CorrectionInput) -> LearningArtifact:
    """Buffer a human correction as a LearningArtifact."""
    service = get_agent_training_service()
    return service.submit_correction(
        document_id=body.document_id,
        field_name=body.field_name,
        original_value=body.original_value,
        corrected_value=body.corrected_value,
        corrected_citation=body.corrected_citation,
        actor_id=body.actor_id,
    )


@router.get("/corrections", response_model=List[LearningArtifact])
async def list_corrections(limit: int = 100) -> List[LearningArtifact]:
    """List buffered corrections pending export."""
    service = get_agent_training_service()
    return service.list_corrections(limit=limit)


@router.post("/jobs")
async def submit_training_job(body: TrainingRunInput) -> Dict[str, str]:
    """Submit a TrainingRunRequest to the Redis training-jobs stream."""
    try:
        from deepiri_modelkit.contracts.training import (
            DatasetManifest,
            TrainingPriority,
            TrainingRunRequest,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="deepiri-modelkit training contracts are not installed",
        ) from exc

    service = get_agent_training_service()
    priority = (
        TrainingPriority.LIVE
        if body.priority == "live"
        else TrainingPriority.BATCH
    )
    manifest = DatasetManifest(
        id=body.dataset_id,
        version=body.dataset_version,
        path=body.dataset_path,
        content_hash=body.content_hash,
        row_count=body.row_count,
        schema=body.schema,
        produced_by=body.produced_by,
    )
    request = TrainingRunRequest(
        experiment_id=body.experiment_id,
        model_name=body.model_name,
        fingerprint=body.fingerprint,
        dataset_manifest=manifest,
        priority=priority,
        hyperparameters=body.hyperparameters,
        tags=body.tags,
    )
    message_id = service.submit_training_run(request)
    return {"message_id": message_id, "experiment_id": body.experiment_id}


@router.post("/agent-jobs")
async def submit_agent_training_job(body: AgentTrainingInput) -> Dict[str, str]:
    """Submit an agent-initiated training job to Helox."""
    service = get_agent_training_service()
    training_run_request = None
    if body.training_run is not None:
        try:
            from deepiri_modelkit.contracts.training import (
                DatasetManifest,
                TrainingPriority,
                TrainingRunRequest,
            )
        except ImportError as exc:
            raise HTTPException(status_code=503, detail="modelkit not installed") from exc

        run = body.training_run
        priority = (
            TrainingPriority.LIVE
            if run.priority == "live"
            else TrainingPriority.BATCH
        )
        manifest = DatasetManifest(
            id=run.dataset_id,
            version=run.dataset_version,
            path=run.dataset_path,
            content_hash=run.content_hash,
            row_count=run.row_count,
            schema=run.schema,
            produced_by=run.produced_by,
        )
        training_run_request = TrainingRunRequest(
            experiment_id=run.experiment_id,
            model_name=run.model_name,
            fingerprint=run.fingerprint,
            dataset_manifest=manifest,
            priority=priority,
            hyperparameters=run.hyperparameters,
            tags=run.tags,
        )

    try:
        message_id = service.submit_agent_training(
            correlation_id=body.correlation_id,
            user_id=body.user_id,
            learning_artifact_ids=body.learning_artifact_ids,
            adapter_target=body.adapter_target,
            training_run_request=training_run_request,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"message_id": message_id, "correlation_id": body.correlation_id}


@router.post("/feedback-loop")
async def start_feedback_loop(body: FeedbackLoopInput) -> Dict[str, Any]:
    """Trigger the feedback-loop trainer for buffered artifact IDs."""
    service = get_agent_training_service()
    return service.start_feedback_loop(body.artifact_ids)


@router.get("/status/{experiment_id}")
async def get_training_status(
    experiment_id: str,
    last_id: str = "0",
) -> Dict[str, Any]:
    """Poll training-events for an experiment."""
    service = get_agent_training_service()
    events = await service.get_training_status(experiment_id, last_id=last_id)
    return {"experiment_id": experiment_id, "events": events, "count": len(events)}


@router.get("/health")
async def training_health() -> Dict[str, Any]:
    """Training subsystem readiness."""
    service = get_agent_training_service()
    return {
        "corrections_buffered": service.correction_trainer.size,
        "job_client_live": service.job_client.is_live,
        "status_monitor_live": service.status_monitor.is_live,
    }
