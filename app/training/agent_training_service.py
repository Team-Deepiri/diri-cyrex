"""Agent training service composing corrections, Helox jobs, and training-orchestrator."""
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.logging_config import get_logger
from app.pipeline.contracts.models import Citation, LearningArtifact
from app.pipeline.registry.correction_store import SqliteCorrectionStore
from app.training.correction_trainer import CorrectionTrainer
from app.training.helox_job_client import HeloxJobClient
from app.training.training_status import TrainingStatusMonitor

logger = get_logger("cyrex.training.agent_service")

try:
    from deepiri_training_orchestrator import (
        FeedbackLoopTrainer,
        LiveFineTuneConfig,
        ReproducibilityController,
        TrainingOrchestrator,
        corrections_to_manifest,
    )

    _ORCH_AVAILABLE = True
except ImportError:
    _ORCH_AVAILABLE = False
    LiveFineTuneConfig = None  # type: ignore[misc, assignment]
    corrections_to_manifest = None  # type: ignore[misc, assignment]

    class TrainingOrchestrator:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            logger.warning("TrainingOrchestrator stub — install deepiri-training-orchestrator")

    class FeedbackLoopTrainer:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

try:
    from deepiri_modelkit.contracts.training import (
        AgentTrainingJob,
        DatasetManifest,
        TrainingPriority,
        TrainingRunRequest,
    )
except ImportError:
    TrainingRunRequest = None  # type: ignore[misc, assignment]
    AgentTrainingJob = None  # type: ignore[misc, assignment]
    DatasetManifest = None  # type: ignore[misc, assignment]
    TrainingPriority = None  # type: ignore[misc, assignment]


class AgentTrainingService:
    """Coordinates correction buffering, job submission, and training status."""

    def __init__(
        self,
        correction_trainer: Optional[CorrectionTrainer] = None,
        job_client: Optional[HeloxJobClient] = None,
        status_monitor: Optional[TrainingStatusMonitor] = None,
        correction_store: Optional[SqliteCorrectionStore] = None,
    ) -> None:
        self.correction_store = correction_store or SqliteCorrectionStore(
            os.getenv("CYREX_CORRECTIONS_DB", "cyrex_corrections.db")
        )
        self.correction_trainer = correction_trainer or CorrectionTrainer()
        self.job_client = job_client or HeloxJobClient()
        self.status_monitor = status_monitor or TrainingStatusMonitor()
        self._feedback_trainer: Optional[FeedbackLoopTrainer] = None
        if _ORCH_AVAILABLE and LiveFineTuneConfig is not None:
            cfg = LiveFineTuneConfig(min_examples=2, max_steps=50)
            repro = ReproducibilityController(seed=cfg.seed)
            repro.set_seeds()
            orch = TrainingOrchestrator(
                {"lr": 1e-4}, reproducibility=repro, max_steps=cfg.max_steps
            )
            self._feedback_trainer = FeedbackLoopTrainer(orch, live_config=cfg)

    def submit_correction(
        self,
        document_id: str,
        field_name: str,
        original_value: Any,
        corrected_value: Any,
        corrected_citation: Citation,
        actor_id: str,
    ) -> LearningArtifact:
        artifact = LearningArtifact(
            document_id=document_id,
            field_name=field_name,
            original_value=original_value,
            corrected_value=corrected_value,
            corrected_citation=corrected_citation,
            actor_id=actor_id,
        )
        self.correction_trainer.buffer(artifact)
        return artifact

    def list_corrections(self, limit: int = 100) -> List[LearningArtifact]:
        return self.correction_trainer.peek(limit=limit)

    def submit_training_run(self, request: TrainingRunRequest) -> str:
        return self.job_client.submit(request)

    def submit_agent_training(
        self,
        correlation_id: str,
        user_id: str,
        learning_artifact_ids: List[str],
        adapter_target: str,
        training_run_request: Optional[TrainingRunRequest] = None,
    ) -> str:
        if AgentTrainingJob is None:
            raise RuntimeError("deepiri-modelkit training contracts are not available")
        job = AgentTrainingJob(
            correlation_id=correlation_id,
            user_id=user_id,
            learning_artifact_ids=learning_artifact_ids,
            adapter_target=adapter_target,
            training_run_request=training_run_request,
        )
        return self.job_client.submit_agent_job(job)

    def start_feedback_loop(self, artifact_ids: List[str]) -> Dict[str, Any]:
        if not _ORCH_AVAILABLE or corrections_to_manifest is None or TrainingRunRequest is None:
            return {
                "status": "stub",
                "message": "training-orchestrator not installed",
                "correlation_id": str(uuid4()),
            }

        examples: List[Dict[str, Any]] = []
        for aid in artifact_ids:
            stored = self.correction_store.get_by_id(aid)
            if stored:
                examples.append(
                    {
                        "text": str(stored.corrected_value),
                        "artifact_id": stored.artifact_id,
                        "field_name": stored.field_name,
                        "document_id": stored.document_id,
                    }
                )
            else:
                for artifact in self.correction_trainer.peek(1000):
                    if artifact.artifact_id == aid:
                        examples.append(
                            {
                                "text": str(artifact.corrected_value),
                                "artifact_id": artifact.artifact_id,
                                "field_name": artifact.field_name,
                                "document_id": artifact.document_id,
                            }
                        )
                        break

        if not examples:
            return {
                "status": "buffered",
                "artifact_ids": artifact_ids,
                "correlation_id": str(uuid4()),
            }

        output_dir = tempfile.mkdtemp(prefix="cyrex-feedback-")
        provenance = corrections_to_manifest(examples, output_dir)

        manifest = DatasetManifest(
            id=provenance.dataset_id,
            version=provenance.version,
            path=provenance.path,
            content_hash=provenance.content_hash,
            row_count=provenance.row_count,
            schema={"fields": {"text": ["str"]}},
            produced_by=provenance.produced_by,
        )
        correlation_id = str(uuid4())
        request = TrainingRunRequest(
            experiment_id=f"cyrex-feedback-{correlation_id[:8]}",
            model_name="cyrex-live-adapter",
            fingerprint=correlation_id,
            dataset_manifest=manifest,
            priority=TrainingPriority.LIVE,
            hyperparameters={"adapter_target": "live", "artifact_ids": artifact_ids},
        )
        job_id = self.job_client.submit(request)
        return {
            "status": "enqueued",
            "correlation_id": correlation_id,
            "experiment_id": request.experiment_id,
            "job_id": job_id,
            "artifact_ids": artifact_ids,
        }

    async def get_training_status(
        self,
        experiment_id: str,
        last_id: str = "0",
    ) -> List[Dict[str, Any]]:
        return await self.status_monitor.poll_events(
            experiment_id=experiment_id,
            last_id=last_id,
        )


_service: Optional[AgentTrainingService] = None


def get_agent_training_service() -> AgentTrainingService:
    global _service
    if _service is None:
        _service = AgentTrainingService()
    return _service
