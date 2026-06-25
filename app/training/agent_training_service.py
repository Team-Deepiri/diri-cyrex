"""Skeleton agent training service composing corrections, Helox jobs, and orchestrator."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.logging_config import get_logger
from app.pipeline.contracts.models import Citation, LearningArtifact
from app.training.correction_trainer import CorrectionTrainer
from app.training.helox_job_client import HeloxJobClient
from app.training.training_status import TrainingStatusMonitor

logger = get_logger("cyrex.training.agent_service")

try:
    from deepiri_training_orchestrator import TrainingOrchestrator
except ImportError:

    class TrainingOrchestrator:  # type: ignore[no-redef]
        """Stub when deepiri-training-orchestrator is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            logger.warning(
                "TrainingOrchestrator stub active — install deepiri-training-orchestrator"
            )

        def fit(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "stub", "message": "training-orchestrator not installed"}


try:
    from deepiri_training_orchestrator.feedback import FeedbackLoopTrainer
except ImportError:
    try:
        from deepiri_modelkit.training.feedback import FeedbackLoopTrainer
    except ImportError:

        class FeedbackLoopTrainer:  # type: ignore[no-redef]
            """Stub when FeedbackLoopTrainer is not yet published."""

            def __init__(self, orchestrator: Optional[TrainingOrchestrator] = None) -> None:
                self.orchestrator = orchestrator

            def enqueue_corrections(self, artifact_ids: List[str]) -> str:
                return f"stub-{artifact_ids[0] if artifact_ids else 'empty'}"

            def run(self) -> Dict[str, Any]:
                return {"status": "stub", "message": "FeedbackLoopTrainer not installed"}


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
    ) -> None:
        self.correction_trainer = correction_trainer or CorrectionTrainer()
        self.job_client = job_client or HeloxJobClient()
        self.status_monitor = status_monitor or TrainingStatusMonitor()
        self._orchestrator = TrainingOrchestrator()
        self._feedback_trainer = FeedbackLoopTrainer(orchestrator=self._orchestrator)

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
        correlation_id = self._feedback_trainer.enqueue_corrections(artifact_ids)
        result = self._feedback_trainer.run()
        result["correlation_id"] = correlation_id
        return result

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
