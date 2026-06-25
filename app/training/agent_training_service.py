"""Agent training service composing corrections, Helox jobs, and training-orchestrator."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.logging_config import get_logger
from app.pipeline.contracts.models import Citation, LearningArtifact
from app.training.correction_trainer import CorrectionTrainer
from app.training.helox_job_client import HeloxJobClient
from app.training.training_status import TrainingStatusMonitor

logger = get_logger("cyrex.training.agent_service")

try:
    from deepiri_training_orchestrator import (
        FeedbackBuffer,
        FeedbackLoopTrainer,
        ReproducibilityController,
        TrainingOrchestrator,
    )

    _ORCH_AVAILABLE = True
except ImportError:
    _ORCH_AVAILABLE = False

    class TrainingOrchestrator:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            logger.warning("TrainingOrchestrator stub — install deepiri-training-orchestrator")

        def fit(self, *args: Any, **kwargs: Any) -> Any:
            return {"status": "stub"}

    class FeedbackLoopTrainer:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def submit(self, *args: Any, **kwargs: Any) -> Any:
            return None

try:
    from deepiri_modelkit.contracts.training import (
        AgentTrainingJob,
        TrainingRunRequest,
    )
except ImportError:
    TrainingRunRequest = None  # type: ignore[misc, assignment]
    AgentTrainingJob = None  # type: ignore[misc, assignment]


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
        self._feedback_trainer: Optional[FeedbackLoopTrainer] = None
        if _ORCH_AVAILABLE:
            repro = ReproducibilityController(seed=42)
            repro.set_seeds()
            orch = TrainingOrchestrator({"lr": 1e-4}, reproducibility=repro, max_steps=50)
            self._feedback_trainer = FeedbackLoopTrainer(orch, min_examples=2)

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
        if self._feedback_trainer is None:
            return {"status": "stub", "message": "training-orchestrator not installed"}
        ctx = None
        for aid in artifact_ids:
            ctx = self._feedback_trainer.submit(
                {"text": f"correction-{aid}", "artifact_id": aid},
                train_step=lambda step, batch: {"loss": 0.1},
            )
        return {
            "status": "completed" if ctx else "buffered",
            "artifact_ids": artifact_ids,
            "steps": getattr(ctx, "step", 0) if ctx else 0,
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
