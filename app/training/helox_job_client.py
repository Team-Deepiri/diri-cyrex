"""Submit training jobs to the Redis training-jobs stream via modelkit."""

from __future__ import annotations

import json
import os
from typing import Any, Optional
from uuid import uuid4

from app.logging_config import get_logger

logger = get_logger("cyrex.training.helox_job_client")

try:
    from deepiri_modelkit.contracts.training import (
        AgentTrainingJob,
        TrainingRunRequest,
    )
    from deepiri_modelkit.training.job_queue import (
        TRAINING_JOBS_STREAM,
        TrainingJobQueue,
    )

    _MODELKIT_AVAILABLE = True
except ImportError:
    _MODELKIT_AVAILABLE = False
    TRAINING_JOBS_STREAM = "training-jobs"

    class TrainingRunRequest:  # type: ignore[no-redef]
        """Stub when deepiri-modelkit is not installed."""

        def __init__(self, **kwargs: Any) -> None:
            self._data = kwargs
            self.id = kwargs.get("id", str(uuid4()))

        def model_dump_json(self) -> str:
            return json.dumps(self._data)

    class AgentTrainingJob:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            self._data = kwargs

        def model_dump_json(self) -> str:
            return json.dumps(self._data)

    class TrainingJobQueue:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            logger.warning("TrainingJobQueue stub active — install deepiri-modelkit")

        def enqueue(self, request: TrainingRunRequest, max_length: int = 10000) -> str:
            return f"stub-{getattr(request, 'id', uuid4())}"


class HeloxJobClient:
    """Redis producer for TrainingRunRequest and AgentTrainingJob messages."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        stream_name: str = TRAINING_JOBS_STREAM,
    ) -> None:
        url = redis_url or os.getenv("REDIS_URL")
        if _MODELKIT_AVAILABLE:
            self._queue = TrainingJobQueue(redis_url=url, stream_name=stream_name)
        else:
            self._queue = TrainingJobQueue()
        self.stream_name = stream_name

    def submit(self, request: TrainingRunRequest) -> str:
        """Enqueue a TrainingRunRequest on the training-jobs stream."""
        message_id = self._queue.enqueue(request)
        logger.info(
            "Submitted training run",
            experiment_id=getattr(request, "experiment_id", None),
            message_id=message_id,
        )
        return message_id

    def submit_agent_job(self, job: AgentTrainingJob) -> str:
        """Enqueue an agent-initiated training job (wraps optional nested run request)."""
        if job.training_run_request is not None:
            return self.submit(job.training_run_request)
        if _MODELKIT_AVAILABLE:
            payload = job.model_dump_json()
            message_id: str = self._queue.redis.xadd(  # type: ignore[attr-defined]
                self.stream_name,
                {"payload": payload},
                maxlen=10000,
                approximate=True,
            )
            return message_id
        return f"stub-agent-{job.correlation_id}"

    def stream_length(self) -> int:
        """Return pending job count when modelkit queue is available."""
        if hasattr(self._queue, "stream_length"):
            return int(self._queue.stream_length())
        return 0

    @property
    def is_live(self) -> bool:
        return _MODELKIT_AVAILABLE
