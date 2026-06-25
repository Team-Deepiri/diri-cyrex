"""Buffer human corrections as LearningArtifacts for few-shot replay."""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterator, List

from app.pipeline.contracts.models import LearningArtifact


class CorrectionTrainer:
    """In-memory buffer of LearningArtifacts pending export to Helox."""

    def __init__(self, max_buffer: int = 1000) -> None:
        self._buffer: Deque[LearningArtifact] = deque(maxlen=max_buffer)

    @property
    def size(self) -> int:
        return len(self._buffer)

    def buffer(self, artifact: LearningArtifact) -> None:
        """Append a correction to the replay buffer."""
        self._buffer.append(artifact)

    def drain(self) -> List[LearningArtifact]:
        """Return and clear all buffered artifacts."""
        items = list(self._buffer)
        self._buffer.clear()
        return items

    def peek(self, limit: int = 100) -> List[LearningArtifact]:
        """Return the most recent artifacts without draining."""
        if limit <= 0:
            return []
        return list(self._buffer)[-limit:]

    def export_jsonl_lines(self) -> Iterator[str]:
        """Serialize buffered artifacts as JSONL lines for Helox ingestion."""
        for artifact in self._buffer:
            yield artifact.model_dump_json()
