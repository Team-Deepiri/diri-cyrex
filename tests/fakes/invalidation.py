"""Fake InvalidationPort for track-local tests."""

from __future__ import annotations

from typing import List

from app.pipeline.contracts.ports import InvalidationPort


class NoOpInvalidation(InvalidationPort):
    """Does nothing — suitable when invalidation behaviour is not under test."""

    async def enqueue(self, artifact_ids: List[str]) -> None:
        pass


class RecordingInvalidation(InvalidationPort):
    """Records enqueued artifact IDs for assertion in tests."""

    def __init__(self) -> None:
        self.enqueued: List[List[str]] = []

    async def enqueue(self, artifact_ids: List[str]) -> None:
        self.enqueued.append(list(artifact_ids))
