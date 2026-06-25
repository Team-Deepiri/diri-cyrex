"""Fake AnticipatePort for track-local tests."""

from __future__ import annotations

from typing import Any, List

from app.pipeline.contracts.models import PredictionRecord
from app.pipeline.contracts.ports import AnticipatePort


class NoOpAnticipate(AnticipatePort):
    """Returns empty predictions — the no-op default for tests."""

    async def run(
        self,
        parsed_doc: Any,
        document_class: str,
    ) -> List[PredictionRecord]:
        return []


class FixedAnticipate(AnticipatePort):
    """Returns fixed prediction records for tests."""

    def __init__(self, records: List[PredictionRecord]) -> None:
        self._records = records

    async def run(
        self,
        parsed_doc: Any,
        document_class: str,
    ) -> List[PredictionRecord]:
        return list(self._records)


# Alias used by contract compliance tests
FakeAnticipate = NoOpAnticipate
