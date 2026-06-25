"""Fake ReckoningReadPort for track-local tests."""

from __future__ import annotations

from typing import List

from app.pipeline.contracts.models import PredictionRecord
from app.pipeline.contracts.ports import ReckoningReadPort


class FakeReckoningRead(ReckoningReadPort):
    """Returns pre-configured prediction records keyed by document_id."""

    def __init__(self) -> None:
        self._records: dict[str, List[PredictionRecord]] = {}

    def set_records(self, document_id: str, records: List[PredictionRecord]) -> None:
        self._records[document_id] = records

    async def get_reckoning(self, document_id: str) -> List[PredictionRecord]:
        return list(self._records.get(document_id, []))
