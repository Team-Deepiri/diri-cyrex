"""Fake PressureSignalSink and PressureReadModelPort for track-local tests."""

from __future__ import annotations

from typing import List, Optional

from app.pipeline.contracts.models import PressureCell
from app.pipeline.contracts.ports import PressureReadModelPort, PressureSignalSink
from app.pipeline.contracts.pressure_events import PressureEvent


class FakePressureSignalSink(PressureSignalSink):
    """Captures events emitted during tests."""

    def __init__(self) -> None:
        self.events: List[PressureEvent] = []

    async def emit(self, event: PressureEvent) -> None:
        self.events.append(event)

    async def emit_many(self, events) -> None:  # type: ignore[override]
        self.events.extend(events)


class FakePressureReadModel(PressureReadModelPort):
    """Returns pre-configured pressure cells for tests."""

    def __init__(self, cells: Optional[List[PressureCell]] = None) -> None:
        self._cells = cells or []

    async def get_pressure(
        self,
        document_id: Optional[str] = None,
    ) -> List[PressureCell]:
        if document_id is None:
            return list(self._cells)
        return [c for c in self._cells if c.document_id == document_id]
