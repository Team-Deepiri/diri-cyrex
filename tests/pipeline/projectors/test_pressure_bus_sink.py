"""Tests for PressureBusSink."""

from __future__ import annotations

import pytest

from app.pipeline.contracts.pressure_events import PassDiscrepancy
from app.pipeline.projectors.pressure_bus_sink import PressureBusSink


class _FakeBus:
    def __init__(self):
        self.calls = []

    async def publish(self, stream, event_type, payload, *, maxlen=50000):
        self.calls.append((stream, event_type, payload))
        return "2-0"


@pytest.mark.asyncio
async def test_pressure_bus_sink_publishes_pipeline_pressure():
    bus = _FakeBus()
    sink = PressureBusSink()
    sink._bus = bus
    event = PassDiscrepancy(
        document_id="doc-1",
        section_id="sec-1",
        field_name="rent",
        pass_a_value="1000",
        pass_b_value="1200",
    )
    await sink.emit(event)
    assert len(bus.calls) == 1
    stream, event_type, payload = bus.calls[0]
    assert stream == "pipeline.pressure.events"
    assert event_type == "pressure.event"
    assert payload["pressure_event_type"] == "pass_discrepancy"
    assert payload["document_id"] == "doc-1"
