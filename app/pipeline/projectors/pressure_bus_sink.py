"""PressureSignalSink that dual-writes Postgres-ready events onto the bus."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from app.integrations.streaming.bus_publisher import (
    PIPELINE_PRESSURE_EVENTS,
    get_bus_publisher,
)
from app.logging_config import get_logger
from app.pipeline.contracts.pressure_events import PressureEvent, discriminated_event_type

logger = get_logger("cyrex.pipeline.pressure_bus_sink")


class PressureBusSink:
    """
    Implements PressureSignalSink by publishing to pipeline.pressure.events.

    Callers that also persist pressure_events/pressure_cells should do that
    first (Postgres source of truth), then emit via this sink for cyrex-agi,
    Telemetry, and Canvas observers.
    """

    def __init__(
        self,
        *,
        redis_client: Any = None,
        local_handlers: Optional[List[Any]] = None,
    ) -> None:
        self._bus = get_bus_publisher(redis_client=redis_client)
        self._local_handlers = list(local_handlers or [])

    async def emit(self, event: PressureEvent) -> None:
        payload = self._to_bus_payload(event)
        await self._bus.publish(
            PIPELINE_PRESSURE_EVENTS,
            "pressure.event",
            payload,
        )
        for handler in self._local_handlers:
            try:
                result = handler(event)
                if hasattr(result, "__await__"):
                    await result
            except Exception as exc:
                logger.warning("pressure_local_handler_failed", error=str(exc))

    async def emit_many(self, events: Iterable[PressureEvent]) -> None:
        for event in events:
            await self.emit(event)

    def _to_bus_payload(self, event: PressureEvent) -> Dict[str, Any]:
        data = event.model_dump()
        event_type = discriminated_event_type(event)
        return {
            "event": "pressure.event",
            "source": "cyrex.pressure_projector",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "document_id": data.get("document_id"),
            "section_id": data.get("section_id"),
            "pressure_event_type": event_type,
            "page": data.get("page"),
            "artifact_id": data.get("artifact_id"),
            "data": data,
        }
