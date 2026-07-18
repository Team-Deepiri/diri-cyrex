"""Publish artifact invalidation waves onto the AGI pipeline bus."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.integrations.streaming.bus_publisher import (
    PIPELINE_ARTIFACT_INVALIDATION,
    get_bus_publisher,
)
from app.logging_config import get_logger

logger = get_logger("cyrex.pipeline.invalidation_publisher")


class InvalidationPublisher:
    """Emits pipeline.artifact.invalidation for Canvas VIZ-14 / cyrex-agi."""

    def __init__(self, *, redis_client: Any = None) -> None:
        self._bus = get_bus_publisher(redis_client=redis_client)

    async def publish(
        self,
        *,
        document_id: str,
        reason: str,
        artifact_id: Optional[str] = None,
        cascade: bool = False,
        data: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        payload = {
            "event": "artifact.invalidation",
            "source": "cyrex.invalidation_worker",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "document_id": document_id,
            "artifact_id": artifact_id,
            "reason": reason,
            "cascade": cascade,
            "data": data or {},
        }
        entry_id = await self._bus.publish(
            PIPELINE_ARTIFACT_INVALIDATION,
            "artifact.invalidation",
            payload,
        )
        logger.info(
            "invalidation_published",
            document_id=document_id,
            artifact_id=artifact_id,
            entry_id=entry_id,
        )
        return entry_id
