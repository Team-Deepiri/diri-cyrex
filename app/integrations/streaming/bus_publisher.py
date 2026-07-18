"""
Sugar-Glider-first bus publisher for Cyrex pipeline / platform streams.

Prefer SYNAPSE_TRANSPORT=sidecar (gRPC Sugar Glider). Fall back to direct
Redis XADD only when sidecar is unavailable or transport=redis.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from app.logging_config import get_logger

logger = get_logger("cyrex.streaming.bus_publisher")

try:
    from deepiri_modelkit.streaming.topics import StreamTopics
except ImportError:  # pragma: no cover
    StreamTopics = None  # type: ignore[misc, assignment]


def _topic(name: str, fallback: str) -> str:
    if StreamTopics is None:
        return fallback
    attr = getattr(StreamTopics, name, None)
    if attr is None:
        return fallback
    return attr.value if hasattr(attr, "value") else str(attr)


# Stable topic strings (also used when modelkit is an older pin).
HELOX_TRAINING_RAW = _topic("HELOX_TRAINING_RAW", "pipeline.helox-training.raw")
HELOX_TRAINING_STRUCTURED = _topic(
    "HELOX_TRAINING_STRUCTURED", "pipeline.helox-training.structured"
)
PIPELINE_PRESSURE_EVENTS = _topic(
    "PIPELINE_PRESSURE_EVENTS", "pipeline.pressure.events"
)
PIPELINE_ARTIFACT_INVALIDATION = _topic(
    "PIPELINE_ARTIFACT_INVALIDATION", "pipeline.artifact.invalidation"
)
PIPELINE_SPLICE_EVENTS = _topic("PIPELINE_SPLICE_EVENTS", "pipeline.splice.events")
PIPELINE_DEAD_LETTER = _topic("PIPELINE_DEAD_LETTER", "pipeline.dead-letter")
PIPELINE_METRICS = _topic("PIPELINE_METRICS", "pipeline.metrics")
MODEL_EVENTS = _topic("MODEL_EVENTS", "model-events")
TRAINING_EVENTS = _topic("TRAINING_EVENTS", "training-events")


class BusPublisher:
    """Publish typed payloads onto Redis Streams via Sugar Glider when configured."""

    def __init__(
        self,
        *,
        sender: str = "cyrex",
        redis_client: Any = None,
    ) -> None:
        self.sender = sender
        self._redis = redis_client
        transport = (os.getenv("SYNAPSE_TRANSPORT", "sidecar") or "sidecar").strip().lower()
        self.use_sidecar = transport == "sidecar"
        self.sidecar_url = (
            os.getenv("SYNAPSE_SUGAR_GLIDER_URL")
            or os.getenv("SYNAPSE_SIDECAR_URL")
            or "http://synapse-sugar-glider:8081"
        ).rstrip("/")
        self._sidecar = None

    async def _get_sidecar(self):
        if self._sidecar is not None:
            return self._sidecar
        if not self.use_sidecar:
            return None
        try:
            from app.integrations.streaming.synapse_sugar_glider_client import (
                SynapseSidecarClient,
            )

            client = SynapseSidecarClient(
                base_url=self.sidecar_url,
                default_sender=self.sender,
            )
            if await client.ready():
                self._sidecar = client
                return client
            logger.warning("sugar_glider_not_ready", url=self.sidecar_url)
        except Exception as exc:
            logger.warning("sugar_glider_unavailable", error=str(exc))
        return None

    async def publish(
        self,
        stream: str,
        event_type: str,
        payload: Dict[str, Any],
        *,
        maxlen: int = 50_000,
    ) -> Optional[str]:
        """Publish payload; returns entry id when known."""
        sidecar = await self._get_sidecar()
        if sidecar is not None:
            try:
                entry_id = await sidecar.publish(
                    stream=stream,
                    event_type=event_type,
                    payload=payload,
                    sender=self.sender,
                )
                return entry_id
            except Exception as exc:
                logger.warning(
                    "sugar_glider_publish_failed_falling_back",
                    stream=stream,
                    error=str(exc),
                )

        if self._redis is None:
            logger.error("bus_publish_no_transport", stream=stream)
            return None

        redis_payload = {
            k: json.dumps(v) if not isinstance(v, str) else v
            for k, v in payload.items()
            if v is not None
        }
        if "event" not in redis_payload:
            redis_payload["event"] = event_type
        entry_id = await self._redis.xadd(
            stream,
            redis_payload,
            maxlen=maxlen,
            approximate=True,
        )
        return entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)


_bus_publisher: Optional[BusPublisher] = None


def get_bus_publisher(*, redis_client: Any = None) -> BusPublisher:
    global _bus_publisher
    if _bus_publisher is None or (
        redis_client is not None and _bus_publisher._redis is not redis_client
    ):
        _bus_publisher = BusPublisher(redis_client=redis_client)
    return _bus_publisher
