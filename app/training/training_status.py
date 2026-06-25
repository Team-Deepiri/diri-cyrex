"""Subscribe to and poll training-events from Redis."""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from app.logging_config import get_logger
from app.settings import settings

logger = get_logger("cyrex.training.status")

try:
    from deepiri_modelkit.contracts.events import TrainingEvent
    from deepiri_modelkit.streaming.event_stream import StreamingClient
    from deepiri_modelkit.streaming.topics import StreamTopics

    _MODELKIT_AVAILABLE = True
    _TRAINING_EVENTS_TOPIC = StreamTopics.TRAINING_EVENTS.value
except ImportError:
    _MODELKIT_AVAILABLE = False
    _TRAINING_EVENTS_TOPIC = "training-events"

    class TrainingEvent:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            self._data = kwargs

        @classmethod
        def model_validate(cls, data: Dict[str, Any]) -> TrainingEvent:
            return cls(**data)


def _redis_url() -> str:
    password = settings.REDIS_PASSWORD
    if password:
        return (
            f"redis://:{password}@{settings.REDIS_HOST}:"
            f"{settings.REDIS_PORT}/{settings.REDIS_DB}"
        )
    return os.getenv(
        "REDIS_URL",
        f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
    )


class TrainingStatusMonitor:
    """Poll or subscribe to training-events for experiment progress."""

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self._redis_url = redis_url or _redis_url()
        self._client: Optional[StreamingClient] = None
        if _MODELKIT_AVAILABLE:
            self._client = StreamingClient(redis_url=self._redis_url)

    async def connect(self) -> None:
        if self._client is not None:
            await self._client.connect()

    async def disconnect(self) -> None:
        if self._client is not None:
            await self._client.disconnect()

    async def poll_events(
        self,
        experiment_id: Optional[str] = None,
        last_id: str = "0",
        count: int = 50,
    ) -> List[Dict[str, Any]]:
        """Read recent training events, optionally filtered by experiment_id."""
        if self._client is None:
            return []

        await self.connect()
        raw_messages = await self._client.redis.xread(
            {_TRAINING_EVENTS_TOPIC: last_id},
            count=count,
        )

        events: List[Dict[str, Any]] = []
        for _stream, messages in raw_messages:
            for message_id, fields in messages:
                payload = self._decode_fields(fields)
                if experiment_id and payload.get("experiment_id") != experiment_id:
                    continue
                payload["message_id"] = message_id
                events.append(payload)
        return events

    async def subscribe(
        self,
        callback: Callable[[Dict[str, Any]], Any],
        experiment_id: Optional[str] = None,
        block_ms: int = 5000,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Block-read training events and invoke callback for each match."""
        last_id = "$"
        while True:
            batch = await self.poll_events(
                experiment_id=experiment_id,
                last_id=last_id,
                count=10,
            )
            if not batch:
                import asyncio

                await asyncio.sleep(block_ms / 1000)
                continue
            for event in batch:
                last_id = event.get("message_id", last_id)
                result = callback(event)
                yield event
                if result is False:
                    return

    @staticmethod
    def _decode_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
        if "payload" in fields:
            return json.loads(fields["payload"])
        if "data" in fields:
            raw = fields["data"]
            return json.loads(raw) if isinstance(raw, str) else raw
        return dict(fields)

    @property
    def is_live(self) -> bool:
        return _MODELKIT_AVAILABLE
