"""Subscribe to model-ready events and hot-reload PEFT adapters."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Callable, Dict, Optional

from app.logging_config import get_logger

logger = get_logger("cyrex.training.model_reload")

try:
    from deepiri_modelkit.contracts.events import ModelReadyEvent
    from deepiri_modelkit.streaming.event_stream import StreamingClient
    from deepiri_modelkit.streaming.topics import StreamTopics
except ImportError:
    ModelReadyEvent = None  # type: ignore[misc, assignment]
    StreamingClient = None  # type: ignore[misc, assignment]
    StreamTopics = None  # type: ignore[misc, assignment]


class ModelReloadListener:
    """Listens for model-ready events and triggers adapter reload callbacks."""

    def __init__(
        self,
        *,
        on_reload: Optional[Callable[[Dict[str, Any]], None]] = None,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
    ) -> None:
        self._on_reload = on_reload or self._default_reload
        host = redis_host or os.getenv("REDIS_HOST", "redis")
        port = redis_port or int(os.getenv("REDIS_PORT", "6379"))
        self._streaming = (
            StreamingClient(redis_host=host, redis_port=port)
            if StreamingClient is not None
            else None
        )

    def _default_reload(self, event: Dict[str, Any]) -> None:
        """Bridge to DynamicLoRAService when available."""
        try:
            from app.services.dynamic_lora_service import DynamicLoRAService

            service = DynamicLoRAService()
            model_name = event.get("model_name", "")
            version = event.get("version", "")
            path = event.get("registry_path") or event.get("checkpoint_path", "")
            if hasattr(service, "reload_adapter"):
                service.reload_adapter(model_name, version, path)
            else:
                logger.info("model_ready_received", model=model_name, version=version)
        except Exception as exc:
            logger.warning("adapter_reload_failed", error=str(exc))

    async def _handle_event(self, payload: Dict[str, Any]) -> None:
        if payload.get("event") != "model-ready":
            return
        logger.info("model_ready_event", model=payload.get("model_name"))
        self._on_reload(payload)

    async def run(self, poll_interval: float = 2.0) -> None:
        """Poll model-events stream (simplified consumer)."""
        if self._streaming is None:
            logger.warning("StreamingClient unavailable — model reload listener idle")
            return
        await self._streaming.connect()
        last_id = "0"
        stream = StreamTopics.MODEL_EVENTS.value if StreamTopics else "model-events"
        while True:
            try:
                messages = await self._streaming.read_stream(stream, last_id=last_id, count=10)
                for msg_id, fields in messages:
                    last_id = msg_id
                    data = {k: json.loads(v) if v and v.startswith("{") else v for k, v in fields.items()}
                    await self._handle_event(data)
            except Exception as exc:
                logger.debug("model_reload_poll_error", error=str(exc))
            await asyncio.sleep(poll_interval)


async def start_model_reload_listener() -> None:
    listener = ModelReloadListener()
    await listener.run()
