"""Subscribe to model-ready events and hot-reload PEFT adapters via Sugar Glider."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Callable, Dict, Optional

from app.logging_config import get_logger

logger = get_logger("cyrex.training.model_reload")

MODEL_EVENTS = "model-events"
CONSUMER_GROUP = "cyrex-model-loader"
CONSUMER_NAME = os.getenv("CYREX_MODEL_RELOAD_CONSUMER_NAME", "cyrex-reload-1")


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
        self._redis_host = redis_host or os.getenv("REDIS_HOST", "redis")
        self._redis_port = redis_port or int(os.getenv("REDIS_PORT", "6379"))
        transport = (os.getenv("SYNAPSE_TRANSPORT", "sidecar") or "sidecar").strip().lower()
        self._use_sidecar = transport == "sidecar"
        self._sidecar_url = (
            os.getenv("SYNAPSE_SUGAR_GLIDER_URL")
            or os.getenv("SYNAPSE_SIDECAR_URL")
            or "http://synapse-sugar-glider:8081"
        ).rstrip("/")
        self._sidecar = None

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

    def _decode_payload(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for key, value in fields.items():
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            if isinstance(value, str) and value[:1] in "{[":
                try:
                    data[key] = json.loads(value)
                    continue
                except json.JSONDecodeError:
                    pass
            data[key] = value
        # Sugar Glider wraps JSON in "payload"
        inner = data.get("payload")
        if isinstance(inner, str) and inner[:1] in "{[":
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, dict):
                    return {**data, **parsed}
            except json.JSONDecodeError:
                pass
        if isinstance(inner, dict):
            return {**data, **inner}
        return data

    async def _handle_event(self, payload: Dict[str, Any]) -> None:
        event_name = payload.get("event") or payload.get("event_type")
        if event_name not in ("model-ready", "model_ready"):
            return
        logger.info("model_ready_event", model=payload.get("model_name"))
        self._on_reload(payload)

    async def _get_sidecar(self):
        if self._sidecar is not None:
            return self._sidecar
        from app.integrations.streaming.synapse_sugar_glider_client import (
            SynapseSidecarClient,
        )

        client = SynapseSidecarClient(
            base_url=self._sidecar_url,
            default_sender="cyrex-model-loader",
        )
        if await client.ready():
            self._sidecar = client
            return client
        raise RuntimeError(f"Sugar Glider not ready at {self._sidecar_url}")

    async def _run_sidecar(self) -> None:
        client = await self._get_sidecar()
        logger.info(
            "model_reload_via_sugar_glider",
            stream=MODEL_EVENTS,
            group=CONSUMER_GROUP,
        )
        while True:
            try:
                events = await client.read(
                    stream=MODEL_EVENTS,
                    consumer_group=CONSUMER_GROUP,
                    consumer_name=CONSUMER_NAME,
                    count=16,
                    block_ms=2000,
                )
                acked = []
                for event in events:
                    payload = self._decode_payload(event.fields)
                    await self._handle_event(payload)
                    acked.append(event.entry_id)
                if acked:
                    await client.ack(MODEL_EVENTS, CONSUMER_GROUP, acked)
            except Exception as exc:
                logger.debug("model_reload_sidecar_error", error=str(exc))
                await asyncio.sleep(2.0)

    async def _run_redis_fallback(self) -> None:
        """Last-resort direct Redis consumer when Sugar Glider is unavailable."""
        try:
            import redis.asyncio as redis
        except ImportError:
            logger.warning("redis_asyncio_unavailable — model reload listener idle")
            return

        client = redis.Redis(
            host=self._redis_host,
            port=self._redis_port,
            decode_responses=True,
        )
        try:
            await client.xgroup_create(
                MODEL_EVENTS, CONSUMER_GROUP, id="0", mkstream=True
            )
        except Exception:
            pass

        logger.warning("model_reload_redis_fallback", stream=MODEL_EVENTS)
        while True:
            try:
                rows = await client.xreadgroup(
                    CONSUMER_GROUP,
                    CONSUMER_NAME,
                    streams={MODEL_EVENTS: ">"},
                    count=16,
                    block=2000,
                )
                if not rows:
                    continue
                for _stream, messages in rows:
                    for entry_id, fields in messages:
                        await self._handle_event(self._decode_payload(fields))
                        await client.xack(MODEL_EVENTS, CONSUMER_GROUP, entry_id)
            except Exception as exc:
                logger.debug("model_reload_redis_error", error=str(exc))
                await asyncio.sleep(2.0)

    async def run(self, poll_interval: float = 2.0) -> None:
        """Consume model-events via Sugar Glider (preferred) or Redis fallback."""
        del poll_interval  # retained for API compatibility
        if self._use_sidecar:
            try:
                await self._run_sidecar()
                return
            except Exception as exc:
                logger.warning(
                    "sugar_glider_model_reload_unavailable_falling_back",
                    error=str(exc),
                )
        await self._run_redis_fallback()


async def start_model_reload_listener() -> None:
    listener = ModelReloadListener()
    await listener.run()
