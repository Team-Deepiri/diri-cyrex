"""
Cyrex Event Publisher
Publishes events to Redis Streams for cross-service communication
Used for inference events, model status, and AGI decisions
"""
import asyncio
import json
from typing import Dict, Any, Optional, AsyncIterator, Callable
from datetime import datetime
import os

from deepiri_modelkit import (
    StreamingClient,
    InferenceEvent,
    ModelReadyEvent,
    PlatformEvent,
    get_logger
)
from deepiri_modelkit.streaming.topics import StreamTopics

from ...settings import settings
from .synapse_sidecar_client import SidecarError, SynapseSidecarClient

logger = get_logger("cyrex.event_publisher")


def _redis_url() -> str:
    """Build Redis URL from env or settings (so REDIS_HOST/PORT work locally)."""
    url = os.getenv("REDIS_URL")
    if url:
        return url
    if settings.REDIS_PASSWORD:
        return f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}"
    return f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("invalid_float_env", name=name, raw_value=raw, fallback=default)
        return default


def _stream_name(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


class CyrexEventPublisher:
    """
    Publishes Cyrex runtime events to streaming platform

    Events published:
    - InferenceEvent: Model predictions
    - PlatformEvent: System status, errors
    - AGIDecisionEvent: AGI system decisions
    """

    def __init__(self):
        self.transport = (os.getenv("SYNAPSE_TRANSPORT", "redis") or "redis").strip().lower()
        self.use_sidecar = self.transport == "sidecar"
        self._connected = False

        self._model_event_group = os.getenv("SYNAPSE_MODEL_EVENTS_GROUP", "cyrex-runtime")
        self._model_event_consumer = os.getenv("SYNAPSE_CONSUMER_NAME", "cyrex-1")

        if self.use_sidecar:
            self.streaming: Optional[StreamingClient] = None
            self.sidecar = SynapseSidecarClient(
                base_url=os.getenv("SYNAPSE_SIDECAR_URL", "http://synapse-sidecar:8081"),
                timeout_sec=_env_float("SYNAPSE_SIDECAR_TIMEOUT_SEC", 5.0),
                default_sender=os.getenv("SYNAPSE_SIDECAR_SENDER", "cyrex"),
                grpc_addr=os.getenv("SYNAPSE_GRPC_ADDR"),
            )
        else:
            self.streaming = StreamingClient(redis_url=_redis_url())
            self.sidecar: Optional[SynapseSidecarClient] = None

    async def connect(self):
        """Connect to streaming platform"""
        if self._connected:
            return

        if self.use_sidecar and self.sidecar:
            ready = await self.sidecar.ready()
            if not ready:
                raise SidecarError("Synapse sidecar is not ready")
            self._connected = True
            logger.info("event_publisher_connected", transport="sidecar")
            return

        await self.streaming.connect()
        self._connected = True
        logger.info("event_publisher_connected", transport="redis")

    async def disconnect(self):
        """Disconnect from streaming platform"""
        if not self._connected:
            return

        if self.use_sidecar:
            self._connected = False
            logger.info("event_publisher_disconnected", transport="sidecar")
            return

        await self.streaming.disconnect()
        self._connected = False
        logger.info("event_publisher_disconnected", transport="redis")

    async def publish_inference(
        self,
        model_name: str,
        version: str,
        latency_ms: float,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
        confidence: Optional[float] = None
    ):
        """Publish inference event"""
        event = InferenceEvent(
            event="inference-complete",
            source="cyrex",
            model_name=model_name,
            version=version,
            user_id=user_id,
            request_id=request_id,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            cost=cost,
            confidence=confidence,
            success=True
        )

        # Exclude None values and convert booleans to strings for Redis compatibility
        event_data = event.model_dump(exclude_none=True) if hasattr(event, 'model_dump') else {k: v for k, v in event.dict().items() if v is not None}
        # Convert boolean values to strings for Redis compatibility
        event_data = {k: str(v) if isinstance(v, bool) else v for k, v in event_data.items()}

        await self._publish(
            _stream_name(StreamTopics.INFERENCE_EVENTS),
            event_data,
            event_type=event_data.get("event", "inference-complete"),
        )

    async def publish_inference_event(self, event: InferenceEvent):
        """Publish model inference event"""
        # Convert Pydantic model to dict, excluding None values and converting booleans to strings for Redis
        event_data = event.model_dump(exclude_none=True) if hasattr(event, 'model_dump') else {k: v for k, v in event.dict().items() if v is not None}

        # Convert boolean values to strings for Redis compatibility
        event_data = {k: str(v) if isinstance(v, bool) else v for k, v in event_data.items()}

        await self._publish(
            "inference-events",
            event_data,
            event_type=event_data.get("event", "inference-complete"),
        )

        logger.info("inference_event_published",
                    model=event.model_name,
                    latency_ms=event.latency_ms if hasattr(event, 'latency_ms') else None)

    async def publish_platform_event(self, event: PlatformEvent):
        """Publish platform event"""
        event_data = event.model_dump() if hasattr(event, 'model_dump') else event.dict()

        await self._publish(
            "platform-events",
            event_data,
            event_type=event_data.get("event_type") or event_data.get("event") or "platform-event",
        )

        logger.info("platform_event_published",
                    event_type=event.event_type if hasattr(event, 'event_type') else None)

    async def subscribe_to_model_events(
        self,
        callback: Callable[[Dict[str, Any]], Any]
    ) -> AsyncIterator[ModelReadyEvent]:
        """Subscribe to model-ready events"""
        if not self.use_sidecar:
            async for event_data in self.streaming.subscribe(
                StreamTopics.MODEL_EVENTS,
                callback,
                consumer_group=self._model_event_group,
                consumer_name=self._model_event_consumer
            ):
                # Validate and yield ModelReadyEvent
                if event_data.get("event") == "model-ready":
                    try:
                        event = ModelReadyEvent(**event_data)
                        yield event
                    except Exception as e:
                        logger.warning("invalid_model_event", error=str(e))
            return

        await self.connect()
        stream = _stream_name(StreamTopics.MODEL_EVENTS)

        while True:
            events = await self.sidecar.read(
                stream=stream,
                consumer_group=self._model_event_group,
                consumer_name=self._model_event_consumer,
                count=10,
                block_ms=5000,
            )
            if not events:
                continue

            ack_ids = []
            for item in events:
                event_data = self._payload_from_sidecar_fields(item.fields)
                await self._emit_callback(callback, event_data)

                if event_data.get("event") == "model-ready":
                    try:
                        event = ModelReadyEvent(**event_data)
                        yield event
                    except Exception as e:
                        logger.warning("invalid_model_event", error=str(e))

                ack_ids.append(item.entry_id)

            if ack_ids:
                await self.sidecar.ack(stream, self._model_event_group, ack_ids)

    async def publish_agi_decision(self, decision: Dict[str, Any]):
        """Publish AGI decision event"""
        event_data = {
            "event": "agi-decision",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **decision
        }

        await self._publish("agi-events", event_data, event_type="agi-decision")

        logger.info("agi_decision_published",
                    decision_type=decision.get("type"))

    async def publish_model_status(self, model_name: str, status: str, **kwargs):
        """Publish model status update"""
        event_data = {
            "event": "model-status",
            "model_name": model_name,
            "status": status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **kwargs
        }

        await self._publish("model-events", event_data, event_type="model-status")

        logger.info("model_status_published",
                    model=model_name,
                    status=status)

    async def publish_error(self, error: Exception, context: Dict[str, Any]):
        """Publish error event"""
        event_data = {
            "event": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        await self._publish("error-events", event_data, event_type="error")

        logger.error("error_event_published",
                    error_type=type(error).__name__,
                    context=context)

    async def _publish(self, stream: str, event_data: Dict[str, Any], event_type: str):
        await self.connect()

        if self.use_sidecar and self.sidecar:
            await self.sidecar.publish(
                stream=stream,
                event_type=event_type,
                payload=event_data,
                sender="cyrex",
            )
            return

        await self.streaming.publish(stream, event_data)

    async def _emit_callback(self, callback: Callable[[Dict[str, Any]], Any], event_data: Dict[str, Any]):
        if callback is None:
            return
        result = callback(event_data)
        if asyncio.iscoroutine(result):
            await result

    @staticmethod
    def _payload_from_sidecar_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
        payload = fields.get("payload", {})
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                payload = {}
        elif not isinstance(payload, dict):
            payload = {}

        if "event" not in payload and fields.get("event_type"):
            payload["event"] = fields.get("event_type")

        if "timestamp" not in payload and fields.get("timestamp"):
            payload["timestamp"] = fields.get("timestamp")

        if "sender" not in payload and fields.get("sender"):
            payload["sender"] = fields.get("sender")

        return payload


# Singleton instance
_publisher: Optional[CyrexEventPublisher] = None


async def get_event_publisher() -> CyrexEventPublisher:
    """Get or create singleton event publisher"""
    global _publisher
    if _publisher is None:
        _publisher = CyrexEventPublisher()
        await _publisher.connect()
    return _publisher


async def shutdown_event_publisher():
    """Shutdown singleton event publisher"""
    global _publisher
    if _publisher:
        await _publisher.disconnect()
        _publisher = None
