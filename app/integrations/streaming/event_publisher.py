"""
<<<<<<< HEAD
Cyrex Event Publisher
Publishes events to Redis Streams for cross-service communication
Used for inference events, model status, and AGI decisions
"""
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import os

from deepiri_modelkit import (
    StreamingClient,
    InferenceEvent,
    ModelReadyEvent,
    PlatformEvent,
    get_logger
)

logger = get_logger("cyrex.event_publisher")


class CyrexEventPublisher:
    """
    Publishes Cyrex runtime events to streaming platform
    
    Events published:
    - InferenceEvent: Model predictions
    - PlatformEvent: System status, errors
    - AGIDecisionEvent: AGI system decisions
    """
    
    def __init__(self):
        self.streaming = StreamingClient(
            redis_url=os.getenv("REDIS_URL", "redis://redis:6379")
        )
        self._connected = False
    
    async def connect(self):
        """Connect to streaming platform"""
        if not self._connected:
            await self.streaming.connect()
            self._connected = True
            logger.info("event_publisher_connected")
    
    async def disconnect(self):
        """Disconnect from streaming platform"""
        if self._connected:
            await self.streaming.disconnect()
            self._connected = False
            logger.info("event_publisher_disconnected")
    
    async def publish_inference_event(self, event: InferenceEvent):
        """Publish model inference event"""
        await self.connect()
        
        # Convert Pydantic model to dict
        event_data = event.model_dump() if hasattr(event, 'model_dump') else event.dict()
        
        await self.streaming.publish("inference-events", event_data)
        
        logger.info("inference_event_published",
                    model=event.model_name,
                    latency_ms=event.latency_ms if hasattr(event, 'latency_ms') else None)
    
    async def publish_platform_event(self, event: PlatformEvent):
        """Publish platform event"""
        await self.connect()
        
        event_data = event.model_dump() if hasattr(event, 'model_dump') else event.dict()
        
        await self.streaming.publish("platform-events", event_data)
        
        logger.info("platform_event_published",
                    event_type=event.event_type if hasattr(event, 'event_type') else None)
    
    async def publish_agi_decision(self, decision: Dict[str, Any]):
        """Publish AGI decision event"""
        await self.connect()
        
        event_data = {
            "event": "agi-decision",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **decision
        }
        
        await self.streaming.publish("agi-events", event_data)
        
        logger.info("agi_decision_published",
                    decision_type=decision.get("type"))
    
    async def publish_model_status(self, model_name: str, status: str, **kwargs):
        """Publish model status update"""
        await self.connect()
        
        event_data = {
            "event": "model-status",
            "model_name": model_name,
            "status": status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **kwargs
        }
        
        await self.streaming.publish("model-events", event_data)
        
        logger.info("model_status_published",
                    model=model_name,
                    status=status)
    
    async def publish_error(self, error: Exception, context: Dict[str, Any]):
        """Publish error event"""
        await self.connect()
        
        event_data = {
            "event": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        await self.streaming.publish("error-events", event_data)
        
        logger.error("error_event_published",
                    error_type=type(error).__name__,
                    context=context)


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
=======
Event publisher for Cyrex runtime
Wraps deepiri-modelkit streaming client
"""
import os
from typing import Dict, Any, Optional, AsyncIterator
from deepiri_modelkit import StreamingClient as BaseStreamingClient
from deepiri_modelkit.streaming.topics import StreamTopics
from deepiri_modelkit.contracts.events import (
    ModelReadyEvent,
    InferenceEvent,
    PlatformEvent
)

class CyrexEventPublisher:
    """Streaming client for Cyrex runtime"""
    
    def __init__(self):
        """Initialize streaming client"""
        redis_host = os.getenv("REDIS_HOST", "redis")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_password = os.getenv("REDIS_PASSWORD", "redispassword")
        
        self.client = BaseStreamingClient(
            redis_host=redis_host,
            redis_port=redis_port,
            redis_password=redis_password
        )
    
    async def connect(self):
        """Connect to Redis"""
        await self.client.connect()
    
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
        
        await self.client.publish(
            StreamTopics.INFERENCE_EVENTS,
            event.model_dump()
        )
    
    async def subscribe_to_model_events(
        self,
        callback: callable
    ) -> AsyncIterator[ModelReadyEvent]:
        """Subscribe to model-ready events"""
        async for event_data in self.client.subscribe(
            StreamTopics.MODEL_EVENTS,
            callback,
            consumer_group="cyrex-runtime",
            consumer_name="cyrex-1"
        ):
            # Validate and yield ModelReadyEvent
            if event_data.get("event") == "model-ready":
                try:
                    event = ModelReadyEvent(**event_data)
                    yield event
                except Exception as e:
                    print(f"Invalid model event: {e}")
>>>>>>> 87ac0a3b871d936687efc3e7822b7b811d189c0d

