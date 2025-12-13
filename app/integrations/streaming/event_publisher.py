"""
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

