"""
Realtime Streaming System
WebSocket and SSE-based streaming using Redis pub/sub
For agent output streaming and live updates
"""
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import json
import uuid
import redis.asyncio as aioredis
from ..logging_config import get_logger
from ..settings import settings

logger = get_logger("cyrex.realtime_streaming")


class StreamEventType(str, Enum):
    """Types of streaming events"""
    # Agent events
    AGENT_START = "agent_start"
    AGENT_TOKEN = "agent_token"
    AGENT_TOOL_CALL = "agent_tool_call"
    AGENT_TOOL_RESULT = "agent_tool_result"
    AGENT_COMPLETE = "agent_complete"
    AGENT_ERROR = "agent_error"
    
    # Workflow events
    WORKFLOW_START = "workflow_start"
    WORKFLOW_STEP = "workflow_step"
    WORKFLOW_PROGRESS = "workflow_progress"
    WORKFLOW_COMPLETE = "workflow_complete"
    
    # System events
    HEARTBEAT = "heartbeat"
    CONNECTION = "connection"
    DISCONNECT = "disconnect"
    
    # Custom events
    CUSTOM = "custom"


@dataclass
class StreamEvent:
    """Streaming event structure"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: StreamEventType = StreamEventType.CUSTOM
    channel: str = "default"
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sequence: int = 0
    
    def to_json(self) -> str:
        return json.dumps({
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "channel": self.channel,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "sequence": self.sequence,
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StreamEvent':
        data = json.loads(json_str)
        data['event_type'] = StreamEventType(data['event_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)
    
    def to_sse(self) -> str:
        """Format as Server-Sent Event"""
        lines = [
            f"id: {self.event_id}",
            f"event: {self.event_type.value}",
            f"data: {json.dumps(self.data)}",
            "",
        ]
        return "\n".join(lines)


class StreamBuffer:
    """Buffer for streaming token output"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buffer: List[str] = []
        self._complete = False
        self._lock = asyncio.Lock()
    
    async def append(self, token: str):
        """Append token to buffer"""
        async with self._lock:
            self._buffer.append(token)
            if len(self._buffer) > self.max_size:
                self._buffer.pop(0)
    
    async def get_all(self) -> str:
        """Get complete buffered content"""
        async with self._lock:
            return "".join(self._buffer)
    
    async def mark_complete(self):
        """Mark streaming as complete"""
        self._complete = True
    
    @property
    def is_complete(self) -> bool:
        return self._complete
    
    async def clear(self):
        """Clear buffer"""
        async with self._lock:
            self._buffer.clear()
            self._complete = False


class RealtimeStreamPublisher:
    """
    Publisher for realtime streaming events
    Uses Redis pub/sub for distribution
    """
    
    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
        self._sequence_counters: Dict[str, int] = {}
        self._buffers: Dict[str, StreamBuffer] = {}
        self.logger = logger
    
    async def connect(self):
        """Connect to Redis"""
        if self._redis is None:
            self._redis = aioredis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                decode_responses=True,
            )
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self._redis:
            await self._redis.close()
            self._redis = None
    
    def _get_sequence(self, channel: str) -> int:
        """Get and increment sequence number for channel"""
        if channel not in self._sequence_counters:
            self._sequence_counters[channel] = 0
        self._sequence_counters[channel] += 1
        return self._sequence_counters[channel]
    
    async def publish_event(
        self,
        event_type: StreamEventType,
        data: Dict[str, Any],
        channel: str = "default",
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> str:
        """Publish a streaming event"""
        await self.connect()
        
        event = StreamEvent(
            event_type=event_type,
            channel=channel,
            session_id=session_id,
            agent_id=agent_id,
            data=data,
            sequence=self._get_sequence(channel),
        )
        
        await self._redis.publish(f"stream:{channel}", event.to_json())
        
        self.logger.debug(f"Published event {event.event_id} to stream:{channel}")
        return event.event_id
    
    async def start_agent_stream(
        self,
        session_id: str,
        agent_id: str,
        task_info: Dict[str, Any],
    ) -> str:
        """Start agent output streaming"""
        channel = f"agent:{session_id}:{agent_id}"
        
        # Create buffer for this stream
        self._buffers[channel] = StreamBuffer()
        
        await self.publish_event(
            StreamEventType.AGENT_START,
            {"task": task_info, "status": "started"},
            channel=channel,
            session_id=session_id,
            agent_id=agent_id,
        )
        
        return channel
    
    async def stream_token(
        self,
        channel: str,
        token: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ):
        """Stream a single token"""
        await self.connect()
        
        # Add to buffer
        if channel in self._buffers:
            await self._buffers[channel].append(token)
        
        await self.publish_event(
            StreamEventType.AGENT_TOKEN,
            {"token": token},
            channel=channel,
            session_id=session_id,
            agent_id=agent_id,
        )
    
    async def stream_tokens(
        self,
        channel: str,
        tokens: AsyncGenerator[str, None],
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ):
        """Stream tokens from async generator"""
        async for token in tokens:
            await self.stream_token(channel, token, session_id, agent_id)
    
    async def stream_tool_call(
        self,
        channel: str,
        tool_name: str,
        parameters: Dict[str, Any],
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ):
        """Stream a tool call event"""
        await self.publish_event(
            StreamEventType.AGENT_TOOL_CALL,
            {"tool": tool_name, "parameters": parameters},
            channel=channel,
            session_id=session_id,
            agent_id=agent_id,
        )
    
    async def stream_tool_result(
        self,
        channel: str,
        tool_name: str,
        result: Any,
        success: bool = True,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ):
        """Stream a tool result event"""
        await self.publish_event(
            StreamEventType.AGENT_TOOL_RESULT,
            {"tool": tool_name, "result": result, "success": success},
            channel=channel,
            session_id=session_id,
            agent_id=agent_id,
        )
    
    async def complete_agent_stream(
        self,
        channel: str,
        final_result: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ):
        """Complete agent streaming"""
        # Get buffered content if no final result provided
        if final_result is None and channel in self._buffers:
            final_result = await self._buffers[channel].get_all()
        
        await self.publish_event(
            StreamEventType.AGENT_COMPLETE,
            {"result": final_result, "metadata": metadata or {}},
            channel=channel,
            session_id=session_id,
            agent_id=agent_id,
        )
        
        # Clean up buffer
        if channel in self._buffers:
            await self._buffers[channel].mark_complete()
            del self._buffers[channel]
    
    async def stream_error(
        self,
        channel: str,
        error: str,
        error_type: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ):
        """Stream an error event"""
        await self.publish_event(
            StreamEventType.AGENT_ERROR,
            {"error": error, "error_type": error_type},
            channel=channel,
            session_id=session_id,
            agent_id=agent_id,
        )
        
        # Clean up buffer on error
        if channel in self._buffers:
            del self._buffers[channel]
    
    async def stream_workflow_progress(
        self,
        workflow_id: str,
        step: str,
        progress: float,
        message: Optional[str] = None,
    ):
        """Stream workflow progress update"""
        channel = f"workflow:{workflow_id}"
        
        await self.publish_event(
            StreamEventType.WORKFLOW_PROGRESS,
            {
                "workflow_id": workflow_id,
                "step": step,
                "progress": progress,
                "message": message,
            },
            channel=channel,
        )
    
    async def send_heartbeat(self, channel: str = "system"):
        """Send heartbeat event"""
        await self.publish_event(
            StreamEventType.HEARTBEAT,
            {"timestamp": datetime.utcnow().isoformat()},
            channel=channel,
        )


class RealtimeStreamSubscriber:
    """
    Subscriber for realtime streaming events
    Supports async iteration for consuming events
    """
    
    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
        self._pubsub: Optional[aioredis.client.PubSub] = None
        self._subscribed_channels: Set[str] = set()
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._running = False
        self._listen_task: Optional[asyncio.Task] = None
        self.logger = logger
    
    async def connect(self):
        """Connect to Redis"""
        if self._redis is None:
            self._redis = aioredis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                decode_responses=True,
            )
            self._pubsub = self._redis.pubsub()
    
    async def disconnect(self):
        """Disconnect from Redis"""
        self._running = False
        
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        
        if self._pubsub:
            await self._pubsub.close()
        
        if self._redis:
            await self._redis.close()
            self._redis = None
    
    async def subscribe(
        self,
        channel: str,
        handler: Optional[Callable[[StreamEvent], Any]] = None,
    ):
        """Subscribe to a channel"""
        await self.connect()
        
        redis_channel = f"stream:{channel}"
        await self._pubsub.subscribe(redis_channel)
        self._subscribed_channels.add(channel)
        
        if handler:
            if channel not in self._event_handlers:
                self._event_handlers[channel] = []
            self._event_handlers[channel].append(handler)
        
        self.logger.info(f"Subscribed to channel: {channel}")
    
    async def subscribe_to_agent(
        self,
        session_id: str,
        agent_id: str,
        handler: Optional[Callable[[StreamEvent], Any]] = None,
    ):
        """Subscribe to agent stream"""
        channel = f"agent:{session_id}:{agent_id}"
        await self.subscribe(channel, handler)
    
    async def subscribe_to_workflow(
        self,
        workflow_id: str,
        handler: Optional[Callable[[StreamEvent], Any]] = None,
    ):
        """Subscribe to workflow stream"""
        channel = f"workflow:{workflow_id}"
        await self.subscribe(channel, handler)
    
    async def unsubscribe(self, channel: str):
        """Unsubscribe from a channel"""
        if self._pubsub:
            await self._pubsub.unsubscribe(f"stream:{channel}")
        
        self._subscribed_channels.discard(channel)
        self._event_handlers.pop(channel, None)
        
        self.logger.info(f"Unsubscribed from channel: {channel}")
    
    async def listen(self) -> AsyncGenerator[StreamEvent, None]:
        """Listen for events (async generator)"""
        await self.connect()
        
        while True:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0
                )
                
                if message and message["type"] == "message":
                    try:
                        event = StreamEvent.from_json(message["data"])
                        yield event
                    except Exception as e:
                        self.logger.error(f"Event parse error: {e}")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Listen error: {e}")
                await asyncio.sleep(1)
    
    async def start_background_listener(self):
        """Start background listener for handler dispatch"""
        self._running = True
        self._listen_task = asyncio.create_task(self._background_listen())
    
    async def _background_listen(self):
        """Background listener task"""
        while self._running:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0
                )
                
                if message and message["type"] == "message":
                    try:
                        event = StreamEvent.from_json(message["data"])
                        await self._dispatch_event(event)
                    except Exception as e:
                        self.logger.error(f"Event dispatch error: {e}")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Background listen error: {e}")
                await asyncio.sleep(1)
    
    async def _dispatch_event(self, event: StreamEvent):
        """Dispatch event to handlers"""
        handlers = self._event_handlers.get(event.channel, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                self.logger.error(f"Handler error: {e}")


class SSEStreamGenerator:
    """
    Server-Sent Events stream generator for HTTP endpoints
    """
    
    def __init__(self, subscriber: RealtimeStreamSubscriber):
        self.subscriber = subscriber
    
    async def generate(
        self,
        channels: List[str],
        heartbeat_interval: int = 30,
    ) -> AsyncGenerator[str, None]:
        """Generate SSE stream for channels"""
        for channel in channels:
            await self.subscriber.subscribe(channel)
        
        last_heartbeat = datetime.utcnow()
        
        async for event in self.subscriber.listen():
            # Yield event as SSE
            yield event.to_sse()
            
            # Send heartbeat if needed
            if (datetime.utcnow() - last_heartbeat).seconds >= heartbeat_interval:
                heartbeat = StreamEvent(
                    event_type=StreamEventType.HEARTBEAT,
                    data={"timestamp": datetime.utcnow().isoformat()},
                )
                yield heartbeat.to_sse()
                last_heartbeat = datetime.utcnow()


# ============================================================================
# Singleton Instances
# ============================================================================

_publisher: Optional[RealtimeStreamPublisher] = None


async def get_stream_publisher() -> RealtimeStreamPublisher:
    """Get or create stream publisher singleton"""
    global _publisher
    if _publisher is None:
        _publisher = RealtimeStreamPublisher()
        await _publisher.connect()
    return _publisher


def create_stream_subscriber() -> RealtimeStreamSubscriber:
    """Create a new stream subscriber"""
    return RealtimeStreamSubscriber()

