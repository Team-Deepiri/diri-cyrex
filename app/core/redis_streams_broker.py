"""
Redis Streams Message Broker
High-performance message broker using Redis Streams (not RabbitMQ)
Supports consumer groups, message persistence, and real-time streaming
"""
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import uuid
import redis.asyncio as aioredis
from ..logging_config import get_logger
from ..settings import settings
from .types import MessagePriority

logger = get_logger("cyrex.redis_streams")


class StreamEventType(str, Enum):
    """Event types for stream messages"""
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_ERROR = "agent_error"
    TASK_QUEUED = "task_queued"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    MESSAGE_RECEIVED = "message_received"
    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"
    STATE_CHANGED = "state_changed"
    MEMORY_STORED = "memory_stored"
    STREAM_CHUNK = "stream_chunk"
    HEARTBEAT = "heartbeat"


@dataclass
class StreamMessage:
    """Message structure for Redis Streams"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stream: str = "default"
    event_type: StreamEventType = StreamEventType.MESSAGE_RECEIVED
    sender: str = "system"
    recipient: Optional[str] = None
    priority: MessagePriority = MessagePriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    ttl: Optional[int] = None  # Time to live in seconds
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to Redis-compatible dictionary (all strings)"""
        return {
            "message_id": self.message_id,
            "event_type": self.event_type.value,
            "sender": self.sender,
            "recipient": self.recipient or "",
            "priority": self.priority.value,
            "payload": json.dumps(self.payload),
            "timestamp": self.timestamp.isoformat(),
            "ttl": str(self.ttl) if self.ttl else "",
            "retry_count": str(self.retry_count),
            "max_retries": str(self.max_retries),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str], stream_id: str = "") -> 'StreamMessage':
        """Create from Redis dictionary"""
        return cls(
            message_id=data.get("message_id", stream_id),
            event_type=StreamEventType(data.get("event_type", "message_received")),
            sender=data.get("sender", "unknown"),
            recipient=data.get("recipient") or None,
            priority=MessagePriority(data.get("priority", "normal")),
            payload=json.loads(data.get("payload", "{}")),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            ttl=int(data.get("ttl")) if data.get("ttl") else None,
            retry_count=int(data.get("retry_count", "0")),
            max_retries=int(data.get("max_retries", "3")),
        )


@dataclass
class ConsumerGroup:
    """Consumer group configuration"""
    name: str
    stream: str
    consumer_name: str
    start_id: str = ">"  # ">" for new messages, "0" for all
    block_ms: int = 5000
    count: int = 10


class RedisStreamsBroker:
    """
    Redis Streams-based message broker
    Replaces RabbitMQ with Redis for simplicity and performance
    
    Features:
    - Pub/Sub with consumer groups
    - Message persistence
    - Real-time streaming
    - Dead letter queue
    - Message acknowledgment
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        max_stream_length: int = 10000,
        cleanup_interval: int = 3600,
    ):
        self.redis_url = redis_url
        self.max_stream_length = max_stream_length
        self.cleanup_interval = cleanup_interval
        self._redis: Optional[aioredis.Redis] = None
        self._subscribers: Dict[str, Set[Callable]] = {}
        self._consumer_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        self.logger = logger
    
    async def connect(self) -> bool:
        """Connect to Redis"""
        try:
            if self.redis_url:
                self._redis = await aioredis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5.0
                )
            else:
                self._redis = aioredis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    password=settings.REDIS_PASSWORD,
                    db=settings.REDIS_DB,
                    decode_responses=True,
                    socket_connect_timeout=5.0,
                )
            
            # Test connection
            await self._redis.ping()
            self._running = True
            self.logger.info("Connected to Redis for streams broker")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Redis"""
        self._running = False
        
        # Cancel consumer tasks
        for task in self._consumer_tasks.values():
            task.cancel()
        
        if self._consumer_tasks:
            await asyncio.gather(*self._consumer_tasks.values(), return_exceptions=True)
        self._consumer_tasks.clear()
        
        if self._redis:
            await self._redis.close()
            self._redis = None
        
        self.logger.info("Disconnected from Redis")
    
    async def _ensure_connected(self):
        """Ensure Redis connection is active"""
        if not self._redis:
            await self.connect()
    
    # ========================================================================
    # Publishing
    # ========================================================================
    
    async def publish(
        self,
        stream: str,
        event_type: StreamEventType,
        payload: Dict[str, Any],
        sender: str = "system",
        recipient: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl: Optional[int] = None,
    ) -> str:
        """
        Publish a message to a stream
        
        Returns:
            Redis stream entry ID
        """
        await self._ensure_connected()
        
        message = StreamMessage(
            stream=stream,
            event_type=event_type,
            sender=sender,
            recipient=recipient,
            priority=priority,
            payload=payload,
            ttl=ttl,
        )
        
        # Add to stream with automatic trimming
        entry_id = await self._redis.xadd(
            stream,
            message.to_dict(),
            maxlen=self.max_stream_length,
        )
        
        # Notify in-memory subscribers
        await self._notify_subscribers(stream, message)
        
        self.logger.debug(
            f"Published message to {stream}: {message.message_id}",
            event_type=event_type.value
        )
        
        return entry_id
    
    async def publish_batch(
        self,
        stream: str,
        messages: List[Dict[str, Any]],
        event_type: StreamEventType = StreamEventType.MESSAGE_RECEIVED,
    ) -> List[str]:
        """Publish multiple messages in a pipeline"""
        await self._ensure_connected()
        
        entry_ids = []
        async with self._redis.pipeline(transaction=True) as pipe:
            for msg_data in messages:
                message = StreamMessage(
                    stream=stream,
                    event_type=event_type,
                    payload=msg_data,
                )
                pipe.xadd(stream, message.to_dict(), maxlen=self.max_stream_length)
            
            results = await pipe.execute()
            entry_ids = [r for r in results if r]
        
        return entry_ids
    
    # ========================================================================
    # Consuming
    # ========================================================================
    
    async def consume(
        self,
        stream: str,
        count: int = 10,
        block_ms: int = 5000,
        last_id: str = "$",
    ) -> List[StreamMessage]:
        """
        Consume messages from a stream (simple mode)
        
        Args:
            stream: Stream name
            count: Max messages to fetch
            block_ms: Blocking timeout in milliseconds
            last_id: Start from this ID ("$" for new, "0" for beginning)
        
        Returns:
            List of StreamMessage objects
        """
        await self._ensure_connected()
        
        result = await self._redis.xread(
            {stream: last_id},
            count=count,
            block=block_ms,
        )
        
        messages = []
        if result:
            for stream_name, entries in result:
                for entry_id, data in entries:
                    msg = StreamMessage.from_dict(data, entry_id)
                    msg.stream = stream_name
                    messages.append(msg)
        
        return messages
    
    async def consume_group(
        self,
        group: ConsumerGroup,
    ) -> List[StreamMessage]:
        """
        Consume messages as part of a consumer group
        Supports message acknowledgment and redelivery
        """
        await self._ensure_connected()
        
        # Ensure consumer group exists
        try:
            await self._redis.xgroup_create(
                group.stream,
                group.name,
                id="0",
                mkstream=True,
            )
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
        
        result = await self._redis.xreadgroup(
            group.name,
            group.consumer_name,
            {group.stream: group.start_id},
            count=group.count,
            block=group.block_ms,
        )
        
        messages = []
        if result:
            for stream_name, entries in result:
                for entry_id, data in entries:
                    msg = StreamMessage.from_dict(data, entry_id)
                    msg.stream = stream_name
                    messages.append(msg)
        
        return messages
    
    async def acknowledge(self, stream: str, group: str, message_ids: List[str]) -> int:
        """Acknowledge processed messages"""
        await self._ensure_connected()
        
        if not message_ids:
            return 0
        
        return await self._redis.xack(stream, group, *message_ids)
    
    async def get_pending(
        self,
        stream: str,
        group: str,
        count: int = 10,
        consumer: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get pending (unacknowledged) messages"""
        await self._ensure_connected()
        
        result = await self._redis.xpending_range(
            stream,
            group,
            min="-",
            max="+",
            count=count,
            consumername=consumer,
        )
        
        return [
            {
                "id": item["message_id"],
                "consumer": item["consumer"],
                "idle_time": item["time_since_delivered"],
                "delivery_count": item["times_delivered"],
            }
            for item in result
        ]
    
    async def claim_messages(
        self,
        stream: str,
        group: str,
        consumer: str,
        min_idle_time: int,
        message_ids: List[str],
    ) -> List[StreamMessage]:
        """Claim messages from other consumers (for recovery)"""
        await self._ensure_connected()
        
        result = await self._redis.xclaim(
            stream,
            group,
            consumer,
            min_idle_time,
            message_ids,
        )
        
        return [StreamMessage.from_dict(data, entry_id) for entry_id, data in result]
    
    # ========================================================================
    # Subscription (In-Memory)
    # ========================================================================
    
    async def subscribe(
        self,
        stream: str,
        callback: Callable[[StreamMessage], Any],
    ) -> str:
        """
        Subscribe to a stream with a callback
        Creates a background consumer task
        """
        await self._ensure_connected()
        
        if stream not in self._subscribers:
            self._subscribers[stream] = set()
        
        self._subscribers[stream].add(callback)
        subscription_id = f"{stream}_{id(callback)}"
        
        # Start consumer task if not already running
        if stream not in self._consumer_tasks:
            self._consumer_tasks[stream] = asyncio.create_task(
                self._consumer_loop(stream)
            )
        
        self.logger.info(f"Subscribed to stream: {stream}")
        return subscription_id
    
    async def unsubscribe(self, stream: str, callback: Callable):
        """Unsubscribe from a stream"""
        if stream in self._subscribers:
            self._subscribers[stream].discard(callback)
            
            # Stop consumer if no subscribers
            if not self._subscribers[stream] and stream in self._consumer_tasks:
                self._consumer_tasks[stream].cancel()
                del self._consumer_tasks[stream]
        
        self.logger.info(f"Unsubscribed from stream: {stream}")
    
    async def _consumer_loop(self, stream: str):
        """Background consumer loop for subscriptions"""
        last_id = "$"
        
        while self._running:
            try:
                messages = await self.consume(
                    stream,
                    count=10,
                    block_ms=5000,
                    last_id=last_id,
                )
                
                for msg in messages:
                    await self._notify_subscribers(stream, msg)
                    # Update last_id to latest message
                    last_id = msg.message_id
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Consumer loop error: {e}")
                await asyncio.sleep(1)
    
    async def _notify_subscribers(self, stream: str, message: StreamMessage):
        """Notify all subscribers of a message"""
        if stream not in self._subscribers:
            return
        
        for callback in self._subscribers[stream]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                self.logger.error(f"Subscriber callback error: {e}")
    
    # ========================================================================
    # Stream Management
    # ========================================================================
    
    async def get_stream_info(self, stream: str) -> Dict[str, Any]:
        """Get stream information"""
        await self._ensure_connected()
        
        try:
            info = await self._redis.xinfo_stream(stream)
            return {
                "length": info.get("length", 0),
                "first_entry": info.get("first-entry"),
                "last_entry": info.get("last-entry"),
                "groups": info.get("groups", 0),
            }
        except aioredis.ResponseError:
            return {"length": 0, "exists": False}
    
    async def get_stream_length(self, stream: str) -> int:
        """Get number of messages in stream"""
        await self._ensure_connected()
        return await self._redis.xlen(stream)
    
    async def trim_stream(self, stream: str, max_length: Optional[int] = None) -> int:
        """Trim stream to max length"""
        await self._ensure_connected()
        max_len = max_length or self.max_stream_length
        return await self._redis.xtrim(stream, maxlen=max_len)
    
    async def delete_stream(self, stream: str) -> bool:
        """Delete a stream entirely"""
        await self._ensure_connected()
        result = await self._redis.delete(stream)
        return result > 0
    
    async def list_streams(self, pattern: str = "cyrex:*") -> List[str]:
        """List all streams matching pattern"""
        await self._ensure_connected()
        keys = await self._redis.keys(pattern)
        
        streams = []
        for key in keys:
            key_type = await self._redis.type(key)
            if key_type == "stream":
                streams.append(key)
        
        return streams
    
    # ========================================================================
    # Consumer Group Management
    # ========================================================================
    
    async def create_consumer_group(
        self,
        stream: str,
        group: str,
        start_id: str = "0",
    ) -> bool:
        """Create a consumer group"""
        await self._ensure_connected()
        
        try:
            await self._redis.xgroup_create(
                stream,
                group,
                id=start_id,
                mkstream=True,
            )
            return True
        except aioredis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                return True  # Already exists
            raise
    
    async def delete_consumer_group(self, stream: str, group: str) -> bool:
        """Delete a consumer group"""
        await self._ensure_connected()
        result = await self._redis.xgroup_destroy(stream, group)
        return result > 0
    
    async def list_consumer_groups(self, stream: str) -> List[Dict[str, Any]]:
        """List consumer groups for a stream"""
        await self._ensure_connected()
        
        try:
            groups = await self._redis.xinfo_groups(stream)
            return [
                {
                    "name": g["name"],
                    "consumers": g["consumers"],
                    "pending": g["pending"],
                    "last_delivered_id": g["last-delivered-id"],
                }
                for g in groups
            ]
        except aioredis.ResponseError:
            return []
    
    # ========================================================================
    # Dead Letter Queue
    # ========================================================================
    
    async def move_to_dlq(
        self,
        stream: str,
        message: StreamMessage,
        reason: str,
    ) -> str:
        """Move a failed message to dead letter queue"""
        dlq_stream = f"{stream}:dlq"
        
        dlq_payload = {
            "original_stream": stream,
            "original_message_id": message.message_id,
            "failure_reason": reason,
            "original_payload": message.payload,
        }
        
        return await self.publish(
            dlq_stream,
            StreamEventType.TASK_FAILED,
            dlq_payload,
            sender="dlq_handler",
        )
    
    async def get_dlq_messages(
        self,
        stream: str,
        count: int = 100,
    ) -> List[StreamMessage]:
        """Get messages from dead letter queue"""
        dlq_stream = f"{stream}:dlq"
        return await self.consume(dlq_stream, count=count, last_id="0", block_ms=0)


# ============================================================================
# Singleton and Factory
# ============================================================================

_streams_broker: Optional[RedisStreamsBroker] = None


async def get_redis_streams_broker() -> RedisStreamsBroker:
    """Get or create Redis Streams broker singleton"""
    global _streams_broker
    if _streams_broker is None:
        _streams_broker = RedisStreamsBroker()
        await _streams_broker.connect()
    return _streams_broker


async def close_redis_streams_broker():
    """Close the Redis Streams broker"""
    global _streams_broker
    if _streams_broker:
        await _streams_broker.disconnect()
        _streams_broker = None

