"""
Queue Producer/Consumer/Publisher System
Comprehensive queue management with Redis backend
Supports task queues, event publishing, and worker orchestration
"""
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import uuid
import redis.asyncio as aioredis
from ..logging_config import get_logger
from ..settings import settings

logger = get_logger("cyrex.queue_system")

T = TypeVar('T')


class QueuePriority(str, Enum):
    """Queue priority levels"""
    CRITICAL = "critical"  # Processed immediately
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BATCH = "batch"  # Processed in batches


class TaskState(str, Enum):
    """Task execution state"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    DEAD = "dead"  # Max retries exceeded


@dataclass
class QueueTask:
    """Task structure for queue processing"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    queue_name: str = "default"
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: QueuePriority = QueuePriority.NORMAL
    state: TaskState = TaskState.PENDING
    
    # Retry configuration
    max_retries: int = 3
    retry_count: int = 0
    retry_delay: int = 5  # seconds
    retry_backoff: float = 2.0  # exponential backoff multiplier
    
    # Timing
    timeout: int = 300  # 5 minutes default
    scheduled_at: Optional[datetime] = None  # For delayed tasks
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Result
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert datetime to ISO format
        for key in ['created_at', 'scheduled_at', 'started_at', 'completed_at']:
            if data[key]:
                data[key] = data[key].isoformat() if isinstance(data[key], datetime) else data[key]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueueTask':
        """Create from dictionary"""
        # Convert ISO strings back to datetime
        for key in ['created_at', 'scheduled_at', 'started_at', 'completed_at']:
            if data.get(key) and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])
        
        # Convert enums
        if isinstance(data.get('priority'), str):
            data['priority'] = QueuePriority(data['priority'])
        if isinstance(data.get('state'), str):
            data['state'] = TaskState(data['state'])
        
        return cls(**data)


@dataclass
class QueueEvent:
    """Event for pub/sub publishing"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    channel: str = "events"
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = "system"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_json(self) -> str:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'QueueEvent':
        data = json.loads(json_str)
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


# ============================================================================
# Queue Producer
# ============================================================================

class QueueProducer:
    """
    Produces tasks and events to queues
    Supports priority queues, delayed tasks, and batch operations
    """
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self._redis = redis_client
        self.logger = logger
    
    async def _get_redis(self) -> aioredis.Redis:
        """Get or create Redis connection"""
        if self._redis is None:
            self._redis = aioredis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                decode_responses=False,
            )
        return self._redis
    
    def _get_queue_key(self, queue_name: str, priority: QueuePriority) -> str:
        """Get Redis key for queue"""
        return f"queue:{queue_name}:{priority.value}"
    
    def _get_task_key(self, task_id: str) -> str:
        """Get Redis key for task data"""
        return f"task:{task_id}"
    
    async def enqueue(
        self,
        task_type: str,
        payload: Dict[str, Any],
        queue_name: str = "default",
        priority: QueuePriority = QueuePriority.NORMAL,
        delay_seconds: int = 0,
        max_retries: int = 3,
        timeout: int = 300,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Enqueue a task for processing
        
        Returns:
            Task ID
        """
        redis = await self._get_redis()
        
        task = QueueTask(
            task_type=task_type,
            queue_name=queue_name,
            payload=payload,
            priority=priority,
            max_retries=max_retries,
            timeout=timeout,
            scheduled_at=datetime.utcnow() + timedelta(seconds=delay_seconds) if delay_seconds > 0 else None,
            state=TaskState.SCHEDULED if delay_seconds > 0 else TaskState.PENDING,
            metadata=metadata or {},
        )
        
        # Store task data
        task_key = self._get_task_key(task.task_id)
        await redis.setex(
            task_key,
            86400 * 7,  # 7 days TTL
            json.dumps(task.to_dict())
        )
        
        if delay_seconds > 0:
            # Add to delayed queue (sorted set with execute time as score)
            delayed_key = f"queue:{queue_name}:delayed"
            execute_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
            await redis.zadd(delayed_key, {task.task_id: execute_at.timestamp()})
        else:
            # Add to priority queue
            queue_key = self._get_queue_key(queue_name, priority)
            priority_score = {
                QueuePriority.CRITICAL: 0,
                QueuePriority.HIGH: 1,
                QueuePriority.NORMAL: 2,
                QueuePriority.LOW: 3,
                QueuePriority.BATCH: 4,
            }[priority]
            
            await redis.zadd(queue_key, {task.task_id: priority_score})
        
        self.logger.info(f"Enqueued task {task.task_id} (type: {task_type}, priority: {priority.value})")
        return task.task_id
    
    async def enqueue_batch(
        self,
        tasks: List[Dict[str, Any]],
        queue_name: str = "default",
    ) -> List[str]:
        """Enqueue multiple tasks in a batch"""
        task_ids = []
        
        redis = await self._get_redis()
        async with redis.pipeline(transaction=True) as pipe:
            for task_data in tasks:
                task = QueueTask(
                    task_type=task_data.get("task_type", "unknown"),
                    queue_name=queue_name,
                    payload=task_data.get("payload", {}),
                    priority=QueuePriority(task_data.get("priority", "normal")),
                    metadata=task_data.get("metadata", {}),
                )
                
                task_key = self._get_task_key(task.task_id)
                pipe.setex(task_key, 86400 * 7, json.dumps(task.to_dict()))
                
                queue_key = self._get_queue_key(queue_name, task.priority)
                pipe.zadd(queue_key, {task.task_id: 2})  # Normal priority
                
                task_ids.append(task.task_id)
            
            await pipe.execute()
        
        self.logger.info(f"Enqueued {len(task_ids)} tasks in batch")
        return task_ids
    
    async def schedule(
        self,
        task_type: str,
        payload: Dict[str, Any],
        execute_at: datetime,
        queue_name: str = "default",
        **kwargs
    ) -> str:
        """Schedule a task for future execution"""
        delay = max(0, (execute_at - datetime.utcnow()).total_seconds())
        return await self.enqueue(
            task_type=task_type,
            payload=payload,
            queue_name=queue_name,
            delay_seconds=int(delay),
            **kwargs
        )
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        redis = await self._get_redis()
        
        # Get task
        task_key = self._get_task_key(task_id)
        task_data = await redis.get(task_key)
        
        if not task_data:
            return False
        
        task = QueueTask.from_dict(json.loads(task_data))
        
        if task.state not in [TaskState.PENDING, TaskState.SCHEDULED]:
            return False
        
        # Update state
        task.state = TaskState.CANCELLED
        await redis.setex(task_key, 86400, json.dumps(task.to_dict()))
        
        # Remove from queue
        queue_key = self._get_queue_key(task.queue_name, task.priority)
        await redis.zrem(queue_key, task_id)
        
        # Remove from delayed queue if applicable
        delayed_key = f"queue:{task.queue_name}:delayed"
        await redis.zrem(delayed_key, task_id)
        
        self.logger.info(f"Cancelled task {task_id}")
        return True


# ============================================================================
# Queue Consumer
# ============================================================================

class QueueConsumer:
    """
    Consumes and processes tasks from queues
    Supports concurrent processing, retries, and dead letter queue
    """
    
    def __init__(
        self,
        queue_name: str = "default",
        redis_client: Optional[aioredis.Redis] = None,
        concurrency: int = 5,
    ):
        self.queue_name = queue_name
        self._redis = redis_client
        self.concurrency = concurrency
        self._handlers: Dict[str, Callable] = {}
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._semaphore = asyncio.Semaphore(concurrency)
        self.logger = logger
    
    async def _get_redis(self) -> aioredis.Redis:
        """Get or create Redis connection"""
        if self._redis is None:
            self._redis = aioredis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                decode_responses=False,
            )
        return self._redis
    
    def register_handler(self, task_type: str, handler: Callable):
        """Register a handler for a task type"""
        self._handlers[task_type] = handler
        self.logger.info(f"Registered handler for task type: {task_type}")
    
    async def _get_next_task(self) -> Optional[QueueTask]:
        """Get next task from queue (priority order)"""
        redis = await self._get_redis()
        
        # Process delayed tasks first
        delayed_key = f"queue:{self.queue_name}:delayed"
        now = datetime.utcnow().timestamp()
        ready_tasks = await redis.zrangebyscore(delayed_key, 0, now, start=0, num=1)
        
        if ready_tasks:
            task_id = ready_tasks[0].decode() if isinstance(ready_tasks[0], bytes) else ready_tasks[0]
            await redis.zrem(delayed_key, task_id)
            
            # Move to normal queue
            normal_key = f"queue:{self.queue_name}:normal"
            await redis.zadd(normal_key, {task_id: 2})
        
        # Check queues in priority order
        for priority in QueuePriority:
            queue_key = f"queue:{self.queue_name}:{priority.value}"
            result = await redis.zpopmin(queue_key, count=1)
            
            if result:
                task_id = result[0][0].decode() if isinstance(result[0][0], bytes) else result[0][0]
                
                # Get task data
                task_key = f"task:{task_id}"
                task_data = await redis.get(task_key)
                
                if task_data:
                    task_json = task_data.decode() if isinstance(task_data, bytes) else task_data
                    return QueueTask.from_dict(json.loads(task_json))
        
        return None
    
    async def _process_task(self, task: QueueTask):
        """Process a single task"""
        redis = await self._get_redis()
        task_key = f"task:{task.task_id}"
        
        # Update state to running
        task.state = TaskState.RUNNING
        task.started_at = datetime.utcnow()
        await redis.setex(task_key, 86400 * 7, json.dumps(task.to_dict()))
        
        handler = self._handlers.get(task.task_type)
        if not handler:
            task.state = TaskState.FAILED
            task.error = f"No handler for task type: {task.task_type}"
            await redis.setex(task_key, 86400, json.dumps(task.to_dict()))
            return
        
        try:
            # Execute with timeout
            if asyncio.iscoroutinefunction(handler):
                result = await asyncio.wait_for(
                    handler(task.payload),
                    timeout=task.timeout
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(handler, task.payload),
                    timeout=task.timeout
                )
            
            task.state = TaskState.COMPLETED
            task.result = result
            task.completed_at = datetime.utcnow()
            self.logger.info(f"Task {task.task_id} completed successfully")
        
        except asyncio.TimeoutError:
            task.error = f"Task timed out after {task.timeout} seconds"
            await self._handle_failure(task, redis, task_key)
        
        except Exception as e:
            task.error = str(e)
            await self._handle_failure(task, redis, task_key)
        
        finally:
            await redis.setex(task_key, 86400 * 7, json.dumps(task.to_dict()))
    
    async def _handle_failure(self, task: QueueTask, redis: aioredis.Redis, task_key: str):
        """Handle task failure with retry logic"""
        task.retry_count += 1
        
        if task.retry_count < task.max_retries:
            # Schedule retry with exponential backoff
            delay = task.retry_delay * (task.retry_backoff ** (task.retry_count - 1))
            task.state = TaskState.RETRYING
            task.scheduled_at = datetime.utcnow() + timedelta(seconds=delay)
            
            # Add to delayed queue
            delayed_key = f"queue:{task.queue_name}:delayed"
            await redis.zadd(delayed_key, {task.task_id: task.scheduled_at.timestamp()})
            
            self.logger.warning(f"Task {task.task_id} failed, retrying in {delay}s ({task.retry_count}/{task.max_retries})")
        else:
            # Move to dead letter queue
            task.state = TaskState.DEAD
            dlq_key = f"queue:{task.queue_name}:dlq"
            await redis.lpush(dlq_key, task.task_id)
            
            self.logger.error(f"Task {task.task_id} moved to DLQ after {task.max_retries} retries")
    
    async def _worker_loop(self, worker_id: int):
        """Main worker loop"""
        self.logger.info(f"Worker {worker_id} started")
        
        while self._running:
            try:
                async with self._semaphore:
                    task = await self._get_next_task()
                    
                    if task:
                        await self._process_task(task)
                    else:
                        await asyncio.sleep(1)  # No tasks, wait
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
                await asyncio.sleep(5)
        
        self.logger.info(f"Worker {worker_id} stopped")
    
    async def start(self, num_workers: Optional[int] = None):
        """Start consumer workers"""
        if self._running:
            return
        
        self._running = True
        workers = num_workers or self.concurrency
        
        for i in range(workers):
            worker = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker)
        
        self.logger.info(f"Started {workers} consumer workers for queue: {self.queue_name}")
    
    async def stop(self):
        """Stop consumer workers"""
        self._running = False
        
        if self._workers:
            for worker in self._workers:
                worker.cancel()
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers.clear()
        
        self.logger.info("Consumer workers stopped")
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        redis = await self._get_redis()
        
        stats = {
            "queue_name": self.queue_name,
            "pending": {},
            "delayed": 0,
            "dead": 0,
            "workers": len(self._workers),
            "running": self._running,
        }
        
        for priority in QueuePriority:
            queue_key = f"queue:{self.queue_name}:{priority.value}"
            count = await redis.zcard(queue_key)
            stats["pending"][priority.value] = count
        
        delayed_key = f"queue:{self.queue_name}:delayed"
        stats["delayed"] = await redis.zcard(delayed_key)
        
        dlq_key = f"queue:{self.queue_name}:dlq"
        stats["dead"] = await redis.llen(dlq_key)
        
        return stats


# ============================================================================
# Event Publisher
# ============================================================================

class EventPublisher:
    """
    Publishes events to Redis pub/sub channels
    For real-time event distribution to subscribers
    """
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self._redis = redis_client
        self.logger = logger
    
    async def _get_redis(self) -> aioredis.Redis:
        """Get or create Redis connection"""
        if self._redis is None:
            self._redis = aioredis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                decode_responses=True,
            )
        return self._redis
    
    async def publish(
        self,
        event_type: str,
        payload: Dict[str, Any],
        channel: str = "events",
        source: str = "system",
    ) -> int:
        """
        Publish an event to a channel
        
        Returns:
            Number of subscribers that received the event
        """
        redis = await self._get_redis()
        
        event = QueueEvent(
            event_type=event_type,
            channel=channel,
            payload=payload,
            source=source,
        )
        
        count = await redis.publish(channel, event.to_json())
        self.logger.debug(f"Published event {event.event_id} to {count} subscribers")
        return count
    
    async def publish_batch(
        self,
        events: List[Dict[str, Any]],
        channel: str = "events",
    ) -> int:
        """Publish multiple events"""
        total = 0
        redis = await self._get_redis()
        
        async with redis.pipeline(transaction=False) as pipe:
            for event_data in events:
                event = QueueEvent(
                    event_type=event_data.get("event_type", "unknown"),
                    channel=channel,
                    payload=event_data.get("payload", {}),
                    source=event_data.get("source", "system"),
                )
                pipe.publish(channel, event.to_json())
            
            results = await pipe.execute()
            total = sum(r for r in results if isinstance(r, int))
        
        return total


# ============================================================================
# Event Listener
# ============================================================================

class EventListener:
    """
    Listens to events from Redis pub/sub channels
    Dispatches events to registered handlers
    """
    
    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
        self._pubsub: Optional[aioredis.client.PubSub] = None
        self._handlers: Dict[str, List[Callable]] = {}
        self._running = False
        self._listen_task: Optional[asyncio.Task] = None
        self.logger = logger
    
    async def connect(self):
        """Connect to Redis"""
        self._redis = aioredis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB,
            decode_responses=True,
        )
        self._pubsub = self._redis.pubsub()
    
    async def subscribe(self, channel: str, handler: Callable[[QueueEvent], Any]):
        """Subscribe to a channel with a handler"""
        if not self._pubsub:
            await self.connect()
        
        await self._pubsub.subscribe(channel)
        
        if channel not in self._handlers:
            self._handlers[channel] = []
        self._handlers[channel].append(handler)
        
        self.logger.info(f"Subscribed to channel: {channel}")
    
    async def unsubscribe(self, channel: str):
        """Unsubscribe from a channel"""
        if self._pubsub:
            await self._pubsub.unsubscribe(channel)
        
        if channel in self._handlers:
            del self._handlers[channel]
    
    async def _listen_loop(self):
        """Main listening loop"""
        while self._running:
            try:
                message = await self._pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                
                if message and message["type"] == "message":
                    channel = message["channel"]
                    data = message["data"]
                    
                    try:
                        event = QueueEvent.from_json(data)
                        
                        for handler in self._handlers.get(channel, []):
                            try:
                                if asyncio.iscoroutinefunction(handler):
                                    await handler(event)
                                else:
                                    handler(event)
                            except Exception as e:
                                self.logger.error(f"Handler error: {e}")
                    except Exception as e:
                        self.logger.error(f"Event parse error: {e}")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Listen loop error: {e}")
                await asyncio.sleep(1)
    
    async def start(self):
        """Start listening for events"""
        if self._running:
            return
        
        if not self._pubsub:
            await self.connect()
        
        self._running = True
        self._listen_task = asyncio.create_task(self._listen_loop())
        self.logger.info("Event listener started")
    
    async def stop(self):
        """Stop listening"""
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
        
        self.logger.info("Event listener stopped")


# ============================================================================
# Singleton Instances
# ============================================================================

_producer: Optional[QueueProducer] = None
_publisher: Optional[EventPublisher] = None


async def get_queue_producer() -> QueueProducer:
    """Get or create queue producer singleton"""
    global _producer
    if _producer is None:
        _producer = QueueProducer()
    return _producer


async def get_event_publisher() -> EventPublisher:
    """Get or create event publisher singleton"""
    global _publisher
    if _publisher is None:
        _publisher = EventPublisher()
    return _publisher


def create_queue_consumer(queue_name: str = "default", concurrency: int = 5) -> QueueConsumer:
    """Create a new queue consumer"""
    return QueueConsumer(queue_name=queue_name, concurrency=concurrency)


def create_event_listener() -> EventListener:
    """Create a new event listener"""
    return EventListener()

