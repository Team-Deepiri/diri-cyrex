"""
Task Queue Manager
Async task queue system with Redis backend
Supports priority queues, retries, and distributed execution
"""
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import asyncio
import redis.asyncio as aioredis
from ..logging_config import get_logger
from ..settings import settings

logger = get_logger("cyrex.queue_manager")


class TaskPriority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    """Task status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task definition"""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    retry_delay: int = 5  # seconds
    timeout: Optional[int] = None  # seconds
    created_at: datetime = None
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    error: Optional[str] = None
    result: Optional[Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class TaskQueueManager:
    """
    Async task queue manager with Redis backend
    Supports priority queues, retries, and distributed execution
    """
    
    def __init__(
        self,
        redis_client: Optional[aioredis.Redis] = None,
        queue_name: str = "cyrex_tasks",
    ):
        self.queue_name = queue_name
        self.redis_client = redis_client
        self.workers: List[asyncio.Task] = []
        self.task_handlers: Dict[str, Callable] = {}
        self.is_running = False
        self.logger = logger
    
    async def _get_redis(self) -> aioredis.Redis:
        """Get or create Redis connection"""
        if self.redis_client is None:
            self.redis_client = aioredis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                decode_responses=False,
            )
        return self.redis_client
    
    def _get_queue_key(self, priority: TaskPriority) -> str:
        """Get Redis queue key for priority"""
        return f"{self.queue_name}:{priority.value}"
    
    def _get_task_key(self, task_id: str) -> str:
        """Get Redis key for task data"""
        return f"{self.queue_name}:task:{task_id}"
    
    async def enqueue(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        timeout: Optional[int] = None,
    ) -> str:
        """
        Enqueue a task
        
        Returns:
            Task ID
        """
        import uuid
        task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            max_retries=max_retries,
            timeout=timeout,
            status=TaskStatus.QUEUED,
        )
        
        redis = await self._get_redis()
        
        # Store task data
        task_key = self._get_task_key(task_id)
        await redis.setex(
            task_key,
            86400 * 7,  # 7 days TTL
            json.dumps(task.to_dict(), default=str)
        )
        
        # Add to priority queue (use score for ordering)
        queue_key = self._get_queue_key(priority)
        priority_score = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.NORMAL: 2,
            TaskPriority.LOW: 3,
        }[priority]
        
        await redis.zadd(queue_key, {task_id: priority_score})
        
        self.logger.info(f"Enqueued task {task_id} (type: {task_type}, priority: {priority.value})")
        return task_id
    
    async def dequeue(self, timeout: int = 5) -> Optional[Task]:
        """
        Dequeue highest priority task
        
        Args:
            timeout: Timeout in seconds
        
        Returns:
            Task or None if no tasks available
        """
        redis = await self._get_redis()
        
        # Check queues in priority order
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]:
            queue_key = self._get_queue_key(priority)
            
            # Get task with highest priority (lowest score)
            result = await redis.zrange(queue_key, 0, 0, withscores=True)
            
            if result:
                task_id = result[0][0].decode() if isinstance(result[0][0], bytes) else result[0][0]
                
                # Remove from queue
                await redis.zrem(queue_key, task_id)
                
                # Get task data
                task_key = self._get_task_key(task_id)
                task_data = await redis.get(task_key)
                
                if task_data:
                    task_dict = json.loads(task_data)
                    task = Task.from_dict(task_dict)
                    task.status = TaskStatus.RUNNING
                    await self._update_task(task)
                    return task
        
        return None
    
    async def _update_task(self, task: Task):
        """Update task in Redis"""
        redis = await self._get_redis()
        task_key = self._get_task_key(task.task_id)
        await redis.setex(
            task_key,
            86400 * 7,
            json.dumps(task.to_dict(), default=str)
        )
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        redis = await self._get_redis()
        task_key = self._get_task_key(task_id)
        task_data = await redis.get(task_key)
        
        if task_data:
            task_dict = json.loads(task_data)
            return Task.from_dict(task_dict)
        
        return None
    
    def register_handler(self, task_type: str, handler: Callable):
        """Register task handler"""
        self.task_handlers[task_type] = handler
        self.logger.info(f"Registered handler for task type: {task_type}")
    
    async def _execute_task(self, task: Task) -> Any:
        """Execute a task"""
        handler = self.task_handlers.get(task.task_type)
        
        if not handler:
            raise ValueError(f"No handler registered for task type: {task.task_type}")
        
        # Execute with timeout if specified
        if task.timeout:
            return await asyncio.wait_for(
                handler(task.payload),
                timeout=task.timeout
            )
        else:
            return await handler(task.payload)
    
    async def _process_task(self, task: Task):
        """Process a single task with retry logic"""
        task.attempts += 1
        
        try:
            result = await self._execute_task(task)
            task.status = TaskStatus.COMPLETED
            task.result = result
            self.logger.info(f"Task {task.task_id} completed successfully")
        
        except asyncio.TimeoutError:
            task.error = "Task execution timeout"
            if task.attempts < task.max_retries:
                task.status = TaskStatus.RETRYING
                self.logger.warning(f"Task {task.task_id} timeout, retrying ({task.attempts}/{task.max_retries})")
                await asyncio.sleep(task.retry_delay)
                await self._process_task(task)
            else:
                task.status = TaskStatus.FAILED
                self.logger.error(f"Task {task.task_id} failed after {task.max_retries} attempts")
        
        except Exception as e:
            task.error = str(e)
            if task.attempts < task.max_retries:
                task.status = TaskStatus.RETRYING
                self.logger.warning(f"Task {task.task_id} failed, retrying ({task.attempts}/{task.max_retries}): {e}")
                await asyncio.sleep(task.retry_delay)
                await self._process_task(task)
            else:
                task.status = TaskStatus.FAILED
                self.logger.error(f"Task {task.task_id} failed after {task.max_retries} attempts: {e}")
        
        finally:
            await self._update_task(task)
    
    async def _worker_loop(self, worker_id: int):
        """Worker loop for processing tasks"""
        self.logger.info(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                task = await self.dequeue(timeout=5)
                
                if task:
                    await self._process_task(task)
                else:
                    await asyncio.sleep(1)  # No tasks, wait a bit
            
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def start_workers(self, num_workers: int = 3):
        """Start worker processes"""
        if self.is_running:
            self.logger.warning("Workers already running")
            return
        
        self.is_running = True
        
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker_loop(i))
            self.workers.append(worker)
        
        self.logger.info(f"Started {num_workers} workers")
    
    async def stop_workers(self):
        """Stop worker processes"""
        self.is_running = False
        
        # Wait for workers to finish current tasks
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
            self.workers.clear()
        
        self.logger.info("Workers stopped")
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        task = await self.get_task(task_id)
        
        if not task:
            return False
        
        if task.status in [TaskStatus.PENDING, TaskStatus.QUEUED]:
            task.status = TaskStatus.CANCELLED
            await self._update_task(task)
            
            # Remove from queue
            redis = await self._get_redis()
            queue_key = self._get_queue_key(task.priority)
            await redis.zrem(queue_key, task_id)
            
            self.logger.info(f"Cancelled task {task_id}")
            return True
        
        return False
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        redis = await self._get_redis()
        
        stats = {
            "total_pending": 0,
            "by_priority": {},
            "workers": len(self.workers),
            "is_running": self.is_running,
        }
        
        for priority in TaskPriority:
            queue_key = self._get_queue_key(priority)
            count = await redis.zcard(queue_key)
            stats["by_priority"][priority.value] = count
            stats["total_pending"] += count
        
        return stats


def get_queue_manager() -> TaskQueueManager:
    """Get global queue manager instance"""
    return TaskQueueManager()

