"""
Request Queue Manager
Handles queuing and batching of LLM requests when Ollama is at capacity
"""
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import hashlib
import json
import time
from collections import defaultdict
from ..logging_config import get_logger
from .queue_system import QueueProducer, QueueConsumer, QueueTask, QueuePriority, TaskState

logger = get_logger("cyrex.request_queue")

# Global state for tracking concurrent requests
_active_requests = 0
_max_concurrent = 0
_request_lock = asyncio.Lock()
_request_history: List[Dict[str, Any]] = []

# Ollama capacity (from docker-compose.dev.yml: OLLAMA_NUM_PARALLEL: "4")
OLLAMA_MAX_PARALLEL = 4

# Batching configuration
BATCH_WINDOW_MS = 100  # Wait 100ms to collect similar requests
BATCH_SIMILARITY_THRESHOLD = 0.8  # 80% similarity to batch together
MAX_BATCH_SIZE = 4  # Max requests per batch


class RequestPriority(str, Enum):
    """Request priority levels"""
    CRITICAL = "critical"  # Process immediately, skip queue
    HIGH = "high"  # High priority in queue
    NORMAL = "normal"  # Normal priority
    LOW = "low"  # Low priority
    BATCH = "batch"  # Can be batched with similar requests


@dataclass
class QueuedRequest:
    """Represents a queued LLM request"""
    request_id: str
    user_input: str
    user_id: Optional[str] = None
    workflow_id: Optional[str] = None
    use_rag: bool = True
    use_tools: bool = True
    use_langgraph: bool = False
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    stream: bool = False
    priority: RequestPriority = RequestPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "request_id": self.request_id,
            "user_input": self.user_input,
            "user_id": self.user_id,
            "workflow_id": self.workflow_id,
            "use_rag": self.use_rag,
            "use_tools": self.use_tools,
            "use_langgraph": self.use_langgraph,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "model": self.model,
            "stream": self.stream,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueuedRequest":
        """Create from dictionary"""
        return cls(
            request_id=data["request_id"],
            user_input=data["user_input"],
            user_id=data.get("user_id"),
            workflow_id=data.get("workflow_id"),
            use_rag=data.get("use_rag", True),
            use_tools=data.get("use_tools", True),
            use_langgraph=data.get("use_langgraph", False),
            max_tokens=data.get("max_tokens"),
            system_prompt=data.get("system_prompt"),
            model=data.get("model"),
            stream=data.get("stream", False),
            priority=RequestPriority(data.get("priority", "normal")),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


class RequestQueueManager:
    """
    Manages request queuing and batching for LLM requests
    """
    
    def __init__(self, max_parallel: int = OLLAMA_MAX_PARALLEL):
        self.max_parallel = max_parallel
        self.queue_producer = QueueProducer()
        self.queue_consumer = QueueConsumer(
            queue_name="llm_requests",
            concurrency=max_parallel
        )
        self.pending_batches: Dict[str, List[QueuedRequest]] = defaultdict(list)
        self.batch_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.request_results: Dict[str, Any] = {}
        self.result_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._running = False
        self._processor_func: Optional[Callable[[QueuedRequest], Awaitable[Dict[str, Any]]]] = None
        
    async def start(self, processor_func: Optional[Callable[[QueuedRequest], Awaitable[Dict[str, Any]]]] = None):
        """Start the queue consumer"""
        if not self._running:
            # Store processor function if provided
            if processor_func:
                self._processor_func = processor_func
            
            # Register handler for llm_request tasks
            async def handler_wrapper(task: QueueTask):
                """Handler for queued requests"""
                return await self._process_queued_request(task, self._processor_func)
            
            self.queue_consumer.register_handler("llm_request", handler_wrapper)
            
            # Start consumer if it has a start method
            if hasattr(self.queue_consumer, 'start'):
                await self.queue_consumer.start()
            self._running = True
            logger.info("Request queue manager started")
    
    async def stop(self):
        """Stop the queue consumer"""
        if self._running:
            if hasattr(self.queue_consumer, 'stop'):
                await self.queue_consumer.stop()
            self._running = False
            logger.info("Request queue manager stopped")
    
    def _get_batch_key(self, request: QueuedRequest) -> str:
        """Generate a batch key for similar requests"""
        # Batch by: model, use_rag, use_tools, use_langgraph
        key_parts = [
            request.model or "default",
            str(request.use_rag),
            str(request.use_tools),
            str(request.use_langgraph),
        ]
        return ":".join(key_parts)
    
    def _calculate_similarity(self, req1: QueuedRequest, req2: QueuedRequest) -> float:
        """Calculate similarity between two requests (0.0 to 1.0)"""
        # Simple similarity based on configuration
        similarity = 0.0
        total_weight = 0.0
        
        # Model similarity (weight: 0.3)
        if req1.model == req2.model:
            similarity += 0.3
        total_weight += 0.3
        
        # Configuration similarity (weight: 0.7)
        config_similarity = 0.0
        if req1.use_rag == req2.use_rag:
            config_similarity += 0.25
        if req1.use_tools == req2.use_tools:
            config_similarity += 0.25
        if req1.use_langgraph == req2.use_langgraph:
            config_similarity += 0.25
        if req1.max_tokens == req2.max_tokens:
            config_similarity += 0.25
        
        similarity += config_similarity * 0.7
        total_weight += 0.7
        
        return similarity / total_weight if total_weight > 0 else 0.0
    
    async def get_concurrent_requests(self) -> int:
        """Get current number of concurrent requests"""
        async with _request_lock:
            return _active_requests
    
    async def get_max_concurrent(self) -> int:
        """Get maximum concurrent requests seen"""
        async with _request_lock:
            return _max_concurrent
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        concurrent = await self.get_concurrent_requests()
        max_concurrent = await self.get_max_concurrent()
        
        # Get queue length
        queue_length = await self.get_queue_length("llm_requests")
        
        return {
            "concurrent_requests": concurrent,
            "max_concurrent_seen": max_concurrent,
            "ollama_capacity": self.max_parallel,
            "queue_length": queue_length,
            "available_slots": max(0, self.max_parallel - concurrent),
            "pending_batches": sum(len(batch) for batch in self.pending_batches.values()),
        }
    
    async def should_queue(self, priority: RequestPriority = RequestPriority.NORMAL) -> bool:
        """
        Determine if a request should be queued
        
        Args:
            priority: Request priority
            
        Returns:
            True if request should be queued, False if can process immediately
        """
        # Critical priority always processes immediately
        if priority == RequestPriority.CRITICAL:
            return False
        
        concurrent = await self.get_concurrent_requests()
        
        # Queue if at capacity (unless critical priority)
        return concurrent >= self.max_parallel
    
    async def enqueue_request(
        self,
        request: QueuedRequest,
        processor_func: Callable[[QueuedRequest], Awaitable[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Enqueue a request for processing
        
        Args:
            request: The request to enqueue
            processor_func: Function to process the request
            
        Returns:
            Dictionary with task_id and status
        """
        # Check if should batch
        if request.priority == RequestPriority.BATCH:
            return await self._try_batch_request(request, processor_func)
        
        # Check if should queue
        if await self.should_queue(request.priority):
            # Enqueue in Redis queue
            task = QueueTask(
                task_type="llm_request",
                queue_name="llm_requests",
                payload=request.to_dict(),
                priority=QueuePriority(request.priority.value.upper()),
            )
            
            # Use enqueue method (not enqueue_task which may not exist)
            task_id = await self.queue_producer.enqueue(
                task_type="llm_request",
                payload=request.to_dict(),
                queue_name="llm_requests",
                priority=QueuePriority(request.priority.value.upper()),
            )
            
            # Register handler for this task type if not already registered
            if "llm_request" not in self.queue_consumer._handlers:
                self.queue_consumer.register_handler(
                    "llm_request",
                    lambda task: self._process_queued_request(task, processor_func)
                )
            
            logger.info(f"Request {request.request_id} queued (task {task_id})")
            
            queue_length = await self.get_queue_length("llm_requests")
            
            return {
                "task_id": task_id,
                "request_id": request.request_id,
                "status": "queued",
                "queue_position": queue_length,
            }
        else:
            # Process immediately
            return await self._process_immediately(request, processor_func)
    
    async def _try_batch_request(
        self,
        request: QueuedRequest,
        processor_func: Callable[[QueuedRequest], Awaitable[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Try to batch request with similar pending requests"""
        batch_key = self._get_batch_key(request)
        
        async with self.batch_locks[batch_key]:
            batch = self.pending_batches[batch_key]
            
            # Check if we can add to existing batch
            if len(batch) > 0:
                # Check similarity with first request in batch
                similarity = self._calculate_similarity(request, batch[0])
                if similarity >= BATCH_SIMILARITY_THRESHOLD and len(batch) < MAX_BATCH_SIZE:
                    batch.append(request)
                    logger.info(f"Request {request.request_id} added to batch {batch_key} (size: {len(batch)})")
                    
                    # If batch is full, process it
                    if len(batch) >= MAX_BATCH_SIZE:
                        return await self._process_batch(batch_key, batch, processor_func)
                    
                    # Otherwise, wait for batch window
                    await asyncio.sleep(BATCH_WINDOW_MS / 1000.0)
                    if len(batch) >= 2:  # At least 2 requests
                        return await self._process_batch(batch_key, batch, processor_func)
                    else:
                        # Batch didn't fill, process individually
                        self.pending_batches[batch_key].remove(request)
                        return await self._process_immediately(request, processor_func)
            
            # Start new batch
            self.pending_batches[batch_key].append(request)
            logger.info(f"Request {request.request_id} started new batch {batch_key}")
            
            # Wait for batch window
            await asyncio.sleep(BATCH_WINDOW_MS / 1000.0)
            
            # Check if batch has more requests
            if len(self.pending_batches[batch_key]) >= 2:
                return await self._process_batch(batch_key, self.pending_batches[batch_key], processor_func)
            else:
                # Only one request, process individually
                self.pending_batches[batch_key].clear()
                return await self._process_immediately(request, processor_func)
    
    async def _process_batch(
        self,
        batch_key: str,
        batch: List[QueuedRequest],
        processor_func: Callable[[QueuedRequest], Awaitable[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Process a batch of requests"""
        logger.info(f"Processing batch {batch_key} with {len(batch)} requests")
        
        # Clear batch
        self.pending_batches[batch_key].clear()
        
        # Process all requests in batch concurrently
        tasks = [processor_func(req) for req in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Return result for first request (caller)
        first_result = results[0]
        if isinstance(first_result, Exception):
            raise first_result
        
        # Store results for other requests
        for i, (req, result) in enumerate(zip(batch[1:], results[1:])):
            async with self.result_locks[req.request_id]:
                if isinstance(result, Exception):
                    self.request_results[req.request_id] = {
                        "success": False,
                        "error": str(result),
                    }
                else:
                    self.request_results[req.request_id] = result
        
        return first_result
    
    async def _process_immediately(
        self,
        request: QueuedRequest,
        processor_func: Callable[[QueuedRequest], Awaitable[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Process request immediately"""
        global _active_requests, _max_concurrent
        
        async with _request_lock:
            _active_requests += 1
            _max_concurrent = max(_max_concurrent, _active_requests)
        
        try:
            logger.info(f"Processing request {request.request_id} immediately")
            result = await processor_func(request)
            return result
        finally:
            async with _request_lock:
                _active_requests = max(0, _active_requests - 1)
    
    async def _process_queued_request(
        self,
        task: QueueTask,
        processor_func: Optional[Callable[[QueuedRequest], Awaitable[Dict[str, Any]]]] = None
    ) -> Dict[str, Any]:
        """Process a queued request (called by queue consumer)"""
        request = QueuedRequest.from_dict(task.payload)
        
        # Use provided processor_func or fall back to stored one
        func_to_use = processor_func or self._processor_func
        
        if func_to_use is None:
            logger.error(f"Processor function not available for request {request.request_id}")
            return {
                "success": False,
                "error": "Processor function not available - handler not properly configured",
                "request_id": request.request_id,
            }
        
        return await self._process_immediately(request, func_to_use)
    
    async def get_request_result(self, request_id: str, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Get result for a request (for polling)"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            async with self.result_locks[request_id]:
                if request_id in self.request_results:
                    result = self.request_results.pop(request_id)
                    return result
            
            await asyncio.sleep(0.1)  # Poll every 100ms
        
        return None  # Timeout


# Global instance
_request_queue_manager: Optional[RequestQueueManager] = None


def get_request_queue_manager() -> RequestQueueManager:
    """Get or create global request queue manager"""
    global _request_queue_manager
    if _request_queue_manager is None:
        _request_queue_manager = RequestQueueManager()
    return _request_queue_manager

