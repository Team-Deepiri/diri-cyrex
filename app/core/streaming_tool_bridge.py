"""
Streaming Tool Bridge

Monitors the LLM token stream and:
1. Detects tool call patterns early (before the full call is generated)
2. Pre-allocates resources (DB connections, API sessions) for predicted tools
3. Streams partial results back as they complete
4. Fuses tool results with model generation seamlessly

This is the "execute immediately, merge seamlessly" component of PDGE.
It does NOT speculatively execute tools -- it only prepares resources.
"""
import asyncio
import time
from typing import Dict, Any, Optional, List, AsyncIterator, Callable
from collections import defaultdict
from ..logging_config import get_logger

logger = get_logger("cyrex.streaming_bridge")


class ResourcePool:
    """
    Pre-allocated resource pool for tools.
    Resources are warmed up (connections opened, sessions created)
    but no tool logic is executed until the model confirms the call.
    """
    def __init__(self):
        self._pools: Dict[str, asyncio.Queue] = {}
        self._warmup_functions: Dict[str, Callable] = {}

    def register_warmup(self, resource_group: str, warmup_fn: Callable) -> None:
        """Register a warmup function for a resource group."""
        self._warmup_functions[resource_group] = warmup_fn
        if resource_group not in self._pools:
            self._pools[resource_group] = asyncio.Queue(maxsize=5)

    async def warm(self, resource_group: str) -> None:
        """Pre-allocate a resource for the given group."""
        if resource_group not in self._warmup_functions:
            return
        try:
            pool = self._pools.get(resource_group)
            if pool and not pool.full():
                resource = await self._warmup_functions[resource_group]()
                await pool.put(resource)
                logger.debug(f"Pre-warmed resource for group: {resource_group}")
        except Exception as e:
            logger.debug(f"Resource warmup failed for {resource_group}: {e}")

    async def acquire(self, resource_group: str) -> Optional[Any]:
        """Get a pre-warmed resource if available."""
        pool = self._pools.get(resource_group)
        if pool:
            try:
                return pool.get_nowait()
            except asyncio.QueueEmpty:
                pass
        return None

    async def release(self, resource_group: str, resource: Any) -> None:
        """Return a resource to the pool."""
        pool = self._pools.get(resource_group)
        if pool:
            try:
                pool.put_nowait(resource)
            except asyncio.QueueFull:
                pass


class ToolCallDetector:
    """
    Lightweight pattern matcher that detects tool call intent
    in partial token streams.

    For Ollama native tool calling, the model returns structured JSON
    tool_calls, not text patterns. So this detector works at the
    message level (after each LLM turn), not at the token level.

    For streaming scenarios, we can detect tool_call fields in
    partial JSON responses.
    """

    # Common tool-related keywords that suggest a tool call is coming
    TOOL_HINT_KEYWORDS = {
        "spreadsheet": "spreadsheet",
        "set_cell": "spreadsheet",
        "get_cell": "spreadsheet",
        "calculate": "compute",
        "search": "search",
        "query": "database",
        "api": "network",
        "http": "network",
        "memory": "memory",
        "store": "memory",
    }

    @classmethod
    def detect_from_text(cls, text: str) -> List[str]:
        """
        Detect likely resource groups from user input text.
        Returns list of resource groups to pre-warm.
        """
        text_lower = text.lower()
        groups = set()
        for keyword, group in cls.TOOL_HINT_KEYWORDS.items():
            if keyword in text_lower:
                groups.add(group)
        return list(groups)


class StreamingToolBridge:
    """
    Main bridge that connects LLM streaming with PDGE execution.

    Flow:
    1. User sends message
    2. Bridge analyzes user text for tool hints
    3. Pre-warms relevant resource pools
    4. LLM generates response (possibly with tool_calls)
    5. PDGE executes tool calls in parallel
    6. Results flow back through the bridge
    """

    def __init__(self, resource_pool: Optional[ResourcePool] = None):
        self.resource_pool = resource_pool or ResourcePool()
        self._pre_warm_tasks: List[asyncio.Task] = []

    async def pre_warm_for_input(self, user_input: str) -> None:
        """
        Analyze user input and pre-warm likely resource groups.
        This runs in parallel with LLM inference -- zero added latency.
        """
        predicted_groups = ToolCallDetector.detect_from_text(user_input)
        if predicted_groups:
            logger.debug(f"Pre-warming resource groups: {predicted_groups}")
            tasks = [self.resource_pool.warm(g) for g in predicted_groups]
            self._pre_warm_tasks = [asyncio.create_task(t) for t in tasks]

    async def cleanup(self) -> None:
        """Cancel any pending pre-warm tasks."""
        for task in self._pre_warm_tasks:
            if not task.done():
                task.cancel()
        self._pre_warm_tasks = []


# Singleton bridge instance
_bridge: Optional[StreamingToolBridge] = None


def get_bridge() -> StreamingToolBridge:
    """Get or create the singleton streaming bridge."""
    global _bridge
    if _bridge is None:
        _bridge = StreamingToolBridge()
    return _bridge

