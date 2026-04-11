"""
Tool Registry
Centralized tool management for LangChain agents
Supports dynamic tool registration, validation, and execution
"""
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass
import math
from ..logging_config import get_logger
from .rate_limit_tools import ToolRateLimitExceeded

logger = get_logger("cyrex.tool_registry")

# When a Redis/Lua limiter is attached, apply this default if metadata has no rate_limit.
DEFAULT_TOOL_RATE_LIMIT_PER_MINUTE = 120

# LangChain imports with graceful fallbacks
HAS_LANGCHAIN_TOOLS = False
try:
    from langchain_core.tools import BaseTool, Tool
    HAS_LANGCHAIN_TOOLS = True
except ImportError as e:
    logger.warning(f"LangChain tools not available: {e}")
    BaseTool = None
    Tool = None


class ToolCategory(str, Enum):
    PRODUCTIVITY = "productivity"
    AUTOMATION = "automation"
    GAMIFICATION = "gamification"
    DATA = "data"
    COMMUNICATION = "communication"
    SYSTEM = "system"
    CUSTOM = "custom"


@dataclass
class ToolMetadata:
    name: str
    description: str
    category: ToolCategory
    version: str = "1.0.0"
    author: Optional[str] = None
    tags: Optional[List[str]] = None
    requires_auth: bool = False
    rate_limit: Optional[int] = None  # calls per minute
    cost_per_call: float = 0.0
    timeout: Optional[int] = None


class ToolRegistry:
    def __init__(self, load_defaults: bool = True):
        self.tools: Dict[str, BaseTool] = {}
        self.metadata: Dict[str, ToolMetadata] = {}
        self.call_counts: Dict[str, int] = {}
        self._rate_limiter: Optional[Any] = None
        self.logger = logger

        if load_defaults:
            self._load_default_tools()

    def _load_default_tools(self):
        """Load default tools (can be overridden)."""
        pass

    def reset(self):
        """Reset registry (useful for tests)."""
        self.tools = {}
        self.metadata = {}
        self.call_counts = {}

    def set_rate_limiter(self, limiter: Optional[Any]):
        self._rate_limiter = limiter

    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        return self.metadata.get(name)

    def register_tool(
        self,
        tool: BaseTool,
        metadata: Optional[ToolMetadata] = None,
        category: ToolCategory = ToolCategory.CUSTOM,
    ):
        tool_name = tool.name

        if tool_name in self.tools:
            self.logger.warning(f"Tool {tool_name} already registered, overwriting")

        self.tools[tool_name] = tool

        if metadata:
            self.metadata[tool_name] = metadata
        else:
            self.metadata[tool_name] = ToolMetadata(
                name=tool_name,
                description=tool.description,
                category=category,
            )

        self.call_counts[tool_name] = 0
        self.logger.info(f"Registered tool: {tool_name} ({category.value})")

    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[ToolMetadata] = None,
        category: ToolCategory = ToolCategory.CUSTOM,
    ):
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"

        tool = Tool(
            name=tool_name,
            description=tool_description,
            func=func,
        )

        self.register_tool(tool, metadata, category)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self.tools.get(name)

    def get_tools(
        self,
        category: Optional[ToolCategory] = None,
        tags: Optional[List[str]] = None,
    ) -> List[BaseTool]:
        result = []

        for name, tool in self.tools.items():
            metadata = self.metadata.get(name)

            if category and metadata and metadata.category != category:
                continue

            if tags and metadata:
                if not any(tag in (metadata.tags or []) for tag in tags):
                    continue

            result.append(tool)

        return result

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "description": tool.description,
                "category": self.metadata.get(name).category.value
                if self.metadata.get(name)
                else "custom",
                "version": self.metadata.get(name).version
                if self.metadata.get(name)
                else "1.0.0",
                "calls": self.call_counts.get(name, 0),
                "tags": self.metadata.get(name).tags
                if self.metadata.get(name)
                else [],
            }
            for name, tool in self.tools.items()
        ]

    def execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Any:
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")

        try:
            self.call_counts[tool_name] += 1
            return tool.invoke(tool_input)
        except Exception as e:
            self.logger.error(f"Tool execution failed: {tool_name}: {e}", exc_info=True)
            raise

    async def aexecute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Any:
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")

        metadata = self.get_metadata(tool_name)

        calls_per_minute = 0
        if self._rate_limiter is not None:
            if metadata is not None and metadata.rate_limit is not None:
                rl = int(metadata.rate_limit)
                if rl > 0:
                    calls_per_minute = rl
                # explicit 0 or negative: do not apply token bucket for this tool
            else:
                # Metadata missing or rate_limit unset — default cap so Lua bucket is always used
                calls_per_minute = DEFAULT_TOOL_RATE_LIMIT_PER_MINUTE

        if self._rate_limiter is not None and calls_per_minute > 0:
            refill_rate = calls_per_minute / 60.0
            cost = (metadata.cost_per_call if metadata else None) or 1.0
            if cost <= 0:
                cost = 1.0

            allowed, remaining = await self._rate_limiter.allow(
                tool_name=tool_name,
                user_id=user_id,
                capacity=float(calls_per_minute),
                refill_rate=refill_rate,
                cost=cost,
            )

            if not allowed:
                retry_after = 60
                if refill_rate > 0:
                    retry_after = math.ceil(max(0, cost - remaining) / refill_rate)

                raise ToolRateLimitExceeded(
                    f"Rate limit exceeded for '{tool_name}'",
                    remaining=int(remaining),
                    retry_after=int(min(retry_after, 3600)),
                )

        try:
            self.call_counts[tool_name] += 1

            if hasattr(tool, "ainvoke"):
                return await tool.ainvoke(tool_input)
            else:
                import asyncio
                return await asyncio.to_thread(tool.invoke, tool_input)

        except Exception as e:
            self.logger.error(f"Async tool failed: {tool_name}: {e}", exc_info=True)
            raise

    def unregister_tool(self, name: str):
        self.tools.pop(name, None)
        self.metadata.pop(name, None)
        self.call_counts.pop(name, None)
        self.logger.info(f"Unregistered tool: {name}")

    def get_tool_stats(self) -> Dict[str, Any]:
        return {
            "total_tools": len(self.tools),
            "total_calls": sum(self.call_counts.values()),
            "tool_calls": dict(self.call_counts),
            "by_category": {
                cat.value: len(self.get_tools(category=cat))
                for cat in ToolCategory
            },
        }


# Global registry
_global_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(
    tool: BaseTool,
    metadata: Optional[ToolMetadata] = None,
    category: ToolCategory = ToolCategory.CUSTOM,
):
    get_tool_registry().register_tool(tool, metadata, category)