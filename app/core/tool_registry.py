"""
Tool Registry
Centralized tool management for LangChain agents
Supports dynamic tool registration, validation, and execution
"""
from typing import Dict, List, Optional, Any, Callable, Type
from enum import Enum
from dataclasses import dataclass
from langchain_core.tools import BaseTool, Tool
from pydantic import BaseModel, Field
import inspect
from ..logging_config import get_logger

logger = get_logger("cyrex.tool_registry")


class ToolCategory(str, Enum):
    """Tool categories"""
    PRODUCTIVITY = "productivity"
    AUTOMATION = "automation"
    GAMIFICATION = "gamification"
    DATA = "data"
    COMMUNICATION = "communication"
    SYSTEM = "system"
    CUSTOM = "custom"


@dataclass
class ToolMetadata:
    """Metadata for registered tools"""
    name: str
    description: str
    category: ToolCategory
    version: str = "1.0.0"
    author: Optional[str] = None
    tags: List[str] = None
    requires_auth: bool = False
    rate_limit: Optional[int] = None  # Calls per minute
    cost_per_call: float = 0.0
    timeout: Optional[int] = None  # Seconds


class ToolRegistry:
    """
    Centralized registry for LangChain tools
    Manages tool registration, validation, and execution
    """
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.metadata: Dict[str, ToolMetadata] = {}
        self.call_counts: Dict[str, int] = {}
        self.logger = logger
    
    def register_tool(
        self,
        tool: BaseTool,
        metadata: Optional[ToolMetadata] = None,
        category: ToolCategory = ToolCategory.CUSTOM,
    ):
        """
        Register a tool in the registry
        
        Args:
            tool: LangChain BaseTool instance
            metadata: Optional tool metadata
            category: Tool category
        """
        tool_name = tool.name
        
        if tool_name in self.tools:
            self.logger.warning(f"Tool {tool_name} already registered, overwriting")
        
        self.tools[tool_name] = tool
        
        if metadata:
            self.metadata[tool_name] = metadata
        else:
            # Create default metadata
            self.metadata[tool_name] = ToolMetadata(
                name=tool_name,
                description=tool.description,
                category=category,
            )
        
        self.call_counts[tool_name] = 0
        self.logger.info(f"Registered tool: {tool_name} (category: {category.value})")
    
    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[ToolMetadata] = None,
        category: ToolCategory = ToolCategory.CUSTOM,
    ):
        """
        Register a Python function as a tool
        
        Args:
            func: Python function to register
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            metadata: Optional tool metadata
            category: Tool category
        """
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"
        
        # Create LangChain tool from function
        tool = Tool(
            name=tool_name,
            description=tool_description,
            func=func,
        )
        
        self.register_tool(tool, metadata, category)
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def get_tools(
        self,
        category: Optional[ToolCategory] = None,
        tags: Optional[List[str]] = None,
    ) -> List[BaseTool]:
        """Get tools filtered by category and/or tags"""
        tools = []
        
        for name, tool in self.tools.items():
            metadata = self.metadata.get(name)
            
            if category and metadata and metadata.category != category:
                continue
            
            if tags and metadata:
                if not any(tag in (metadata.tags or []) for tag in tags):
                    continue
            
            tools.append(tool)
        
        return tools
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with metadata"""
        result = []
        
        for name, tool in self.tools.items():
            metadata = self.metadata.get(name)
            result.append({
                "name": name,
                "description": tool.description,
                "category": metadata.category.value if metadata else "custom",
                "version": metadata.version if metadata else "1.0.0",
                "calls": self.call_counts.get(name, 0),
                "tags": metadata.tags if metadata else [],
            })
        
        return result
    
    def execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Any:
        """
        Execute a tool with validation and rate limiting
        
        Args:
            tool_name: Name of tool to execute
            tool_input: Input parameters for tool
            user_id: Optional user ID for rate limiting
        
        Returns:
            Tool execution result
        """
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")
        
        metadata = self.metadata.get(tool_name)
        
        # Check rate limiting
        if metadata and metadata.rate_limit:
            # TODO: Implement rate limiting per user
            pass
        
        # Execute tool
        try:
            self.call_counts[tool_name] = self.call_counts.get(tool_name, 0) + 1
            result = tool.invoke(tool_input)
            self.logger.info(f"Executed tool: {tool_name}")
            return result
        
        except Exception as e:
            self.logger.error(f"Tool execution failed: {tool_name}, error: {e}", exc_info=True)
            raise
    
    async def aexecute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Any:
        """Async execute tool"""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")
        
        try:
            self.call_counts[tool_name] = self.call_counts.get(tool_name, 0) + 1
            
            if hasattr(tool, 'ainvoke'):
                result = await tool.ainvoke(tool_input)
            else:
                import asyncio
                result = await asyncio.to_thread(tool.invoke, tool_input)
            
            self.logger.info(f"Async executed tool: {tool_name}")
            return result
        
        except Exception as e:
            self.logger.error(f"Async tool execution failed: {tool_name}, error: {e}", exc_info=True)
            raise
    
    def unregister_tool(self, name: str):
        """Unregister a tool"""
        if name in self.tools:
            del self.tools[name]
            if name in self.metadata:
                del self.metadata[name]
            if name in self.call_counts:
                del self.call_counts[name]
            self.logger.info(f"Unregistered tool: {name}")
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get statistics about tool usage"""
        total_calls = sum(self.call_counts.values())
        
        return {
            "total_tools": len(self.tools),
            "total_calls": total_calls,
            "tool_calls": dict(self.call_counts),
            "by_category": {
                cat.value: len(self.get_tools(category=cat))
                for cat in ToolCategory
            },
        }


# Global tool registry instance
_global_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get global tool registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(
    tool: BaseTool,
    metadata: Optional[ToolMetadata] = None,
    category: ToolCategory = ToolCategory.CUSTOM,
):
    """Convenience function to register tool in global registry"""
    registry = get_tool_registry()
    registry.register_tool(tool, metadata, category)

