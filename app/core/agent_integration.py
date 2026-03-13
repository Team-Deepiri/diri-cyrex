"""
Agent Integration Layer
Connects Agent Playground with prompts from app/agents/prompts and tools from app/agents/tools
"""
from typing import Optional, Dict, Any, List
from langchain_core.tools import StructuredTool
from ..logging_config import get_logger

logger = get_logger("cyrex.agent_integration")


# ============================================================================
# Prompt Loading
# ============================================================================

def load_prompt_for_agent_type(agent_type: str, fallback_prompt: Optional[str] = None) -> str:
    """
    Load specialized prompt based on agent type
    
    Args:
        agent_type: The type of agent (vendor_fraud, task_decomposer, etc.)
        fallback_prompt: Fallback prompt if agent_type not found
    
    Returns:
        System prompt string
    """
    # Default fallback
    if fallback_prompt is None:
        fallback_prompt = "You are a helpful AI assistant with access to tools."
    
    # Map agent types to prompts
    try:
        if agent_type == "vendor_fraud":
            from ..agents.prompts.vendor_fraud_prompts import VENDOR_FRAUD_SYSTEM_PROMPT
            return VENDOR_FRAUD_SYSTEM_PROMPT.format(task="", context="")
        
        elif agent_type == "task_decomposer":
            from ..agents.prompts.task_decomposer_prompts import TASK_DECOMPOSER_PROMPT
            return TASK_DECOMPOSER_PROMPT
        
        elif agent_type == "creative_sparker":
            from ..agents.prompts.creative_sparker_prompts import CREATIVE_SPARKER_PROMPT
            return CREATIVE_SPARKER_PROMPT
        
        elif agent_type == "quality_assurance":
            from ..agents.prompts.quality_assurance_prompts import QUALITY_ASSURANCE_PROMPT
            return QUALITY_ASSURANCE_PROMPT
        
        elif agent_type == "time_optimizer":
            from ..agents.prompts.time_optimizer_prompts import TIME_OPTIMIZER_PROMPT
            return TIME_OPTIMIZER_PROMPT
        
        elif agent_type == "engagement_specialist":
            from ..agents.prompts.engagement_specialist_prompts import ENGAGEMENT_SPECIALIST_PROMPT
            return ENGAGEMENT_SPECIALIST_PROMPT
        
        elif agent_type == "react":
            from ..agents.prompts.react_agent_prompts import REACT_AGENT_SYSTEM_PROMPT
            return REACT_AGENT_SYSTEM_PROMPT
        
        else:
            logger.info(f"No specialized prompt for agent_type '{agent_type}', using fallback")
            return fallback_prompt
    
    except ImportError as e:
        logger.warning(f"Failed to load prompt for agent_type '{agent_type}': {e}")
        return fallback_prompt


def get_available_agent_types() -> List[str]:
    """
    Get list of available agent types with specialized prompts
    
    Returns:
        List of agent type strings
    """
    return [
        "vendor_fraud",
        "task_decomposer",
        "creative_sparker",
        "quality_assurance",
        "time_optimizer",
        "engagement_specialist",
        "react",
    ]


# ============================================================================
# Tool Registration
# ============================================================================

async def register_all_agent_tools(tool_registry, instance_id: Optional[str] = None) -> Dict[str, int]:
    """
    Register all tools from app/agents/tools with the ToolRegistry
    
    Args:
        tool_registry: ToolRegistry instance
        instance_id: Optional instance ID for instance-aware tools
    
    Returns:
        Dict with counts of registered tools by category
    """
    from .tool_registry import ToolCategory
    import asyncio
    
    stats = {
        "api_tools": 0,
        "memory_tools": 0,
        "utility_tools": 0,
        "errors": []
    }
    
    # Create a tool collector that bridges BaseAgent tool format to ToolRegistry
    class ToolCollector:
        """Collects tools registered via BaseAgent.register_tool() interface"""
        def __init__(self):
            self.collected_tools = []
        
        def register_tool(self, name: str, func, description: str):
            """Collect tool in BaseAgent format"""
            self.collected_tools.append({
                "name": name,
                "func": func,
                "description": description
            })
    
    # Helper to create sync wrapper for async functions
    def make_sync_wrapper(async_func):
        """Create sync wrapper for async function that works in ThreadPoolExecutor"""
        def sync_wrapper(*args, **kwargs):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(async_func(*args, **kwargs))
            finally:
                loop.close()
        return sync_wrapper
    
    # Register API tools
    try:
        from ..agents.tools.api_tools import register_api_tools
        collector = ToolCollector()
        await register_api_tools(collector)
        
        for tool_def in collector.collected_tools:
            try:
                func = tool_def["func"]
                # Check if async and wrap if needed
                if asyncio.iscoroutinefunction(func):
                    func = make_sync_wrapper(func)
                
                # Convert to StructuredTool for native tool calling
                tool = StructuredTool.from_function(
                    func=func,
                    name=tool_def["name"],
                    description=tool_def["description"]
                )
                tool_registry.register_tool(tool, category=ToolCategory.CUSTOM)
                stats["api_tools"] += 1
            except Exception as e:
                logger.warning(f"Failed to register API tool {tool_def['name']}: {e}", exc_info=True)
                stats["errors"].append(f"api_tool_{tool_def['name']}: {str(e)}")
        
        logger.info(f"Registered {stats['api_tools']} API tools")
    except Exception as e:
        logger.warning(f"Failed to register API tools: {e}", exc_info=True)
        stats["errors"].append(f"api_tools: {str(e)}")
    
    # Register memory tools
    try:
        from ..agents.tools.memory_tools import register_memory_tools
        collector = ToolCollector()
        await register_memory_tools(collector)
        
        for tool_def in collector.collected_tools:
            try:
                func = tool_def["func"]
                if asyncio.iscoroutinefunction(func):
                    func = make_sync_wrapper(func)
                
                tool = StructuredTool.from_function(
                    func=func,
                    name=tool_def["name"],
                    description=tool_def["description"]
                )
                tool_registry.register_tool(tool, category=ToolCategory.CUSTOM)
                stats["memory_tools"] += 1
            except Exception as e:
                logger.warning(f"Failed to register memory tool {tool_def['name']}: {e}", exc_info=True)
                stats["errors"].append(f"memory_tool_{tool_def['name']}: {str(e)}")
        
        logger.info(f"Registered {stats['memory_tools']} memory tools")
    except Exception as e:
        logger.warning(f"Failed to register memory tools: {e}", exc_info=True)
        stats["errors"].append(f"memory_tools: {str(e)}")
    
    # Register utility tools
    try:
        from ..agents.tools.utility_tools import register_utility_tools
        collector = ToolCollector()
        await register_utility_tools(collector)
        
        for tool_def in collector.collected_tools:
            try:
                func = tool_def["func"]
                if asyncio.iscoroutinefunction(func):
                    func = make_sync_wrapper(func)
                
                tool = StructuredTool.from_function(
                    func=func,
                    name=tool_def["name"],
                    description=tool_def["description"]
                )
                tool_registry.register_tool(tool, category=ToolCategory.CUSTOM)
                stats["utility_tools"] += 1
            except Exception as e:
                logger.warning(f"Failed to register utility tool {tool_def['name']}: {e}", exc_info=True)
                stats["errors"].append(f"utility_tool_{tool_def['name']}: {str(e)}")
        
        logger.info(f"Registered {stats['utility_tools']} utility tools")
    except Exception as e:
        logger.warning(f"Failed to register utility tools: {e}", exc_info=True)
        stats["errors"].append(f"utility_tools: {str(e)}")
    
    total = stats["api_tools"] + stats["memory_tools"] + stats["utility_tools"]
    logger.info(f"Agent tool registration complete: {total} tools registered")
    
    if stats["errors"]:
        logger.warning(f"Encountered {len(stats['errors'])} errors during tool registration")
    
    return stats


def get_tool_registry_stats(tool_registry) -> Dict[str, Any]:
    """
    Get statistics about registered tools
    
    Args:
        tool_registry: ToolRegistry instance
    
    Returns:
        Dict with tool statistics
    """
    try:
        tools = tool_registry.get_tools()
        return {
            "total_tools": len(tools),
            "tool_names": [t.name for t in tools],
            "categories": tool_registry.get_tool_stats() if hasattr(tool_registry, "get_tool_stats") else {}
        }
    except Exception as e:
        logger.error(f"Failed to get tool registry stats: {e}")
        return {"error": str(e)}


# ============================================================================
# Integration Helper
# ============================================================================

async def initialize_agent_system(tool_registry) -> Dict[str, Any]:
    """
    Initialize the full agent system with prompts and tools
    
    Args:
        tool_registry: ToolRegistry instance
    
    Returns:
        Dict with initialization results
    """
    logger.info("Initializing agent system...")
    
    # Register all agent tools
    tool_stats = await register_all_agent_tools(tool_registry)
    
    # Get available agent types
    agent_types = get_available_agent_types()
    
    logger.info(f"Agent system initialized: {len(agent_types)} agent types, {tool_stats.get('api_tools', 0) + tool_stats.get('memory_tools', 0) + tool_stats.get('utility_tools', 0)} tools")
    
    return {
        "agent_types": agent_types,
        "tool_stats": tool_stats,
        "registry_stats": get_tool_registry_stats(tool_registry)
    }

