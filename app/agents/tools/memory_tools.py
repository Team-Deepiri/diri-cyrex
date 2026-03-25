"""
Memory Tools for Agents
Tools for memory management and retrieval
"""
from typing import List, Optional
from ...core.types import MemoryType
from ...core.memory_manager import get_memory_manager
from ...logging_config import get_logger

logger = get_logger("cyrex.agent.tools.memory")


async def register_memory_tools(agent):
    """Register memory tools with an agent"""
    memory_manager = await get_memory_manager()
    
    async def search_memories(query: str, limit: int = 10, memory_type: Optional[str] = None) -> List[str]:
        """Search memories by query"""
        try:
            mem_type = MemoryType(memory_type) if memory_type else None
            memories = await memory_manager.search_memories(
                query=query,
                session_id=agent.session_id,
                memory_type=mem_type,
                limit=limit,
            )
            return [m.content for m in memories]
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []
    
    async def store_memory(content: str, memory_type: str = "long_term", importance: float = 0.5) -> str:
        """Store information in memory"""
        try:
            mem_type = MemoryType(memory_type)
            memory_id = await memory_manager.store_memory(
                content=content,
                memory_type=mem_type,
                session_id=agent.session_id,
                importance=importance,
            )
            return f"Memory stored: {memory_id}"
        except Exception as e:
            logger.error(f"Memory storage failed: {e}")
            return f"Error: {str(e)}"
    
    async def get_context() -> dict:
        """Get current context from memories"""
        try:
            return await memory_manager.build_context(
                session_id=agent.session_id,
            )
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return {}
    
    agent.register_tool("search_memories", search_memories, "Search memories by query")
    agent.register_tool("store_memory", store_memory, "Store information in memory")
    agent.register_tool("get_context", get_context, "Get current context from memories")

