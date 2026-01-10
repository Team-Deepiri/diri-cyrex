"""
Enhanced Memory Tools for Agents
Comprehensive memory management with vector search, persistence, and context building
"""
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import asyncio
import json
import uuid
from ...core.memory_manager import get_memory_manager, MemoryManager
from ...core.types import Memory, MemoryType
from ...database.postgres import get_postgres_manager
from ...logging_config import get_logger

logger = get_logger("cyrex.agent.tools.memory")


class MemorySearchMode(str, Enum):
    """Memory search modes"""
    SEMANTIC = "semantic"  # Vector similarity search
    KEYWORD = "keyword"    # Text-based search
    HYBRID = "hybrid"      # Combined approach
    RECENCY = "recency"    # Most recent first
    IMPORTANCE = "importance"  # Most important first


@dataclass
class MemorySearchResult:
    """Search result with metadata"""
    memory_id: str
    content: str
    score: float
    memory_type: MemoryType
    metadata: Dict[str, Any]
    created_at: datetime


class EnhancedMemoryTools:
    """
    Enhanced memory tools for agent use
    Provides comprehensive memory operations with context awareness
    """
    
    def __init__(self, session_id: Optional[str] = None, user_id: Optional[str] = None):
        self.session_id = session_id
        self.user_id = user_id
        self._memory_manager: Optional[MemoryManager] = None
        self._working_memory: Dict[str, Any] = {}  # Fast in-memory cache
        self.logger = logger
    
    async def _get_manager(self) -> MemoryManager:
        """Get or initialize memory manager"""
        if not self._memory_manager:
            self._memory_manager = await get_memory_manager()
        return self._memory_manager
    
    # ========================================================================
    # Core Memory Operations
    # ========================================================================
    
    async def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.LONG_TERM,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """
        Store information in memory
        
        Args:
            content: The content to remember
            memory_type: Type of memory (short_term, long_term, episodic, semantic)
            importance: Importance score (0.0 - 1.0)
            metadata: Additional metadata
            tags: Tags for categorization
            ttl: Time to live in seconds (None for permanent)
        
        Returns:
            Memory ID
        """
        manager = await self._get_manager()
        
        full_metadata = metadata or {}
        if tags:
            full_metadata["tags"] = tags
        
        memory_id = await manager.store_memory(
            content=content,
            memory_type=memory_type,
            session_id=self.session_id,
            user_id=self.user_id,
            importance=importance,
            metadata=full_metadata,
            ttl=ttl,
        )
        
        self.logger.debug(f"Stored memory: {memory_id}", type=memory_type.value)
        return memory_id
    
    async def recall(
        self,
        query: str,
        limit: int = 5,
        mode: MemorySearchMode = MemorySearchMode.HYBRID,
        memory_types: Optional[List[MemoryType]] = None,
        min_importance: float = 0.0,
        tags: Optional[List[str]] = None,
    ) -> List[MemorySearchResult]:
        """
        Recall memories matching a query
        
        Args:
            query: Search query
            limit: Maximum results
            mode: Search mode (semantic, keyword, hybrid)
            memory_types: Filter by memory types
            min_importance: Minimum importance score
            tags: Filter by tags
        
        Returns:
            List of matching memories with scores
        """
        manager = await self._get_manager()
        
        memories = await manager.search_memories(
            query=query,
            session_id=self.session_id,
            user_id=self.user_id,
            limit=limit * 2,  # Get more for filtering
        )
        
        # Filter and score results
        results = []
        for memory in memories:
            # Apply filters
            if memory.importance < min_importance:
                continue
            
            if memory_types and memory.memory_type not in memory_types:
                continue
            
            if tags:
                memory_tags = memory.metadata.get("tags", [])
                if not any(t in memory_tags for t in tags):
                    continue
            
            # Calculate relevance score based on mode
            score = self._calculate_score(memory, query, mode)
            
            results.append(MemorySearchResult(
                memory_id=memory.memory_id,
                content=memory.content,
                score=score,
                memory_type=memory.memory_type,
                metadata=memory.metadata,
                created_at=memory.created_at,
            ))
        
        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    def _calculate_score(self, memory: Memory, query: str, mode: MemorySearchMode) -> float:
        """Calculate relevance score for a memory"""
        score = 0.5  # Base score
        
        if mode in [MemorySearchMode.KEYWORD, MemorySearchMode.HYBRID]:
            # Keyword matching
            query_terms = query.lower().split()
            content_lower = memory.content.lower()
            matches = sum(1 for term in query_terms if term in content_lower)
            score += 0.2 * (matches / max(len(query_terms), 1))
        
        if mode in [MemorySearchMode.IMPORTANCE, MemorySearchMode.HYBRID]:
            # Importance boost
            score += 0.15 * memory.importance
        
        if mode in [MemorySearchMode.RECENCY, MemorySearchMode.HYBRID]:
            # Recency boost
            age_hours = (datetime.utcnow() - memory.created_at).total_seconds() / 3600
            recency_factor = max(0, 1 - (age_hours / 168))  # Decay over 1 week
            score += 0.1 * recency_factor
        
        # Access count boost
        score += 0.05 * min(1, memory.access_count / 10)
        
        return min(1.0, score)
    
    async def forget(
        self,
        memory_id: Optional[str] = None,
        query: Optional[str] = None,
        older_than: Optional[datetime] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """
        Forget (delete) memories
        
        Args:
            memory_id: Specific memory to forget
            query: Delete memories matching query
            older_than: Delete memories older than this date
            memory_type: Only delete this type of memory
        
        Returns:
            Number of memories deleted
        """
        postgres = await get_postgres_manager()
        
        if memory_id:
            await postgres.execute(
                "DELETE FROM memories WHERE memory_id = $1",
                memory_id
            )
            return 1
        
        # Build delete query
        conditions = []
        params = []
        param_count = 0
        
        if self.session_id:
            param_count += 1
            conditions.append(f"session_id = ${param_count}")
            params.append(self.session_id)
        
        if older_than:
            param_count += 1
            conditions.append(f"created_at < ${param_count}")
            params.append(older_than)
        
        if memory_type:
            param_count += 1
            conditions.append(f"memory_type = ${param_count}")
            params.append(memory_type.value)
        
        if conditions:
            query_sql = f"DELETE FROM memories WHERE {' AND '.join(conditions)}"
            result = await postgres.execute(query_sql, *params)
            # Parse result to get count
            return int(result.split()[-1]) if result else 0
        
        return 0
    
    # ========================================================================
    # Working Memory (Fast Cache)
    # ========================================================================
    
    def set_working(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Set a value in working memory (fast cache)"""
        self._working_memory[key] = {
            "value": value,
            "expires": datetime.utcnow() + timedelta(seconds=ttl_seconds) if ttl_seconds else None
        }
    
    def get_working(self, key: str) -> Optional[Any]:
        """Get a value from working memory"""
        if key not in self._working_memory:
            return None
        
        entry = self._working_memory[key]
        if entry["expires"] and entry["expires"] < datetime.utcnow():
            del self._working_memory[key]
            return None
        
        return entry["value"]
    
    def clear_working(self):
        """Clear all working memory"""
        self._working_memory.clear()
    
    # ========================================================================
    # Episodic Memory (Events)
    # ========================================================================
    
    async def store_event(
        self,
        event_type: str,
        description: str,
        data: Dict[str, Any],
        importance: float = 0.5,
    ) -> str:
        """Store an event in episodic memory"""
        content = f"[{event_type}] {description}"
        
        return await self.store(
            content=content,
            memory_type=MemoryType.EPISODIC,
            importance=importance,
            metadata={
                "event_type": event_type,
                "event_data": data,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    
    async def get_event_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get event history"""
        postgres = await get_postgres_manager()
        
        query = """
            SELECT * FROM memories 
            WHERE memory_type = 'episodic'
        """
        params = []
        param_count = 0
        
        if self.session_id:
            param_count += 1
            query += f" AND session_id = ${param_count}"
            params.append(self.session_id)
        
        if event_type:
            param_count += 1
            query += f" AND metadata->>'event_type' = ${param_count}"
            params.append(event_type)
        
        query += f" ORDER BY created_at DESC LIMIT ${param_count + 1}"
        params.append(limit)
        
        rows = await postgres.fetch(query, *params)
        
        return [
            {
                "memory_id": row["memory_id"],
                "content": row["content"],
                "event_type": json.loads(row["metadata"]).get("event_type") if row["metadata"] else None,
                "event_data": json.loads(row["metadata"]).get("event_data") if row["metadata"] else {},
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    
    # ========================================================================
    # Semantic Memory (Facts)
    # ========================================================================
    
    async def store_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        confidence: float = 1.0,
        source: Optional[str] = None,
    ) -> str:
        """Store a fact in semantic memory (knowledge graph style)"""
        content = f"{subject} {predicate} {object_}"
        
        return await self.store(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            importance=confidence,
            metadata={
                "subject": subject,
                "predicate": predicate,
                "object": object_,
                "confidence": confidence,
                "source": source,
            }
        )
    
    async def query_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query facts from semantic memory"""
        postgres = await get_postgres_manager()
        
        conditions = ["memory_type = 'semantic'"]
        params = []
        param_count = 0
        
        if subject:
            param_count += 1
            conditions.append(f"metadata->>'subject' ILIKE ${param_count}")
            params.append(f"%{subject}%")
        
        if predicate:
            param_count += 1
            conditions.append(f"metadata->>'predicate' ILIKE ${param_count}")
            params.append(f"%{predicate}%")
        
        if object_:
            param_count += 1
            conditions.append(f"metadata->>'object' ILIKE ${param_count}")
            params.append(f"%{object_}%")
        
        query = f"SELECT * FROM memories WHERE {' AND '.join(conditions)} ORDER BY importance DESC LIMIT 50"
        rows = await postgres.fetch(query, *params)
        
        result = []
        for row in rows:
            memory_dict = {"memory_id": row["memory_id"]}
            if row["metadata"]:
                memory_dict.update(json.loads(row["metadata"]))
            result.append(memory_dict)
        return result
    
    # ========================================================================
    # Context Building
    # ========================================================================
    
    async def build_context(
        self,
        query: str,
        max_tokens: int = 2000,
        include_working: bool = True,
        include_recent: bool = True,
        include_relevant: bool = True,
    ) -> Dict[str, Any]:
        """
        Build comprehensive context for agent use
        
        Args:
            query: Current task/query for relevance
            max_tokens: Approximate max tokens for context
            include_working: Include working memory
            include_recent: Include recent memories
            include_relevant: Include query-relevant memories
        
        Returns:
            Context dictionary with categorized memories
        """
        manager = await self._get_manager()
        
        context = {
            "working_memory": {},
            "short_term": [],
            "long_term": [],
            "episodic": [],
            "semantic": [],
            "summary": "",
        }
        
        # Working memory
        if include_working:
            for key, entry in self._working_memory.items():
                if not entry["expires"] or entry["expires"] > datetime.utcnow():
                    context["working_memory"][key] = entry["value"]
        
        # Get memory context from manager
        memory_context = await manager.build_context(
            session_id=self.session_id,
            user_id=self.user_id,
            query=query if include_relevant else None,
        )
        
        context["short_term"] = memory_context.get("short_term", [])
        context["long_term"] = memory_context.get("long_term", [])
        context["episodic"] = memory_context.get("episodic", [])
        context["semantic"] = memory_context.get("semantic", [])
        
        # Build summary
        total_items = sum(len(v) if isinstance(v, list) else len(v.keys()) 
                        for v in context.values() if v)
        context["summary"] = f"Context includes {total_items} memory items across {len([k for k, v in context.items() if v and k != 'summary'])} categories"
        
        return context
    
    async def summarize_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        time_range: Optional[timedelta] = None,
    ) -> str:
        """Generate a summary of memories"""
        postgres = await get_postgres_manager()
        
        conditions = []
        params = []
        param_count = 0
        
        if self.session_id:
            param_count += 1
            conditions.append(f"session_id = ${param_count}")
            params.append(self.session_id)
        
        if memory_type:
            param_count += 1
            conditions.append(f"memory_type = ${param_count}")
            params.append(memory_type.value)
        
        if time_range:
            param_count += 1
            conditions.append(f"created_at > ${param_count}")
            params.append(datetime.utcnow() - time_range)
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        # Get stats
        stats_query = f"""
            SELECT 
                memory_type,
                COUNT(*) as count,
                AVG(importance) as avg_importance
            FROM memories
            {where_clause}
            GROUP BY memory_type
        """
        
        rows = await postgres.fetch(stats_query, *params)
        
        summary_parts = []
        for row in rows:
            summary_parts.append(
                f"- {row['memory_type']}: {row['count']} items (avg importance: {row['avg_importance']:.2f})"
            )
        
        return "Memory Summary:\n" + "\n".join(summary_parts) if summary_parts else "No memories found"


# ============================================================================
# Tool Registration Helper
# ============================================================================

async def register_memory_tools(agent, session_id: Optional[str] = None, user_id: Optional[str] = None):
    """Register memory tools with an agent"""
    memory_tools = EnhancedMemoryTools(session_id=session_id, user_id=user_id)
    
    # Store memory tool
    async def store_memory(
        content: str,
        memory_type: str = "long_term",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Store information in long-term memory"""
        mem_type = MemoryType(memory_type) if memory_type in [e.value for e in MemoryType] else MemoryType.LONG_TERM
        return await memory_tools.store(content, mem_type, importance, tags=tags)
    
    agent.register_tool("store_memory", store_memory, "Store information in memory for later recall")
    
    # Recall memory tool
    async def recall_memory(
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search and recall memories matching a query"""
        results = await memory_tools.recall(query, limit=limit)
        return [{"content": r.content, "score": r.score, "type": r.memory_type.value} for r in results]
    
    agent.register_tool("recall_memory", recall_memory, "Search and recall relevant memories")
    
    # Store event tool
    async def store_event(
        event_type: str,
        description: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store an event in episodic memory"""
        return await memory_tools.store_event(event_type, description, data or {})
    
    agent.register_tool("store_event", store_event, "Store an event for later reference")
    
    # Get context tool
    async def get_context(query: str = "") -> Dict[str, Any]:
        """Get relevant context for the current task"""
        return await memory_tools.build_context(query)
    
    agent.register_tool("get_context", get_context, "Get relevant context from memory")
    
    # Store fact tool
    async def store_fact(
        subject: str,
        predicate: str,
        object_: str,
        confidence: float = 1.0,
    ) -> str:
        """Store a fact (knowledge triple) in semantic memory"""
        return await memory_tools.store_fact(subject, predicate, object_, confidence)
    
    agent.register_tool("store_fact", store_fact, "Store a fact in semantic memory")
    
    # Query facts tool
    async def query_facts(
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query facts from semantic memory"""
        return await memory_tools.query_facts(subject=subject, predicate=predicate)
    
    agent.register_tool("query_facts", query_facts, "Query facts from knowledge base")
    
    return memory_tools

