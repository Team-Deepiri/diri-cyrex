"""
Memory Management System
Comprehensive memory/context tracking with short-term and long-term storage
Integrates with vector store for semantic search
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
from ..database.postgres import get_postgres_manager
from ..core.types import Memory, MemoryType
from ..integrations.milvus_store import get_milvus_store
from ..logging_config import get_logger
import json

logger = get_logger("cyrex.memory_manager")


class MemoryManager:
    """
    Manages short-term and long-term memory with semantic search
    Handles memory storage, retrieval, and context building
    """
    
    def __init__(self):
        self._short_term_memory: Dict[str, Memory] = {}
        self._lock = asyncio.Lock()
        self.logger = logger
        self.vector_store = None
    
    async def initialize(self):
        """Initialize memory manager and vector store"""
        # Create memories table in PostgreSQL
        postgres = await get_postgres_manager()
        await postgres.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id VARCHAR(255) PRIMARY KEY,
                session_id VARCHAR(255),
                user_id VARCHAR(255),
                memory_type VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB,
                importance FLOAT DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP NOT NULL,
                created_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id);
            CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_expires_at ON memories(expires_at);
        """)
        
        # Initialize vector store for semantic search
        try:
            self.vector_store = await get_milvus_store(collection_name="cyrex_memories")
        except Exception as e:
            self.logger.warning(f"Vector store not available for memory search: {e}")
        
        self.logger.info("Memory manager initialized")
    
    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Store a memory"""
        memory = Memory(
            session_id=session_id,
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            importance=importance,
            metadata=metadata or {},
            expires_at=datetime.utcnow() + timedelta(seconds=ttl) if ttl else None,
        )
        
        async with self._lock:
            # Store in PostgreSQL
            postgres = await get_postgres_manager()
            await postgres.execute("""
                INSERT INTO memories (memory_id, session_id, user_id, memory_type, content,
                                    metadata, importance, access_count, last_accessed, created_at, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """, memory.memory_id, memory.session_id, memory.user_id, memory.memory_type.value,
                memory.content, json.dumps(memory.metadata), memory.importance,
                memory.access_count, memory.last_accessed, memory.created_at, memory.expires_at)
            
            # Store in vector store for semantic search (if available)
            if self.vector_store and memory_type in [MemoryType.LONG_TERM, MemoryType.EPISODIC, MemoryType.SEMANTIC]:
                try:
                    await self.vector_store.add_documents(
                        documents=[memory.content],
                        metadatas=[{
                            "memory_id": memory.memory_id,
                            "session_id": session_id or "",
                            "user_id": user_id or "",
                            "memory_type": memory_type.value,
                            "importance": importance,
                        }],
                        ids=[memory.memory_id],
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to store memory in vector store: {e}")
            
            # Cache short-term memories in memory
            if memory_type == MemoryType.SHORT_TERM:
                self._short_term_memory[memory.memory_id] = memory
            
            self.logger.debug(f"Memory stored: {memory.memory_id}", type=memory_type.value)
            return memory.memory_id
    
    async def retrieve_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID"""
        # Check short-term cache first
        if memory_id in self._short_term_memory:
            memory = self._short_term_memory[memory_id]
            # Update access count
            memory.access_count += 1
            memory.last_accessed = datetime.utcnow()
            return memory
        
        # Load from PostgreSQL
        postgres = await get_postgres_manager()
        row = await postgres.fetchrow(
            "SELECT * FROM memories WHERE memory_id = $1", memory_id
        )
        
        if row:
            memory = Memory(
                memory_id=row['memory_id'],
                session_id=row['session_id'],
                user_id=row['user_id'],
                memory_type=MemoryType(row['memory_type']),
                content=row['content'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                importance=row['importance'],
                access_count=row['access_count'],
                last_accessed=row['last_accessed'],
                created_at=row['created_at'],
                expires_at=row['expires_at'],
            )
            
            # Update access count
            await postgres.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = $1 WHERE memory_id = $2",
                datetime.utcnow(), memory_id
            )
            
            return memory
        
        return None
    
    async def search_memories(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[Memory]:
        """Search memories by semantic similarity"""
        memories = []
        
        # Use vector store for semantic search if available
        if self.vector_store:
            try:
                # Build metadata filter
                metadata_filter = {}
                if session_id:
                    metadata_filter["session_id"] = session_id
                if user_id:
                    metadata_filter["user_id"] = user_id
                if memory_type:
                    metadata_filter["memory_type"] = memory_type.value
                
                # Search vector store
                results = await self.vector_store.similarity_search_with_score(
                    query=query,
                    k=limit,
                    filter=metadata_filter if metadata_filter else None,
                )
                
                # Retrieve full memory objects
                for doc, score in results:
                    memory_id = doc.metadata.get("memory_id")
                    if memory_id:
                        memory = await self.retrieve_memory(memory_id)
                        if memory:
                            memories.append(memory)
            except Exception as e:
                self.logger.warning(f"Vector search failed, falling back to text search: {e}")
        
        # Fallback to text search in PostgreSQL
        if not memories:
            postgres = await get_postgres_manager()
            query_sql = """
                SELECT * FROM memories
                WHERE content ILIKE $1
            """
            params = [f"%{query}%"]
            param_count = 1
            
            if session_id:
                param_count += 1
                query_sql += f" AND session_id = ${param_count}"
                params.append(session_id)
            
            if user_id:
                param_count += 1
                query_sql += f" AND user_id = ${param_count}"
                params.append(user_id)
            
            if memory_type:
                param_count += 1
                query_sql += f" AND memory_type = ${param_count}"
                params.append(memory_type.value)
            
            query_sql += f" ORDER BY importance DESC, last_accessed DESC LIMIT ${param_count + 1}"
            params.append(limit)
            
            rows = await postgres.fetch(query_sql, *params)
            
            for row in rows:
                memory = Memory(
                    memory_id=row['memory_id'],
                    session_id=row['session_id'],
                    user_id=row['user_id'],
                    memory_type=MemoryType(row['memory_type']),
                    content=row['content'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    importance=row['importance'],
                    access_count=row['access_count'],
                    last_accessed=row['last_accessed'],
                    created_at=row['created_at'],
                    expires_at=row['expires_at'],
                )
                memories.append(memory)
        
        return memories
    
    async def build_context(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Build context from relevant memories"""
        context = {
            "short_term": [],
            "long_term": [],
            "episodic": [],
            "semantic": [],
        }
        
        # Get short-term memories from session
        if session_id:
            postgres = await get_postgres_manager()
            rows = await postgres.fetch("""
                SELECT * FROM memories
                WHERE session_id = $1 AND memory_type = 'short_term'
                ORDER BY created_at DESC
                LIMIT $2
            """, session_id, limit // 2)
            
            for row in rows:
                memory = Memory(
                    memory_id=row['memory_id'],
                    session_id=row['session_id'],
                    user_id=row['user_id'],
                    memory_type=MemoryType(row['memory_type']),
                    content=row['content'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    importance=row['importance'],
                    access_count=row['access_count'],
                    last_accessed=row['last_accessed'],
                    created_at=row['created_at'],
                    expires_at=row['expires_at'],
                )
                context["short_term"].append(memory.content)
        
        # Search for relevant long-term memories if query provided
        if query:
            relevant = await self.search_memories(
                query=query,
                session_id=session_id,
                user_id=user_id,
                limit=limit // 2,
            )
            
            for memory in relevant:
                context[memory.memory_type.value].append(memory.content)
        
        return context


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


async def get_memory_manager() -> MemoryManager:
    """Get or create memory manager singleton"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
        await _memory_manager.initialize()
    return _memory_manager

