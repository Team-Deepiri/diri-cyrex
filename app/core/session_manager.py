"""
Session Management System
Comprehensive session lifecycle management with PostgreSQL persistence
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
from ..database.postgres import get_postgres_manager
from ..core.types import Session, AgentStatus
from ..logging_config import get_logger
import json

logger = get_logger("cyrex.session_manager")


class SessionManager:
    """
    Manages user and agent sessions with full lifecycle support
    Handles session creation, updates, expiration, and cleanup
    """
    
    def __init__(self, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self.logger = logger
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize session manager and start cleanup task"""
        # Create sessions table if it doesn't exist
        postgres = await get_postgres_manager()
        await postgres.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255),
                agent_id VARCHAR(255),
                status VARCHAR(50) NOT NULL,
                context JSONB,
                metadata JSONB,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP,
                last_activity TIMESTAMP NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_agent_id ON sessions(agent_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
        """)
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
        self.logger.info("Session manager initialized")
    
    async def create_session(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> Session:
        """Create a new session"""
        async with self._lock:
            session = Session(
                user_id=user_id,
                agent_id=agent_id,
                context=context or {},
                metadata=metadata or {},
                expires_at=datetime.utcnow() + timedelta(seconds=ttl or self.default_ttl),
            )
            
            # Store in memory
            self._sessions[session.session_id] = session
            
            # Persist to PostgreSQL
            postgres = await get_postgres_manager()
            await postgres.execute("""
                INSERT INTO sessions (session_id, user_id, agent_id, status, context, metadata, 
                                    created_at, updated_at, expires_at, last_activity)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (session_id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    agent_id = EXCLUDED.agent_id,
                    status = EXCLUDED.status,
                    context = EXCLUDED.context,
                    metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at,
                    expires_at = EXCLUDED.expires_at,
                    last_activity = EXCLUDED.last_activity
            """, session.session_id, session.user_id, session.agent_id, session.status,
                json.dumps(session.context), json.dumps(session.metadata),
                session.created_at, session.updated_at, session.expires_at, session.last_activity)
            
            self.logger.info(f"Session created: {session.session_id}", user_id=user_id, agent_id=agent_id)
            return session
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID"""
        # Check memory first
        if session_id in self._sessions:
            session = self._sessions[session_id]
            # Check if expired
            if session.expires_at and session.expires_at < datetime.utcnow():
                await self.delete_session(session_id)
                return None
            return session
        
        # Load from PostgreSQL
        postgres = await get_postgres_manager()
        row = await postgres.fetchrow(
            "SELECT * FROM sessions WHERE session_id = $1", session_id
        )
        
        if row:
            session = Session(
                session_id=row['session_id'],
                user_id=row['user_id'],
                agent_id=row['agent_id'],
                status=row['status'],
                context=json.loads(row['context']) if row['context'] else {},
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                expires_at=row['expires_at'],
                last_activity=row['last_activity'],
            )
            # Cache in memory
            self._sessions[session_id] = session
            return session
        
        return None
    
    async def update_session(
        self,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
    ) -> bool:
        """Update a session"""
        session = await self.get_session(session_id)
        if not session:
            return False
        
        async with self._lock:
            if context is not None:
                session.context.update(context)
            if metadata is not None:
                session.metadata.update(metadata)
            if status is not None:
                session.status = status
            
            session.updated_at = datetime.utcnow()
            session.last_activity = datetime.utcnow()
            
            # Update PostgreSQL
            postgres = await get_postgres_manager()
            await postgres.execute("""
                UPDATE sessions SET
                    context = $1,
                    metadata = $2,
                    status = $3,
                    updated_at = $4,
                    last_activity = $5
                WHERE session_id = $6
            """, json.dumps(session.context), json.dumps(session.metadata),
                session.status, session.updated_at, session.last_activity, session_id)
            
            return True
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        async with self._lock:
            # Remove from memory
            self._sessions.pop(session_id, None)
            
            # Remove from PostgreSQL
            postgres = await get_postgres_manager()
            await postgres.execute("DELETE FROM sessions WHERE session_id = $1", session_id)
            
            self.logger.info(f"Session deleted: {session_id}")
            return True
    
    async def list_sessions(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Session]:
        """List sessions with filters"""
        postgres = await get_postgres_manager()
        
        query = "SELECT * FROM sessions WHERE 1=1"
        params = []
        param_count = 0
        
        if user_id:
            param_count += 1
            query += f" AND user_id = ${param_count}"
            params.append(user_id)
        
        if agent_id:
            param_count += 1
            query += f" AND agent_id = ${param_count}"
            params.append(agent_id)
        
        if status:
            param_count += 1
            query += f" AND status = ${param_count}"
            params.append(status)
        
        query += f" ORDER BY last_activity DESC LIMIT ${param_count + 1}"
        params.append(limit)
        
        rows = await postgres.fetch(query, *params)
        
        sessions = []
        for row in rows:
            session = Session(
                session_id=row['session_id'],
                user_id=row['user_id'],
                agent_id=row['agent_id'],
                status=row['status'],
                context=json.loads(row['context']) if row['context'] else {},
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                expires_at=row['expires_at'],
                last_activity=row['last_activity'],
            )
            sessions.append(session)
        
        return sessions
    
    async def _cleanup_expired_sessions(self):
        """Background task to clean up expired sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                now = datetime.utcnow()
                expired_ids = []
                
                async with self._lock:
                    for session_id, session in list(self._sessions.items()):
                        if session.expires_at and session.expires_at < now:
                            expired_ids.append(session_id)
                    
                    for session_id in expired_ids:
                        await self.delete_session(session_id)
                
                # Also clean up in PostgreSQL
                postgres = await get_postgres_manager()
                await postgres.execute(
                    "DELETE FROM sessions WHERE expires_at < $1", now
                )
                
                if expired_ids:
                    self.logger.info(f"Cleaned up {len(expired_ids)} expired sessions")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in session cleanup: {e}")


# Global session manager instance
_session_manager: Optional[SessionManager] = None


async def get_session_manager() -> SessionManager:
    """Get or create session manager singleton"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
        await _session_manager.initialize()
    return _session_manager

