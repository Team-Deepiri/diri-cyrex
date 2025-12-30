"""
PostgreSQL Database Connection
Async PostgreSQL connection management with connection pooling
Integrates with the postgres container from docker-compose
"""
from typing import Optional, AsyncGenerator, Dict, Any, List
import asyncpg
from contextlib import asynccontextmanager
from ..settings import settings
from ..logging_config import get_logger
import os

logger = get_logger("cyrex.database.postgres")

# Global connection pool
_pool: Optional[asyncpg.Pool] = None


class PostgreSQLManager:
    """
    PostgreSQL connection manager with async support
    Handles connection pooling, migrations, and query execution
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 5432,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        min_size: int = 5,
        max_size: int = 20,
    ):
        self.host = host or os.getenv("POSTGRES_HOST", "postgres")
        self.port = port
        self.database = database or os.getenv("POSTGRES_DB", "deepiri")
        self.user = user or os.getenv("POSTGRES_USER", "deepiri")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "deepiripassword")
        self.min_size = min_size
        self.max_size = max_size
        self._pool: Optional[asyncpg.Pool] = None
        self.logger = logger
    
    async def initialize(self) -> bool:
        """Initialize connection pool"""
        try:
            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=60,
            )
            self.logger.info(
                f"PostgreSQL connection pool initialized: {self.host}:{self.port}/{self.database}",
                pool_size=f"{self.min_size}-{self.max_size}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            return False
    
    async def close(self):
        """Close connection pool"""
        if self._pool:
            await self._pool.close()
            self.logger.info("PostgreSQL connection pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool"""
        if not self._pool:
            raise RuntimeError("PostgreSQL pool not initialized. Call initialize() first.")
        async with self._pool.acquire() as connection:
            yield connection
    
    async def execute(self, query: str, *args) -> str:
        """Execute a query and return the result"""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """Fetch multiple rows"""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Fetch a single row"""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args) -> Any:
        """Fetch a single value"""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            version = await self.fetchval("SELECT version()")
            return {
                "healthy": True,
                "version": version.split(",")[0] if version else "unknown",
                "pool_size": self._pool.get_size() if self._pool else 0,
                "idle_size": self._pool.get_idle_size() if self._pool else 0,
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }


# Global PostgreSQL manager instance
_postgres_manager: Optional[PostgreSQLManager] = None


async def get_postgres_manager() -> PostgreSQLManager:
    """Get or create PostgreSQL manager singleton"""
    global _postgres_manager
    if _postgres_manager is None:
        _postgres_manager = PostgreSQLManager()
        await _postgres_manager.initialize()
    return _postgres_manager


async def close_postgres():
    """Close PostgreSQL connection pool"""
    global _postgres_manager
    if _postgres_manager:
        await _postgres_manager.close()
        _postgres_manager = None

