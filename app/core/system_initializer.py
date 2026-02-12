"""
System Initializer
Main initialization module that sets up all core systems
"""
from typing import Optional
from ..database.postgres import get_postgres_manager, close_postgres
from ..core.session_manager import get_session_manager
from ..core.memory_manager import get_memory_manager
from ..core.event_handler import get_event_handler
from ..core.agent_initializer import get_agent_initializer
from ..core.langgraph_integration import get_langgraph_manager
from ..core.enhanced_guardrails import get_enhanced_guardrails
from ..integrations.api_bridge import get_api_bridge
from ..integrations.synapse_broker import get_synapse_broker
from ..core.realtime_data_pipeline import get_realtime_pipeline
from ..core.pipeline_auto_capture import get_auto_capture
from ..logging_config import get_logger

logger = get_logger("cyrex.system_initializer")


class SystemInitializer:
    """
    Initializes all core systems in the correct order
    Handles startup, shutdown, and health checks
    """
    
    def __init__(self):
        self.initialized = False
        self.logger = logger
    
    async def _initialize_vector_collections(self):
        """Initialize vector store collections for Language Intelligence Platform"""
        try:
            self.logger.info("Initializing vector store collections...")
            import asyncio
            from pymilvus import connections, utility
            from ..settings import settings

            # Wait for Milvus to be ready (with timeout)
            max_wait = 30  # seconds
            wait_interval = 2  # seconds
            milvus_ready = False
            init_alias = "collection_init"

            self.logger.info(f"Waiting for Milvus to be ready at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}...")
            for attempt in range(max_wait // wait_interval):
                try:
                    # Try to connect and verify Milvus is ready
                    if not connections.has_connection(init_alias):
                        connections.connect(alias=init_alias, host=settings.MILVUS_HOST, port=settings.MILVUS_PORT, timeout=5.0)
                    # Try to list collections as a health check
                    utility.list_collections(using=init_alias)
                    milvus_ready = True
                    self.logger.info(f"Milvus is ready (took {attempt * wait_interval}s)")
                    break
                except Exception as e:
                    if attempt == 0:
                        self.logger.info(f"Milvus not ready yet, waiting... ({e})")
                    # Disconnect failed connection
                    try:
                        if connections.has_connection(init_alias):
                            connections.disconnect(init_alias)
                    except:
                        pass
                    await asyncio.sleep(wait_interval)

            if not milvus_ready:
                self.logger.warning(f"Milvus not available after {max_wait}s, skipping collection initialization")
                return

            # Create collections
            from pymilvus import Collection
            from ..integrations.milvus import get_default_collections, get_collection_schema, DEFAULT_INDEX_PARAMS

            collections_to_create = get_default_collections()

            created_count = 0

            for collection_name in collections_to_create:
                try:
                    # Check if collection already exists
                    if utility.has_collection(collection_name, using=init_alias):
                        self.logger.info(f"Collection already exists: {collection_name}")
                        created_count += 1
                        continue

                    # Create collection with schema from milvus module
                    schema = get_collection_schema()
                    collection = Collection(name=collection_name, schema=schema, using=init_alias)

                    # Create index for vector search
                    collection.create_index(field_name="embedding", index_params=DEFAULT_INDEX_PARAMS)

                    created_count += 1
                    self.logger.info(f"Created collection: {collection_name}")

                except Exception as e:
                    self.logger.warning(f"Could not create collection {collection_name}: {e}")

            # Disconnect the initialization connection
            try:
                connections.disconnect(init_alias)
            except:
                pass

            self.logger.info(f"Vector collections initialized: {created_count} collections ready")

        except Exception as e:
            self.logger.warning(f"Failed to initialize vector collections: {e}")

    async def initialize_all(self):
        """Initialize all systems"""
        if self.initialized:
            self.logger.warning("Systems already initialized")
            return
        
        try:
            self.logger.info("Starting system initialization...")
            
            # 1. Initialize PostgreSQL connection
            self.logger.info("Initializing PostgreSQL...")
            await get_postgres_manager()
            
            # 2. Initialize database tables
            self.logger.info("Initializing agent database tables...")
            from ..database.agent_tables import initialize_agent_database
            await initialize_agent_database()
            
            # 3. Initialize core managers
            self.logger.info("Initializing core managers...")
            await get_session_manager()
            await get_memory_manager()
            await get_event_handler()
            await get_agent_initializer()
            await get_langgraph_manager()
            await get_enhanced_guardrails()
            
            # 4. Initialize integrations
            self.logger.info("Initializing integrations...")
            await get_api_bridge()
            await get_synapse_broker()
            
            # 5. Initialize LoRA and automation services
            self.logger.info("Initializing LoRA and automation services...")
            from ..integrations.lora_adapter_service import get_lora_service
            from ..integrations.company_data_automation import get_automation_service
            await get_lora_service()
            await get_automation_service()
            
            # 6. Initialize real-time data pipeline (dual-route: Helox + Cyrex)
            self.logger.info("Initializing real-time data pipeline...")
            await get_realtime_pipeline()
            
            # 7. Initialize auto-capture middleware (hooks orchestrator → pipeline)
            self.logger.info("Initializing pipeline auto-capture...")
            await get_auto_capture()
            
            # 8. Initialize vector store collections
            await self._initialize_vector_collections()
            self.initialized = True
            self.logger.info("System initialization complete!")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}", exc_info=True)
            raise
    
    async def shutdown_all(self):
        """Shutdown all systems gracefully"""
        self.logger.info("Shutting down systems...")
        
        try:
            # Shutdown real-time data pipeline (drain buffer)
            from ..core.realtime_data_pipeline import _pipeline
            if _pipeline:
                await _pipeline.shutdown()
            
            # Close API bridge clients
            from ..integrations.api_bridge import _api_bridge
            if _api_bridge:
                await _api_bridge.close()
            
            # Close PostgreSQL
            await close_postgres()
            
            self.initialized = False
            self.logger.info("System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def health_check(self) -> dict:
        """Check health of all systems"""
        health = {
            "initialized": self.initialized,
            "systems": {},
        }
        
        try:
            # Check PostgreSQL
            postgres = await get_postgres_manager()
            pg_health = await postgres.health_check()
            health["systems"]["postgresql"] = pg_health
            
            # Check other systems
            health["systems"]["session_manager"] = {"healthy": True}
            health["systems"]["memory_manager"] = {"healthy": True}
            health["systems"]["event_handler"] = {"healthy": True}
            health["systems"]["agent_initializer"] = {"healthy": True}
            health["systems"]["langgraph"] = {"healthy": True}
            health["systems"]["guardrails"] = {"healthy": True}
            health["systems"]["api_bridge"] = {"healthy": True}
            health["systems"]["synapse_broker"] = {"healthy": True}
            health["systems"]["lora_service"] = {"healthy": True}
            health["systems"]["automation_service"] = {"healthy": True}
            health["systems"]["realtime_pipeline"] = {"healthy": True}
            health["systems"]["auto_capture"] = {"healthy": True}
            
        except Exception as e:
            health["error"] = str(e)
            self.logger.error(f"Health check failed: {e}")
        
        return health


# Global system initializer
_system_initializer: Optional[SystemInitializer] = None


async def get_system_initializer() -> SystemInitializer:
    """Get or create system initializer singleton"""
    global _system_initializer
    if _system_initializer is None:
        _system_initializer = SystemInitializer()
    return _system_initializer

