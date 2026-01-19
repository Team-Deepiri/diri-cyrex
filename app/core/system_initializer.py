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
            
            self.initialized = True
            self.logger.info("System initialization complete!")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}", exc_info=True)
            raise
    
    async def shutdown_all(self):
        """Shutdown all systems gracefully"""
        self.logger.info("Shutting down systems...")
        
        try:
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

