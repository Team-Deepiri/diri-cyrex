"""
Event Handler System
Comprehensive event handling with event routing, filtering, and processing
"""
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime
import asyncio
from collections import defaultdict
from ..core.types import Event
from ..core.event_registry import get_event_registry
from ..database.postgres import get_postgres_manager
from ..logging_config import get_logger
import json

logger = get_logger("cyrex.event_handler")


class EventHandler:
    """
    Manages event handling with routing, filtering, and async processing
    Supports event subscriptions, middleware, and event persistence
    """
    
    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._middleware: List[Callable] = []
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processing = False
        self._lock = asyncio.Lock()
        self.logger = logger
        self._processing_task: Optional[asyncio.Task] = None
        self._event_registry = None
    
    async def initialize(self):
        """Initialize event handler and create database tables"""
        # Get event registry
        self._event_registry = get_event_registry()
        
        # Create events table
        postgres = await get_postgres_manager()
        await postgres.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id VARCHAR(255) PRIMARY KEY,
                event_type VARCHAR(255) NOT NULL,
                source VARCHAR(255) NOT NULL,
                target VARCHAR(255),
                payload JSONB NOT NULL,
                metadata JSONB,
                timestamp TIMESTAMP NOT NULL,
                processed BOOLEAN DEFAULT FALSE
            );
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);
            CREATE INDEX IF NOT EXISTS idx_events_target ON events(target);
            CREATE INDEX IF NOT EXISTS idx_events_processed ON events(processed);
            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
        """)
        
        # Start event processing task
        self._processing_task = asyncio.create_task(self._process_events())
        self.logger.info("Event handler initialized")
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to an event type"""
        async with self._lock:
            self._handlers[event_type].append(handler)
            self.logger.debug(f"Handler subscribed to event type: {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from an event type"""
        async with self._lock:
            if event_type in self._handlers:
                self._handlers[event_type].remove(handler)
                self.logger.debug(f"Handler unsubscribed from event type: {event_type}")
    
    def add_middleware(self, middleware: Callable):
        """Add middleware to event processing pipeline"""
        self._middleware.append(middleware)
        self.logger.debug("Middleware added")
    
    async def emit(
        self,
        event_type: str,
        payload: Dict[str, Any],
        source: str = "system",
        target: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Emit an event"""
        # Validate payload if schema exists
        if self._event_registry:
            is_valid, errors = self._event_registry.validate_event_payload(event_type, payload)
            if not is_valid:
                self.logger.warning(f"Event payload validation failed: {errors}", event_type=event_type)
        
        event = Event(
            event_type=event_type,
            source=source,
            target=target,
            payload=payload,
            metadata=metadata or {},
        )
        
        # Store in database
        postgres = await get_postgres_manager()
        await postgres.execute("""
            INSERT INTO events (event_id, event_type, source, target, payload, metadata, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, event.event_id, event.event_type, event.source, event.target,
            json.dumps(event.payload), json.dumps(event.metadata), event.timestamp)
        
        # Add to processing queue
        await self._event_queue.put(event)
        
        self.logger.debug(f"Event emitted: {event.event_id}", type=event_type, source=source)
        return event.event_id
    
    async def _process_events(self):
        """Background task to process events"""
        while True:
            try:
                # Get event from queue with timeout
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process middleware
                for middleware in self._middleware:
                    try:
                        if asyncio.iscoroutinefunction(middleware):
                            event = await middleware(event)
                        else:
                            event = middleware(event)
                    except Exception as e:
                        self.logger.warning(f"Middleware error: {e}")
                
                # Route to handlers from registry
                if self._event_registry:
                    registered_handlers = self._event_registry.get_handlers(event.event_type)
                    for registered_handler in registered_handlers:
                        # Check filter if present
                        if registered_handler.filter_func:
                            try:
                                if asyncio.iscoroutinefunction(registered_handler.filter_func):
                                    should_run = await registered_handler.filter_func(event)
                                else:
                                    should_run = registered_handler.filter_func(event)
                                if not should_run:
                                    continue
                            except Exception as e:
                                self.logger.warning(f"Filter function error: {e}")
                                continue
                        
                        try:
                            if asyncio.iscoroutinefunction(registered_handler.handler):
                                await registered_handler.handler(event)
                            else:
                                registered_handler.handler(event)
                        except Exception as e:
                            self.logger.error(f"Registered handler error for {event.event_type}: {e}")
                
                # Also route to legacy handlers
                handlers = self._handlers.get(event.event_type, [])
                handlers.extend(self._handlers.get("*", []))  # Wildcard handlers
                
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        self.logger.error(f"Handler error for {event.event_type}: {e}")
                
                # Mark as processed
                postgres = await get_postgres_manager()
                await postgres.execute(
                    "UPDATE events SET processed = TRUE WHERE event_id = $1",
                    event.event_id
                )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing events: {e}")
    
    async def get_events(
        self,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        target: Optional[str] = None,
        processed: Optional[bool] = None,
        limit: int = 100,
    ) -> List[Event]:
        """Query events from database"""
        postgres = await get_postgres_manager()
        
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        param_count = 0
        
        if event_type:
            param_count += 1
            query += f" AND event_type = ${param_count}"
            params.append(event_type)
        
        if source:
            param_count += 1
            query += f" AND source = ${param_count}"
            params.append(source)
        
        if target:
            param_count += 1
            query += f" AND target = ${param_count}"
            params.append(target)
        
        if processed is not None:
            param_count += 1
            query += f" AND processed = ${param_count}"
            params.append(processed)
        
        query += f" ORDER BY timestamp DESC LIMIT ${param_count + 1}"
        params.append(limit)
        
        rows = await postgres.fetch(query, *params)
        
        events = []
        for row in rows:
            event = Event(
                event_id=row['event_id'],
                event_type=row['event_type'],
                source=row['source'],
                target=row['target'],
                payload=json.loads(row['payload']),
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                timestamp=row['timestamp'],
                processed=row['processed'],
            )
            events.append(event)
        
        return events


# Global event handler instance
_event_handler: Optional[EventHandler] = None


async def get_event_handler() -> EventHandler:
    """Get or create event handler singleton"""
    global _event_handler
    if _event_handler is None:
        _event_handler = EventHandler()
        await _event_handler.initialize()
    return _event_handler

