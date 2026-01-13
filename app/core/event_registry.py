"""
Event Registry
Centralized registry for event types, schemas, and handlers
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import json
import uuid
from ..logging_config import get_logger

logger = get_logger("cyrex.event_registry")


class EventPriority(str, Enum):
    """Event priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class EventCategory(str, Enum):
    """Event categories"""
    AGENT = "agent"
    TOOL = "tool"
    MEMORY = "memory"
    SESSION = "session"
    SYSTEM = "system"
    USER = "user"
    WORKFLOW = "workflow"
    ERROR = "error"
    AUDIT = "audit"


@dataclass
class EventSchema:
    """Schema definition for an event type"""
    event_type: str
    category: EventCategory
    description: str
    payload_schema: Dict[str, Any]  # JSON schema
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    priority: EventPriority = EventPriority.NORMAL
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegisteredHandler:
    """Registered event handler"""
    handler_id: str
    event_type: str
    handler: Callable
    priority: int = 0  # Higher priority handlers run first
    filter_func: Optional[Callable] = None  # Optional filter function
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventRegistry:
    """
    Centralized registry for event types and handlers
    Provides event schema validation and handler management
    """
    
    def __init__(self):
        self._schemas: Dict[str, EventSchema] = {}
        self._handlers: Dict[str, List[RegisteredHandler]] = {}  # event_type -> handlers
        self._wildcard_handlers: List[RegisteredHandler] = []  # Handlers for all events
        self.logger = logger
        self._register_default_events()
    
    def _register_default_events(self):
        """Register default event types"""
        default_events = [
            EventSchema(
                event_type="agent.created",
                category=EventCategory.AGENT,
                description="Agent instance created",
                payload_schema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "role": {"type": "string"},
                        "name": {"type": "string"},
                    },
                    "required": ["agent_id", "role"]
                },
                required_fields=["agent_id", "role"],
            ),
            EventSchema(
                event_type="agent.invoked",
                category=EventCategory.AGENT,
                description="Agent invoke method called",
                payload_schema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "input_text": {"type": "string"},
                        "session_id": {"type": "string"},
                    },
                    "required": ["agent_id", "input_text"]
                },
                required_fields=["agent_id", "input_text"],
            ),
            EventSchema(
                event_type="agent.response",
                category=EventCategory.AGENT,
                description="Agent response generated",
                payload_schema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "response_content": {"type": "string"},
                        "confidence": {"type": "number"},
                        "tool_calls": {"type": "array"},
                    },
                    "required": ["agent_id", "response_content"]
                },
                required_fields=["agent_id", "response_content"],
            ),
            EventSchema(
                event_type="tool.called",
                category=EventCategory.TOOL,
                description="Tool execution started",
                payload_schema={
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string"},
                        "parameters": {"type": "object"},
                        "agent_id": {"type": "string"},
                    },
                    "required": ["tool_name"]
                },
                required_fields=["tool_name"],
            ),
            EventSchema(
                event_type="tool.completed",
                category=EventCategory.TOOL,
                description="Tool execution completed",
                payload_schema={
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string"},
                        "result": {"type": "object"},
                        "execution_time_ms": {"type": "number"},
                    },
                    "required": ["tool_name"]
                },
                required_fields=["tool_name"],
            ),
            EventSchema(
                event_type="memory.stored",
                category=EventCategory.MEMORY,
                description="Memory stored",
                payload_schema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"},
                        "memory_type": {"type": "string"},
                        "session_id": {"type": "string"},
                    },
                    "required": ["memory_id", "memory_type"]
                },
                required_fields=["memory_id", "memory_type"],
            ),
            EventSchema(
                event_type="session.created",
                category=EventCategory.SESSION,
                description="Session created",
                payload_schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "agent_id": {"type": "string"},
                    },
                    "required": ["session_id"]
                },
                required_fields=["session_id"],
            ),
            EventSchema(
                event_type="workflow.started",
                category=EventCategory.WORKFLOW,
                description="Workflow execution started",
                payload_schema={
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string"},
                        "workflow_type": {"type": "string"},
                    },
                    "required": ["workflow_id"]
                },
                required_fields=["workflow_id"],
            ),
            EventSchema(
                event_type="error.occurred",
                category=EventCategory.ERROR,
                description="Error occurred",
                payload_schema={
                    "type": "object",
                    "properties": {
                        "error_type": {"type": "string"},
                        "error_message": {"type": "string"},
                        "component": {"type": "string"},
                        "stack_trace": {"type": "string"},
                    },
                    "required": ["error_type", "error_message"]
                },
                required_fields=["error_type", "error_message"],
                priority=EventPriority.HIGH,
            ),
        ]
        
        for event_schema in default_events:
            self.register_event_schema(event_schema)
    
    def register_event_schema(self, schema: EventSchema):
        """Register an event schema"""
        self._schemas[schema.event_type] = schema
        self.logger.info(f"Registered event schema: {schema.event_type}")
    
    def get_event_schema(self, event_type: str) -> Optional[EventSchema]:
        """Get event schema by type"""
        return self._schemas.get(event_type)
    
    def register_handler(
        self,
        event_type: str,
        handler: Callable,
        priority: int = 0,
        filter_func: Optional[Callable] = None,
        handler_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register an event handler
        
        Args:
            event_type: Event type to handle (use "*" for all events)
            handler: Handler function (async or sync)
            priority: Handler priority (higher = runs first)
            filter_func: Optional filter function to determine if handler should run
            handler_id: Optional handler ID (auto-generated if not provided)
            metadata: Optional metadata
        
        Returns:
            Handler ID
        """
        handler_id = handler_id or str(uuid.uuid4())
        
        registered_handler = RegisteredHandler(
            handler_id=handler_id,
            event_type=event_type,
            handler=handler,
            priority=priority,
            filter_func=filter_func,
            metadata=metadata or {},
        )
        
        if event_type == "*":
            self._wildcard_handlers.append(registered_handler)
            self._wildcard_handlers.sort(key=lambda h: h.priority, reverse=True)
        else:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(registered_handler)
            self._handlers[event_type].sort(key=lambda h: h.priority, reverse=True)
        
        self.logger.info(f"Registered handler: {handler_id} for event type: {event_type}")
        return handler_id
    
    def unregister_handler(self, handler_id: str) -> bool:
        """Unregister a handler by ID"""
        # Remove from specific event handlers
        for event_type, handlers in self._handlers.items():
            self._handlers[event_type] = [h for h in handlers if h.handler_id != handler_id]
        
        # Remove from wildcard handlers
        self._wildcard_handlers = [h for h in self._wildcard_handlers if h.handler_id != handler_id]
        
        self.logger.info(f"Unregistered handler: {handler_id}")
        return True
    
    def get_handlers(self, event_type: str) -> List[RegisteredHandler]:
        """Get all handlers for an event type (including wildcard)"""
        handlers = self._handlers.get(event_type, []).copy()
        handlers.extend(self._wildcard_handlers)
        handlers.sort(key=lambda h: h.priority, reverse=True)
        return handlers
    
    def validate_event_payload(self, event_type: str, payload: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate event payload against schema
        
        Returns:
            (is_valid, list_of_errors)
        """
        schema = self.get_event_schema(event_type)
        if not schema:
            return True, []  # No schema = no validation
        
        errors = []
        
        # Check required fields
        for field in schema.required_fields:
            if field not in payload:
                errors.append(f"Required field '{field}' is missing")
        
        # TODO: Add JSON schema validation if needed
        
        return len(errors) == 0, errors
    
    def list_event_types(self, category: Optional[EventCategory] = None) -> List[str]:
        """List all registered event types"""
        event_types = list(self._schemas.keys())
        if category:
            event_types = [
                et for et in event_types
                if self._schemas[et].category == category
            ]
        return sorted(event_types)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_event_types": len(self._schemas),
            "total_handlers": sum(len(handlers) for handlers in self._handlers.values()) + len(self._wildcard_handlers),
            "handlers_by_event": {
                event_type: len(handlers)
                for event_type, handlers in self._handlers.items()
            },
            "wildcard_handlers": len(self._wildcard_handlers),
            "events_by_category": {
                cat.value: len([s for s in self._schemas.values() if s.category == cat])
                for cat in EventCategory
            },
        }


# Global event registry instance
_event_registry: Optional[EventRegistry] = None


def get_event_registry() -> EventRegistry:
    """Get or create event registry singleton"""
    global _event_registry
    if _event_registry is None:
        _event_registry = EventRegistry()
    return _event_registry

