"""
Core Data Types and Globals
Comprehensive type definitions for the Cyrex AI system
"""
from typing import Dict, List, Optional, Any, Union, Literal, TypedDict, Protocol
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid


# ============================================================================
# Agent Types and States
# ============================================================================

class AgentRole(str, Enum):
    """Agent role types"""
    ORCHESTRATOR = "orchestrator"
    TASK_DECOMPOSER = "task_decomposer"
    TIME_OPTIMIZER = "time_optimizer"
    CREATIVE_SPARKER = "creative_sparker"
    QUALITY_ASSURANCE = "quality_assurance"
    ENGAGEMENT_SPECIALIST = "engagement_specialist"
    MEMORY_MANAGER = "memory_manager"
    TOOL_EXECUTOR = "tool_executor"
    GUARDRAIL_ENFORCER = "guardrail_enforcer"
    # Cyrex Vendor Fraud Detection Agents
    INVOICE_ANALYZER = "invoice_analyzer"
    VENDOR_INTELLIGENCE = "vendor_intelligence"
    PRICING_BENCHMARK = "pricing_benchmark"
    FRAUD_DETECTOR = "fraud_detector"
    DOCUMENT_PROCESSOR = "document_processor"
    RISK_ASSESSOR = "risk_assessor"


class IndustryNiche(str, Enum):
    """Industry niches for vendor fraud detection"""
    PROPERTY_MANAGEMENT = "property_management"
    CORPORATE_PROCUREMENT = "corporate_procurement"
    INSURANCE_PC = "insurance_pc"
    GENERAL_CONTRACTORS = "general_contractors"
    RETAIL_ECOMMERCE = "retail_ecommerce"
    LAW_FIRMS = "law_firms"
    GENERIC = "generic"


class VendorFraudType(str, Enum):
    """Types of vendor fraud"""
    INFLATED_INVOICE = "inflated_invoice"
    PHANTOM_WORK = "phantom_work"
    DUPLICATE_BILLING = "duplicate_billing"
    UNNECESSARY_SERVICES = "unnecessary_services"
    KICKBACK_SCHEME = "kickback_scheme"
    PRICE_GOUGING = "price_gouging"
    CONTRACT_NON_COMPLIANCE = "contract_non_compliance"
    FORGED_DOCUMENTS = "forged_documents"


class RiskLevel(str, Enum):
    """Risk level assessment"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentStatus(str, Enum):
    """Agent execution status"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class MessagePriority(str, Enum):
    """Message priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryType(str, Enum):
    """Memory storage types"""
    SHORT_TERM = "short_term"  # Session-based, in-memory
    LONG_TERM = "long_term"  # Persistent, database
    EPISODIC = "episodic"  # Event-based memories
    SEMANTIC = "semantic"  # Factual knowledge
    WORKING = "working"  # Current context window


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class AgentConfig:
    """Agent configuration and initialization data"""
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: AgentRole = AgentRole.ORCHESTRATOR
    name: str = ""
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    model_config: Dict[str, Any] = field(default_factory=dict)
    temperature: float = 0.7
    max_tokens: int = 2000
    system_prompt: Optional[str] = None
    guardrails: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Session:
    """Session management data structure"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    status: str = "active"
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_activity: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Memory:
    """Memory entry data structure"""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    memory_type: MemoryType = MemoryType.SHORT_TERM
    content: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5  # 0.0 to 1.0
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


@dataclass
class Message:
    """Message structure for Synapse broker"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    channel: str = "default"
    sender: str = ""
    recipient: Optional[str] = None
    priority: MessagePriority = MessagePriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ToolCall:
    """Tool/API call structure"""
    tool_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    api_endpoint: Optional[str] = None
    method: str = "GET"
    parameters: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 30
    retry_count: int = 0
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


@dataclass
class Event:
    """Event structure for event handlers"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    source: str = ""
    target: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    processed: bool = False


@dataclass
class LangGraphState:
    """LangGraph state machine state"""
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    current_node: str = ""
    next_nodes: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "running"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


# ============================================================================
# TypedDict for API responses
# ============================================================================

class AgentResponse(TypedDict):
    """Agent response structure"""
    agent_id: str
    role: str
    response: str
    confidence: float
    metadata: Dict[str, Any]


class TaskResponse(TypedDict):
    """Task response structure"""
    task_id: str
    status: str
    result: Any
    metadata: Dict[str, Any]


# ============================================================================
# Protocol definitions
# ============================================================================

class MemoryStore(Protocol):
    """Protocol for memory storage backends"""
    async def store(self, memory: Memory) -> str:
        """Store a memory"""
        ...
    
    async def retrieve(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID"""
        ...
    
    async def search(self, query: str, limit: int = 10) -> List[Memory]:
        """Search memories by content"""
        ...


class MessageBroker(Protocol):
    """Protocol for message broker backends"""
    async def publish(self, message: Message) -> bool:
        """Publish a message"""
        ...
    
    async def subscribe(self, channel: str, callback: callable) -> str:
        """Subscribe to a channel"""
        ...
    
    async def consume(self, channel: str, timeout: int = 30) -> Optional[Message]:
        """Consume a message from a channel"""
        ...


# ============================================================================
# Global Constants
# ============================================================================

# Default timeouts
DEFAULT_TIMEOUT = 30
DEFAULT_LLM_TIMEOUT = 300
DEFAULT_API_TIMEOUT = 60

# Session defaults
DEFAULT_SESSION_TTL = 3600  # 1 hour
MAX_SESSION_TTL = 86400  # 24 hours

# Memory defaults
DEFAULT_MEMORY_TTL = 604800  # 7 days
MAX_MEMORY_ITEMS = 1000

# Message broker defaults
DEFAULT_QUEUE_SIZE = 1000
DEFAULT_MESSAGE_TTL = 3600

# Agent defaults
DEFAULT_AGENT_TEMPERATURE = 0.7
DEFAULT_AGENT_MAX_TOKENS = 2000

# Event system defaults
MAX_EVENT_HANDLERS = 100
DEFAULT_EVENT_TIMEOUT = 5

