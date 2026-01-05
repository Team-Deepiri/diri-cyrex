"""
Automation Tools and Agent Purposes
Defines the tools needed for task automation and agent capabilities
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum


class AgentPurpose(str, Enum):
    """Purposes/roles for agents in the automation system"""
    
    # Task Management
    TASK_DECOMPOSER = "task_decomposer"
    TASK_EXECUTOR = "task_executor"
    TASK_VALIDATOR = "task_validator"
    
    # Code & Development
    CODE_GENERATOR = "code_generator"
    CODE_REVIEWER = "code_reviewer"
    BUG_FIXER = "bug_fixer"
    
    # Data & Analysis
    DATA_ANALYST = "data_analyst"
    DATA_TRANSFORMER = "data_transformer"
    REPORT_GENERATOR = "report_generator"
    
    # Document Processing
    DOCUMENT_PROCESSOR = "document_processor"
    DOCUMENT_SUMMARIZER = "document_summarizer"
    INFORMATION_EXTRACTOR = "information_extractor"
    
    # Communication
    EMAIL_COMPOSER = "email_composer"
    NOTIFICATION_MANAGER = "notification_manager"
    
    # Integration
    API_INTEGRATOR = "api_integrator"
    DATABASE_MANAGER = "database_manager"
    FILE_MANAGER = "file_manager"
    
    # Quality & Safety
    QUALITY_ASSURANCE = "quality_assurance"
    SECURITY_AUDITOR = "security_auditor"
    COMPLIANCE_CHECKER = "compliance_checker"
    
    # Specialized
    VENDOR_FRAUD_DETECTOR = "vendor_fraud_detector"
    INVOICE_ANALYZER = "invoice_analyzer"
    SCHEDULER = "scheduler"
    ORCHESTRATOR = "orchestrator"


class ToolCategory(str, Enum):
    """Categories of automation tools"""
    MEMORY = "memory"
    HTTP = "http"
    DATABASE = "database"
    FILE = "file"
    SEARCH = "search"
    MATH = "math"
    TEXT = "text"
    DATA = "data"
    NOTIFICATION = "notification"
    SCHEDULING = "scheduling"
    VALIDATION = "validation"
    INTEGRATION = "integration"


@dataclass
class AutomationTool:
    """Definition of an automation tool"""
    name: str
    description: str
    category: ToolCategory
    purpose: List[AgentPurpose]  # Which agent purposes can use this tool
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)
    returns: str = "Any"
    async_required: bool = False
    rate_limited: bool = False
    requires_auth: bool = False
    examples: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# AUTOMATION TOOLS REGISTRY
# ============================================================================

AUTOMATION_TOOLS: Dict[str, AutomationTool] = {}

# -----------------------------------------------------------------------------
# Memory Tools
# -----------------------------------------------------------------------------

AUTOMATION_TOOLS["store_memory"] = AutomationTool(
    name="store_memory",
    description="Store information in agent memory for later retrieval",
    category=ToolCategory.MEMORY,
    purpose=[
        AgentPurpose.TASK_DECOMPOSER,
        AgentPurpose.DATA_ANALYST,
        AgentPurpose.DOCUMENT_PROCESSOR,
        AgentPurpose.ORCHESTRATOR,
    ],
    parameters={
        "content": {"type": "string", "description": "Content to store"},
        "memory_type": {"type": "string", "description": "Type: short_term, long_term, episodic, semantic"},
        "importance": {"type": "number", "description": "Importance score 0-1"},
        "tags": {"type": "array", "description": "Tags for categorization"},
    },
    required_params=["content"],
    returns="Memory ID string",
    async_required=True,
)

AUTOMATION_TOOLS["recall_memory"] = AutomationTool(
    name="recall_memory",
    description="Search and retrieve memories by semantic similarity",
    category=ToolCategory.MEMORY,
    purpose=[
        AgentPurpose.TASK_DECOMPOSER,
        AgentPurpose.DATA_ANALYST,
        AgentPurpose.DOCUMENT_PROCESSOR,
        AgentPurpose.ORCHESTRATOR,
        AgentPurpose.CODE_GENERATOR,
    ],
    parameters={
        "query": {"type": "string", "description": "Search query"},
        "limit": {"type": "integer", "description": "Max results to return"},
        "memory_type": {"type": "string", "description": "Filter by memory type"},
    },
    required_params=["query"],
    returns="List of matching memories with scores",
    async_required=True,
)

AUTOMATION_TOOLS["store_fact"] = AutomationTool(
    name="store_fact",
    description="Store a fact as a knowledge triple (subject, predicate, object)",
    category=ToolCategory.MEMORY,
    purpose=[
        AgentPurpose.INFORMATION_EXTRACTOR,
        AgentPurpose.DATA_ANALYST,
        AgentPurpose.DOCUMENT_PROCESSOR,
    ],
    parameters={
        "subject": {"type": "string", "description": "Subject of the fact"},
        "predicate": {"type": "string", "description": "Relationship/predicate"},
        "object": {"type": "string", "description": "Object of the fact"},
        "confidence": {"type": "number", "description": "Confidence score 0-1"},
    },
    required_params=["subject", "predicate", "object"],
    returns="Memory ID string",
    async_required=True,
)

# -----------------------------------------------------------------------------
# HTTP Tools
# -----------------------------------------------------------------------------

AUTOMATION_TOOLS["http_get"] = AutomationTool(
    name="http_get",
    description="Make HTTP GET request to external API",
    category=ToolCategory.HTTP,
    purpose=[
        AgentPurpose.API_INTEGRATOR,
        AgentPurpose.DATA_ANALYST,
        AgentPurpose.VENDOR_FRAUD_DETECTOR,
    ],
    parameters={
        "url": {"type": "string", "description": "URL to request"},
        "headers": {"type": "object", "description": "Request headers"},
        "params": {"type": "object", "description": "Query parameters"},
    },
    required_params=["url"],
    returns="Response data (JSON or text)",
    async_required=True,
    rate_limited=True,
)

AUTOMATION_TOOLS["http_post"] = AutomationTool(
    name="http_post",
    description="Make HTTP POST request with JSON body",
    category=ToolCategory.HTTP,
    purpose=[
        AgentPurpose.API_INTEGRATOR,
        AgentPurpose.NOTIFICATION_MANAGER,
    ],
    parameters={
        "url": {"type": "string", "description": "URL to request"},
        "data": {"type": "object", "description": "JSON body to send"},
        "headers": {"type": "object", "description": "Request headers"},
    },
    required_params=["url"],
    returns="Response data",
    async_required=True,
    rate_limited=True,
)

# -----------------------------------------------------------------------------
# Database Tools
# -----------------------------------------------------------------------------

AUTOMATION_TOOLS["db_query"] = AutomationTool(
    name="db_query",
    description="Execute SELECT query on database",
    category=ToolCategory.DATABASE,
    purpose=[
        AgentPurpose.DATABASE_MANAGER,
        AgentPurpose.DATA_ANALYST,
        AgentPurpose.REPORT_GENERATOR,
    ],
    parameters={
        "query": {"type": "string", "description": "SQL SELECT query"},
        "params": {"type": "array", "description": "Query parameters"},
    },
    required_params=["query"],
    returns="List of rows as dictionaries",
    async_required=True,
)

AUTOMATION_TOOLS["db_execute"] = AutomationTool(
    name="db_execute",
    description="Execute write operation (INSERT, UPDATE, DELETE)",
    category=ToolCategory.DATABASE,
    purpose=[
        AgentPurpose.DATABASE_MANAGER,
    ],
    parameters={
        "query": {"type": "string", "description": "SQL query"},
        "params": {"type": "array", "description": "Query parameters"},
    },
    required_params=["query"],
    returns="Affected row count",
    async_required=True,
    requires_auth=True,
)

# -----------------------------------------------------------------------------
# File Tools
# -----------------------------------------------------------------------------

AUTOMATION_TOOLS["read_file"] = AutomationTool(
    name="read_file",
    description="Read contents of a file",
    category=ToolCategory.FILE,
    purpose=[
        AgentPurpose.FILE_MANAGER,
        AgentPurpose.DOCUMENT_PROCESSOR,
        AgentPurpose.CODE_REVIEWER,
    ],
    parameters={
        "path": {"type": "string", "description": "File path to read"},
        "encoding": {"type": "string", "description": "File encoding (default: utf-8)"},
    },
    required_params=["path"],
    returns="File contents as string",
    async_required=True,
)

AUTOMATION_TOOLS["write_file"] = AutomationTool(
    name="write_file",
    description="Write contents to a file",
    category=ToolCategory.FILE,
    purpose=[
        AgentPurpose.FILE_MANAGER,
        AgentPurpose.CODE_GENERATOR,
        AgentPurpose.REPORT_GENERATOR,
    ],
    parameters={
        "path": {"type": "string", "description": "File path to write"},
        "content": {"type": "string", "description": "Content to write"},
        "mode": {"type": "string", "description": "Write mode: overwrite, append"},
    },
    required_params=["path", "content"],
    returns="Success boolean",
    async_required=True,
    requires_auth=True,
)

# -----------------------------------------------------------------------------
# Search Tools
# -----------------------------------------------------------------------------

AUTOMATION_TOOLS["search_documents"] = AutomationTool(
    name="search_documents",
    description="Search documents using vector similarity",
    category=ToolCategory.SEARCH,
    purpose=[
        AgentPurpose.DOCUMENT_PROCESSOR,
        AgentPurpose.INFORMATION_EXTRACTOR,
        AgentPurpose.DATA_ANALYST,
    ],
    parameters={
        "query": {"type": "string", "description": "Search query"},
        "limit": {"type": "integer", "description": "Max results"},
        "filters": {"type": "object", "description": "Metadata filters"},
    },
    required_params=["query"],
    returns="List of matching documents with scores",
    async_required=True,
)

AUTOMATION_TOOLS["search_web"] = AutomationTool(
    name="search_web",
    description="Search the web for information",
    category=ToolCategory.SEARCH,
    purpose=[
        AgentPurpose.DATA_ANALYST,
        AgentPurpose.VENDOR_FRAUD_DETECTOR,
    ],
    parameters={
        "query": {"type": "string", "description": "Search query"},
        "num_results": {"type": "integer", "description": "Number of results"},
    },
    required_params=["query"],
    returns="Search results with titles and snippets",
    async_required=True,
    rate_limited=True,
)

# -----------------------------------------------------------------------------
# Text Processing Tools
# -----------------------------------------------------------------------------

AUTOMATION_TOOLS["summarize_text"] = AutomationTool(
    name="summarize_text",
    description="Generate summary of text content",
    category=ToolCategory.TEXT,
    purpose=[
        AgentPurpose.DOCUMENT_SUMMARIZER,
        AgentPurpose.REPORT_GENERATOR,
    ],
    parameters={
        "text": {"type": "string", "description": "Text to summarize"},
        "max_length": {"type": "integer", "description": "Max summary length"},
        "style": {"type": "string", "description": "Summary style: bullet, paragraph, executive"},
    },
    required_params=["text"],
    returns="Summarized text",
    async_required=True,
)

AUTOMATION_TOOLS["extract_entities"] = AutomationTool(
    name="extract_entities",
    description="Extract named entities from text",
    category=ToolCategory.TEXT,
    purpose=[
        AgentPurpose.INFORMATION_EXTRACTOR,
        AgentPurpose.DOCUMENT_PROCESSOR,
        AgentPurpose.INVOICE_ANALYZER,
    ],
    parameters={
        "text": {"type": "string", "description": "Text to extract from"},
        "entity_types": {"type": "array", "description": "Types to extract: person, org, date, money, etc."},
    },
    required_params=["text"],
    returns="Dictionary of extracted entities by type",
    async_required=True,
)

AUTOMATION_TOOLS["analyze_sentiment"] = AutomationTool(
    name="analyze_sentiment",
    description="Analyze sentiment of text",
    category=ToolCategory.TEXT,
    purpose=[
        AgentPurpose.DATA_ANALYST,
        AgentPurpose.QUALITY_ASSURANCE,
    ],
    parameters={
        "text": {"type": "string", "description": "Text to analyze"},
    },
    required_params=["text"],
    returns="Sentiment score and label",
)

# -----------------------------------------------------------------------------
# Data Tools
# -----------------------------------------------------------------------------

AUTOMATION_TOOLS["transform_data"] = AutomationTool(
    name="transform_data",
    description="Transform data using field mapping",
    category=ToolCategory.DATA,
    purpose=[
        AgentPurpose.DATA_TRANSFORMER,
        AgentPurpose.DATA_ANALYST,
    ],
    parameters={
        "data": {"type": "object", "description": "Data to transform"},
        "mapping": {"type": "object", "description": "Field mapping"},
    },
    required_params=["data", "mapping"],
    returns="Transformed data object",
)

AUTOMATION_TOOLS["calculate_statistics"] = AutomationTool(
    name="calculate_statistics",
    description="Calculate statistics for numeric data",
    category=ToolCategory.MATH,
    purpose=[
        AgentPurpose.DATA_ANALYST,
        AgentPurpose.REPORT_GENERATOR,
    ],
    parameters={
        "numbers": {"type": "array", "description": "List of numbers"},
        "operations": {"type": "array", "description": "Stats: mean, median, std, min, max"},
    },
    required_params=["numbers"],
    returns="Dictionary of calculated statistics",
)

AUTOMATION_TOOLS["compare_data"] = AutomationTool(
    name="compare_data",
    description="Compare two datasets and find differences",
    category=ToolCategory.DATA,
    purpose=[
        AgentPurpose.DATA_ANALYST,
        AgentPurpose.QUALITY_ASSURANCE,
        AgentPurpose.VENDOR_FRAUD_DETECTOR,
    ],
    parameters={
        "data1": {"type": "object", "description": "First dataset"},
        "data2": {"type": "object", "description": "Second dataset"},
        "key_fields": {"type": "array", "description": "Fields to use as keys"},
    },
    required_params=["data1", "data2"],
    returns="Comparison results with additions, deletions, changes",
)

# -----------------------------------------------------------------------------
# Validation Tools
# -----------------------------------------------------------------------------

AUTOMATION_TOOLS["validate_json"] = AutomationTool(
    name="validate_json",
    description="Validate JSON against a schema",
    category=ToolCategory.VALIDATION,
    purpose=[
        AgentPurpose.QUALITY_ASSURANCE,
        AgentPurpose.DATA_TRANSFORMER,
    ],
    parameters={
        "data": {"type": "object", "description": "JSON data to validate"},
        "schema": {"type": "object", "description": "JSON schema"},
    },
    required_params=["data", "schema"],
    returns="Validation result with errors if any",
)

AUTOMATION_TOOLS["check_safety"] = AutomationTool(
    name="check_safety",
    description="Check content for safety issues",
    category=ToolCategory.VALIDATION,
    purpose=[
        AgentPurpose.SECURITY_AUDITOR,
        AgentPurpose.QUALITY_ASSURANCE,
        AgentPurpose.COMPLIANCE_CHECKER,
    ],
    parameters={
        "content": {"type": "string", "description": "Content to check"},
        "policies": {"type": "array", "description": "Policies to enforce"},
    },
    required_params=["content"],
    returns="Safety check result with risk level",
    async_required=True,
)

# -----------------------------------------------------------------------------
# Notification Tools
# -----------------------------------------------------------------------------

AUTOMATION_TOOLS["send_notification"] = AutomationTool(
    name="send_notification",
    description="Send notification via configured channel",
    category=ToolCategory.NOTIFICATION,
    purpose=[
        AgentPurpose.NOTIFICATION_MANAGER,
        AgentPurpose.ORCHESTRATOR,
    ],
    parameters={
        "channel": {"type": "string", "description": "Channel: email, slack, webhook"},
        "recipient": {"type": "string", "description": "Recipient identifier"},
        "message": {"type": "string", "description": "Message content"},
        "priority": {"type": "string", "description": "Priority: low, normal, high, urgent"},
    },
    required_params=["channel", "recipient", "message"],
    returns="Send result with message ID",
    async_required=True,
    requires_auth=True,
)

# -----------------------------------------------------------------------------
# Scheduling Tools
# -----------------------------------------------------------------------------

AUTOMATION_TOOLS["schedule_task"] = AutomationTool(
    name="schedule_task",
    description="Schedule a task for future execution",
    category=ToolCategory.SCHEDULING,
    purpose=[
        AgentPurpose.SCHEDULER,
        AgentPurpose.ORCHESTRATOR,
    ],
    parameters={
        "task_type": {"type": "string", "description": "Type of task to schedule"},
        "execute_at": {"type": "string", "description": "ISO datetime for execution"},
        "payload": {"type": "object", "description": "Task payload"},
        "priority": {"type": "string", "description": "Priority level"},
    },
    required_params=["task_type", "execute_at"],
    returns="Task ID",
    async_required=True,
)

AUTOMATION_TOOLS["cancel_task"] = AutomationTool(
    name="cancel_task",
    description="Cancel a scheduled task",
    category=ToolCategory.SCHEDULING,
    purpose=[
        AgentPurpose.SCHEDULER,
        AgentPurpose.ORCHESTRATOR,
    ],
    parameters={
        "task_id": {"type": "string", "description": "Task ID to cancel"},
    },
    required_params=["task_id"],
    returns="Cancellation result",
    async_required=True,
)

# -----------------------------------------------------------------------------
# Vendor Fraud Detection Tools
# -----------------------------------------------------------------------------

AUTOMATION_TOOLS["analyze_invoice"] = AutomationTool(
    name="analyze_invoice",
    description="Analyze invoice for fraud indicators",
    category=ToolCategory.VALIDATION,
    purpose=[
        AgentPurpose.VENDOR_FRAUD_DETECTOR,
        AgentPurpose.INVOICE_ANALYZER,
    ],
    parameters={
        "invoice_data": {"type": "object", "description": "Parsed invoice data"},
        "vendor_history": {"type": "object", "description": "Historical vendor data"},
        "industry": {"type": "string", "description": "Industry for benchmarking"},
    },
    required_params=["invoice_data"],
    returns="Fraud risk assessment with score and indicators",
    async_required=True,
)

AUTOMATION_TOOLS["benchmark_prices"] = AutomationTool(
    name="benchmark_prices",
    description="Compare prices against market benchmarks",
    category=ToolCategory.VALIDATION,
    purpose=[
        AgentPurpose.VENDOR_FRAUD_DETECTOR,
        AgentPurpose.DATA_ANALYST,
    ],
    parameters={
        "items": {"type": "array", "description": "Items with prices"},
        "industry": {"type": "string", "description": "Industry for comparison"},
        "region": {"type": "string", "description": "Geographic region"},
    },
    required_params=["items"],
    returns="Price analysis with deviations from benchmarks",
    async_required=True,
)


# ============================================================================
# AGENT PURPOSE DEFINITIONS
# ============================================================================

@dataclass
class AgentPurposeDefinition:
    """Full definition of an agent purpose"""
    purpose: AgentPurpose
    name: str
    description: str
    primary_tools: List[str]
    optional_tools: List[str]
    system_prompt_template: str
    capabilities: List[str]
    use_cases: List[str]


AGENT_PURPOSE_DEFINITIONS: Dict[AgentPurpose, AgentPurposeDefinition] = {
    AgentPurpose.TASK_DECOMPOSER: AgentPurposeDefinition(
        purpose=AgentPurpose.TASK_DECOMPOSER,
        name="Task Decomposer Agent",
        description="Breaks down complex tasks into manageable subtasks with dependencies",
        primary_tools=["store_memory", "recall_memory"],
        optional_tools=["search_documents"],
        system_prompt_template="""You are a Task Decomposition Agent specialized in breaking down complex tasks.

Your capabilities:
- Analyze task complexity
- Identify dependencies between subtasks
- Estimate time and effort
- Prioritize subtasks

Always provide structured output with clear subtask definitions.""",
        capabilities=["task analysis", "dependency mapping", "time estimation", "prioritization"],
        use_cases=["Project planning", "Sprint planning", "Work breakdown"],
    ),
    
    AgentPurpose.CODE_GENERATOR: AgentPurposeDefinition(
        purpose=AgentPurpose.CODE_GENERATOR,
        name="Code Generator Agent",
        description="Generates code based on requirements and specifications",
        primary_tools=["recall_memory", "search_documents", "write_file"],
        optional_tools=["read_file", "http_get"],
        system_prompt_template="""You are a Code Generation Agent specialized in writing clean, efficient code.

Your capabilities:
- Generate code in multiple languages
- Follow best practices
- Include documentation
- Handle edge cases

Always provide complete, working code with examples.""",
        capabilities=["multi-language", "documentation", "testing", "best practices"],
        use_cases=["Feature development", "API creation", "Script generation"],
    ),
    
    AgentPurpose.DATA_ANALYST: AgentPurposeDefinition(
        purpose=AgentPurpose.DATA_ANALYST,
        name="Data Analyst Agent",
        description="Analyzes data and provides insights with visualizations",
        primary_tools=["db_query", "calculate_statistics", "transform_data", "compare_data"],
        optional_tools=["search_documents", "http_get", "store_memory"],
        system_prompt_template="""You are a Data Analysis Agent specialized in extracting insights from data.

Your capabilities:
- Statistical analysis
- Pattern recognition
- Trend identification
- Anomaly detection

Always provide clear insights with supporting evidence.""",
        capabilities=["statistics", "visualization", "trend analysis", "anomaly detection"],
        use_cases=["Business intelligence", "Performance analysis", "Market research"],
    ),
    
    AgentPurpose.VENDOR_FRAUD_DETECTOR: AgentPurposeDefinition(
        purpose=AgentPurpose.VENDOR_FRAUD_DETECTOR,
        name="Vendor Fraud Detection Agent",
        description="Detects fraudulent vendor activities and invoice anomalies",
        primary_tools=["analyze_invoice", "benchmark_prices", "compare_data", "db_query"],
        optional_tools=["search_web", "recall_memory", "check_safety"],
        system_prompt_template="""You are a Vendor Fraud Detection Agent specialized in identifying fraudulent activities.

Your capabilities:
- Invoice analysis
- Price benchmarking
- Pattern detection
- Risk scoring

Always provide detailed risk assessments with evidence.""",
        capabilities=["fraud detection", "risk scoring", "anomaly detection", "compliance"],
        use_cases=["Invoice verification", "Vendor vetting", "Audit support"],
    ),
    
    AgentPurpose.ORCHESTRATOR: AgentPurposeDefinition(
        purpose=AgentPurpose.ORCHESTRATOR,
        name="Orchestrator Agent",
        description="Coordinates multiple agents and manages workflow execution",
        primary_tools=["schedule_task", "cancel_task", "send_notification", "store_memory"],
        optional_tools=["recall_memory", "check_safety"],
        system_prompt_template="""You are an Orchestrator Agent that coordinates complex workflows.

Your capabilities:
- Multi-agent coordination
- Workflow management
- Error handling
- Progress tracking

Always maintain clear state and handle failures gracefully.""",
        capabilities=["coordination", "scheduling", "error handling", "monitoring"],
        use_cases=["Complex workflows", "Multi-step processes", "Agent coordination"],
    ),
}


def get_tools_for_purpose(purpose: AgentPurpose) -> List[AutomationTool]:
    """Get all tools available for an agent purpose"""
    return [
        tool for tool in AUTOMATION_TOOLS.values()
        if purpose in tool.purpose
    ]


def get_purpose_definition(purpose: AgentPurpose) -> Optional[AgentPurposeDefinition]:
    """Get the definition for an agent purpose"""
    return AGENT_PURPOSE_DEFINITIONS.get(purpose)


def list_all_tools() -> List[AutomationTool]:
    """List all automation tools"""
    return list(AUTOMATION_TOOLS.values())


def list_all_purposes() -> List[AgentPurposeDefinition]:
    """List all agent purpose definitions"""
    return list(AGENT_PURPOSE_DEFINITIONS.values())

