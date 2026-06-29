"""
Comprehensive API Tools for Agents
Full suite of API integrations for agent tasks, delegating portable tools to
diri-agent-toolbox and deepiri-gpu-utils.
"""

from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from diri_agent_toolbox import AsyncHttpToolbox, ToolRunner
from diri_agent_toolbox.caching import AdvancedCacheManager
from diri_agent_toolbox.confidence import ConfidenceCalculator
from diri_agent_toolbox.database import DatabaseToolbox
from diri_agent_toolbox.device import get_device
from diri_agent_toolbox.files import SandboxedFileToolbox
from diri_agent_toolbox.logging import StructuredLogger
from diri_agent_toolbox.models import ToolResult as ToolboxToolResult
from diri_agent_toolbox.monitoring import MetricsCollector
from diri_agent_toolbox.processing import AsyncBatchProcessor, BatchProcessingConfig

try:
    from deepiri_gpu_utils.detect import detect
    from deepiri_gpu_utils.torch_device import resolve_torch_device

    HAS_GPU_UTILS = True
except ImportError:
    HAS_GPU_UTILS = False

from ...integrations.api_bridge import get_api_bridge
from ...logging_config import get_logger
from ...settings import settings

logger = get_logger("cyrex.agent.tools.api")


class APIMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class ToolCategory(str, Enum):
    DATABASE = "database"
    HTTP = "http"
    FILE = "file"
    SEARCH = "search"
    MATH = "math"
    TEXT = "text"
    DATA = "data"
    CACHE = "cache"
    CONFIDENCE = "confidence"
    DEVICE = "device"
    MONITORING = "monitoring"
    PROCESSING = "processing"
    LOGGING = "logging"
    EXTERNAL_API = "external_api"
    INTERNAL_API = "internal_api"


@dataclass
class ToolDefinition:
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)
    returns: str = "Any"
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": self.parameters,
            "required_params": self.required_params,
            "returns": self.returns,
            "examples": self.examples,
        }


@dataclass
class ToolResult:
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


def _from_toolbox_result(tb: ToolboxToolResult) -> ToolResult:
    return ToolResult(
        success=tb.success,
        result=tb.result,
        error=tb.error,
        execution_time_ms=tb.execution_time_ms,
        metadata=dict(tb.metadata) if tb.metadata else {},
    )


class ComprehensiveAPITools:
    """
    Comprehensive suite of API tools for agent use.

    Portable tools (HTTP, files, data, math, text) delegate to
    ``diri-agent-toolbox``.  New capabilities (caching, confidence, device,
    monitoring, processing, logging) are wired directly from the toolbox.
    GPU/device detection comes from ``deepiri-gpu-utils`` when available.
    Database tools use toolbox ``DatabaseToolbox`` instead of Cyrex-native
    ``get_postgres_manager()``.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id

        # Portable toolbox components
        self._http_toolbox = AsyncHttpToolbox(timeout=30.0, block_private_hosts=False)
        sandbox_root = Path(settings.AGENT_FILE_SANDBOX_ROOT)
        sandbox_root.mkdir(parents=True, exist_ok=True)
        self._file_toolbox = SandboxedFileToolbox(root_dir=sandbox_root)
        self._tool_runner = ToolRunner(http=self._http_toolbox, files=self._file_toolbox)

        # New toolbox components
        self._cache_manager = AdvancedCacheManager()
        self._confidence_calculator = ConfidenceCalculator()
        self._monitor = MetricsCollector(log_dir=str(sandbox_root / "metrics"))
        self._batch_processor = AsyncBatchProcessor(BatchProcessingConfig())
        self._structured_logger = StructuredLogger("cyrex.agent.tools.api")

        # Database toolbox (lazy init with DSN from settings)
        db_dsn: Optional[str] = None
        if hasattr(settings, "DATABASE_URL") and settings.DATABASE_URL:
            db_dsn = settings.DATABASE_URL
        elif hasattr(settings, "database_url") and settings.database_url:
            db_dsn = settings.database_url
        self._database = DatabaseToolbox(dsn=db_dsn)

        self._tool_registry: Dict[str, ToolDefinition] = {}
        self._tool_implementations: Dict[str, Callable] = {}
        self.logger = logger

        self._register_all_tools()

    async def close(self):
        await self._http_toolbox.aclose()
        if self._database:
            await self._database.close()

    def _register_all_tools(self):
        # ====================================================================
        # HTTP Tools
        # ====================================================================
        self._register_tool(
            ToolDefinition(
                name="http_get",
                description="Make an HTTP GET request to a URL",
                category=ToolCategory.HTTP,
                parameters={
                    "url": {"type": "string", "description": "URL to request"},
                    "headers": {"type": "object", "description": "Optional headers"},
                    "params": {"type": "object", "description": "Query parameters"},
                },
                required_params=["url"],
                returns="Response data (JSON or text)",
                examples=[{"url": "https://api.example.com/data", "params": {"limit": 10}}],
            ),
            self._http_get,
        )

        self._register_tool(
            ToolDefinition(
                name="http_post",
                description="Make an HTTP POST request",
                category=ToolCategory.HTTP,
                parameters={
                    "url": {"type": "string", "description": "URL to request"},
                    "data": {"type": "object", "description": "JSON body to send"},
                    "headers": {"type": "object", "description": "Optional headers"},
                },
                required_params=["url"],
                returns="Response data",
            ),
            self._http_post,
        )

        self._register_tool(
            ToolDefinition(
                name="http_request",
                description="Make a custom HTTP request with any method",
                category=ToolCategory.HTTP,
                parameters={
                    "method": {"type": "string", "description": "HTTP method"},
                    "url": {"type": "string", "description": "URL to request"},
                    "data": {"type": "object", "description": "Request body"},
                    "headers": {"type": "object", "description": "Headers"},
                },
                required_params=["method", "url"],
                returns="Response data",
            ),
            self._http_request,
        )

        # ====================================================================
        # Database Tools (via toolbox DatabaseToolbox)
        # ====================================================================
        self._register_tool(
            ToolDefinition(
                name="db_query",
                description="Execute a database query (SELECT only)",
                category=ToolCategory.DATABASE,
                parameters={
                    "query": {"type": "string", "description": "SQL SELECT query"},
                    "params": {"type": "array", "description": "Query parameters"},
                },
                required_params=["query"],
                returns="List of rows as dictionaries",
            ),
            self._db_query,
        )

        self._register_tool(
            ToolDefinition(
                name="db_execute",
                description="Execute a database write operation (INSERT, UPDATE, DELETE)",
                category=ToolCategory.DATABASE,
                parameters={
                    "query": {"type": "string", "description": "SQL query"},
                    "params": {"type": "array", "description": "Query parameters"},
                },
                required_params=["query"],
                returns="Affected row count",
            ),
            self._db_execute,
        )

        self._register_tool(
            ToolDefinition(
                name="db_get_tables",
                description="List all tables in the database",
                category=ToolCategory.DATABASE,
                parameters={},
                required_params=[],
                returns="List of table names",
            ),
            self._db_get_tables,
        )

        # ====================================================================
        # Search Tools
        # ====================================================================
        self._register_tool(
            ToolDefinition(
                name="search_documents",
                description="Search documents in the vector store",
                category=ToolCategory.SEARCH,
                parameters={
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results (default 5)"},
                    "filter": {"type": "object", "description": "Metadata filters"},
                },
                required_params=["query"],
                returns="List of matching documents with scores",
            ),
            self._search_documents,
        )

        self._register_tool(
            ToolDefinition(
                name="search_web",
                description="Search the web for information",
                category=ToolCategory.SEARCH,
                parameters={
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results (default 5)",
                    },
                },
                required_params=["query"],
                returns="Search results with titles and snippets",
            ),
            self._search_web,
        )

        # ====================================================================
        # Math Tools
        # ====================================================================
        self._register_tool(
            ToolDefinition(
                name="calculate",
                description="Evaluate a mathematical expression",
                category=ToolCategory.MATH,
                parameters={
                    "expression": {"type": "string", "description": "Math expression to evaluate"},
                },
                required_params=["expression"],
                returns="Numeric result",
                examples=[{"expression": "2 + 2 * 3"}, {"expression": "sqrt(16) + pow(2, 3)"}],
            ),
            self._calculate,
        )

        self._register_tool(
            ToolDefinition(
                name="statistics",
                description="Calculate statistics for a list of numbers",
                category=ToolCategory.MATH,
                parameters={
                    "numbers": {"type": "array", "description": "List of numbers"},
                    "operations": {
                        "type": "array",
                        "description": "Stats to calculate: mean, median, std, min, max, sum",
                    },
                },
                required_params=["numbers"],
                returns="Dictionary of statistics",
            ),
            self._statistics,
        )

        # ====================================================================
        # Text Tools
        # ====================================================================
        self._register_tool(
            ToolDefinition(
                name="text_summarize",
                description="Summarize text content",
                category=ToolCategory.TEXT,
                parameters={
                    "text": {"type": "string", "description": "Text to summarize"},
                    "max_length": {"type": "integer", "description": "Max summary length"},
                },
                required_params=["text"],
                returns="Summarized text",
            ),
            self._text_summarize,
        )

        self._register_tool(
            ToolDefinition(
                name="text_extract",
                description="Extract structured data from text",
                category=ToolCategory.TEXT,
                parameters={
                    "text": {"type": "string", "description": "Text to extract from"},
                    "fields": {"type": "array", "description": "Fields to extract"},
                },
                required_params=["text", "fields"],
                returns="Dictionary of extracted values",
            ),
            self._text_extract,
        )

        # ====================================================================
        # Data Tools
        # ====================================================================
        self._register_tool(
            ToolDefinition(
                name="json_parse",
                description="Parse JSON string into object",
                category=ToolCategory.DATA,
                parameters={
                    "json_string": {"type": "string", "description": "JSON string to parse"},
                },
                required_params=["json_string"],
                returns="Parsed object",
            ),
            self._json_parse,
        )

        self._register_tool(
            ToolDefinition(
                name="json_format",
                description="Format object as JSON string",
                category=ToolCategory.DATA,
                parameters={
                    "data": {"type": "object", "description": "Data to format"},
                    "indent": {"type": "integer", "description": "Indentation (default 2)"},
                },
                required_params=["data"],
                returns="Formatted JSON string",
            ),
            self._json_format,
        )

        self._register_tool(
            ToolDefinition(
                name="data_transform",
                description="Transform data using a mapping",
                category=ToolCategory.DATA,
                parameters={
                    "data": {"type": "object", "description": "Data to transform"},
                    "mapping": {
                        "type": "object",
                        "description": "Field mapping (new_field: old_field or expression)",
                    },
                },
                required_params=["data", "mapping"],
                returns="Transformed data",
            ),
            self._data_transform,
        )

        # ====================================================================
        # Cache Tools (toolbox AdvancedCacheManager)
        # ====================================================================
        self._register_tool(
            ToolDefinition(
                name="cache_get",
                description="Get a value from the agent cache by key",
                category=ToolCategory.CACHE,
                parameters={
                    "key": {"type": "string", "description": "Cache key"},
                },
                required_params=["key"],
                returns="Cached value or null if not found",
            ),
            self._cache_get,
        )

        self._register_tool(
            ToolDefinition(
                name="cache_set",
                description="Set a value in the agent cache",
                category=ToolCategory.CACHE,
                parameters={
                    "key": {"type": "string", "description": "Cache key"},
                    "value": {"description": "Value to cache"},
                    "ttl_seconds": {
                        "type": "integer",
                        "description": "Time-to-live in seconds (optional)",
                    },
                },
                required_params=["key", "value"],
                returns="Success status",
            ),
            self._cache_set,
        )

        self._register_tool(
            ToolDefinition(
                name="cache_delete",
                description="Delete a key from the agent cache",
                category=ToolCategory.CACHE,
                parameters={
                    "key": {"type": "string", "description": "Cache key to delete"},
                },
                required_params=["key"],
                returns="Whether the key was deleted",
            ),
            self._cache_delete,
        )

        # ====================================================================
        # Device / GPU Tools (deepiri-gpu-utils + toolbox get_device)
        # ====================================================================
        self._register_tool(
            ToolDefinition(
                name="device_info",
                description="Detect available compute device (CUDA/MPS/CPU) using deepiri-gpu-utils",
                category=ToolCategory.DEVICE,
                parameters={
                    "policy": {
                        "type": "string",
                        "description": "Device policy: auto, cuda, mps, cpu (default auto)",
                    },
                },
                required_params=[],
                returns="Device detection result with backend, confidence, and details",
            ),
            self._device_info,
        )

        # ====================================================================
        # Confidence Tools (toolbox ConfidenceCalculator)
        # ====================================================================
        self._register_tool(
            ToolDefinition(
                name="confidence_score",
                description="Calculate confidence score for a prediction",
                category=ToolCategory.CONFIDENCE,
                parameters={
                    "score": {"type": "number", "description": "Raw prediction score (0-1)"},
                    "source": {
                        "type": "string",
                        "description": "Confidence source (model_prediction, feature_quality, etc.)",
                    },
                },
                required_params=["score"],
                returns="Confidence assessment with level, explanation, and uncertainty",
            ),
            self._confidence_score,
        )

        # ====================================================================
        # Monitoring Tools (toolbox MetricsCollector)
        # ====================================================================
        self._register_tool(
            ToolDefinition(
                name="monitor_record",
                description="Record a metric/event for monitoring",
                category=ToolCategory.MONITORING,
                parameters={
                    "operation": {"type": "string", "description": "Operation name"},
                    "data": {"type": "object", "description": "Metric data key-value pairs"},
                },
                required_params=["operation", "data"],
                returns="Success status",
            ),
            self._monitor_record,
        )

        # ====================================================================
        # Processing Tools (toolbox AsyncBatchProcessor)
        # ====================================================================
        self._register_tool(
            ToolDefinition(
                name="batch_process",
                description="Process a batch of items using a processing function description",
                category=ToolCategory.PROCESSING,
                parameters={
                    "items": {"type": "array", "description": "List of items to process"},
                    "processor_description": {
                        "type": "string",
                        "description": "Description of what to do with each item (the agent executes this)",
                    },
                },
                required_params=["items"],
                returns="Batch processing result with success/failure counts",
            ),
            self._batch_process,
        )

        # ====================================================================
        # Logging Tools (toolbox StructuredLogger)
        # ====================================================================
        self._register_tool(
            ToolDefinition(
                name="log_event",
                description="Log a structured event for debugging or audit",
                category=ToolCategory.LOGGING,
                parameters={
                    "event": {"type": "string", "description": "Event name"},
                    "level": {
                        "type": "string",
                        "description": "Log level: info, warning, error (default info)",
                    },
                    "data": {"type": "object", "description": "Additional event data"},
                },
                required_params=["event"],
                returns="Success status",
            ),
            self._log_event,
        )

        # ====================================================================
        # External API Tools
        # ====================================================================
        self._register_tool(
            ToolDefinition(
                name="call_external_api",
                description="Call a registered external API",
                category=ToolCategory.EXTERNAL_API,
                parameters={
                    "api_name": {"type": "string", "description": "Name of the API to call"},
                    "endpoint": {"type": "string", "description": "API endpoint"},
                    "params": {"type": "object", "description": "API parameters"},
                },
                required_params=["api_name", "endpoint"],
                returns="API response",
            ),
            self._call_external_api,
        )

        self._register_tool(
            ToolDefinition(
                name="get_current_time",
                description="Get current date and time",
                category=ToolCategory.DATA,
                parameters={
                    "timezone": {"type": "string", "description": "Timezone (default UTC)"},
                    "format": {"type": "string", "description": "Date format string"},
                },
                required_params=[],
                returns="Current datetime string",
            ),
            self._get_current_time,
        )

        # ====================================================================
        # File Tools (sandboxed; diri-agent-toolbox SandboxedFileToolbox)
        # ====================================================================
        self._register_tool(
            ToolDefinition(
                name="file_read",
                description="Read a text file under the agent file sandbox root",
                category=ToolCategory.FILE,
                parameters={
                    "path": {"type": "string", "description": "Path relative to the sandbox root"},
                    "encoding": {"type": "string", "description": "Text encoding (default utf-8)"},
                },
                required_params=["path"],
                returns="File contents",
            ),
            self._file_read,
        )

        self._register_tool(
            ToolDefinition(
                name="file_write",
                description="Write text to a path under the agent file sandbox root",
                category=ToolCategory.FILE,
                parameters={
                    "path": {"type": "string", "description": "Path relative to the sandbox root"},
                    "content": {"type": "string", "description": "Text content to write"},
                    "encoding": {"type": "string", "description": "Text encoding (default utf-8)"},
                },
                required_params=["path", "content"],
                returns="Write result metadata",
            ),
            self._file_write,
        )

        self._register_tool(
            ToolDefinition(
                name="file_list_dir",
                description="List directory entries under the agent file sandbox root",
                category=ToolCategory.FILE,
                parameters={
                    "path": {
                        "type": "string",
                        "description": "Directory relative to the sandbox root (default '.')",
                    },
                },
                required_params=[],
                returns="List of directory entries",
            ),
            self._file_list_dir,
        )

        self._register_tool(
            ToolDefinition(
                name="file_stat",
                description="Get file metadata under the agent file sandbox root",
                category=ToolCategory.FILE,
                parameters={
                    "path": {"type": "string", "description": "Path relative to the sandbox root"},
                },
                required_params=["path"],
                returns="File metadata",
            ),
            self._file_stat,
        )

        self._register_tool(
            ToolDefinition(
                name="file_delete",
                description="Delete a file under the agent file sandbox root",
                category=ToolCategory.FILE,
                parameters={
                    "path": {"type": "string", "description": "Path relative to the sandbox root"},
                },
                required_params=["path"],
                returns="Delete result metadata",
            ),
            self._file_delete,
        )

        self._register_tool(
            ToolDefinition(
                name="file_copy",
                description="Copy a file within the sandbox",
                category=ToolCategory.FILE,
                parameters={
                    "src": {
                        "type": "string",
                        "description": "Source path relative to sandbox root",
                    },
                    "dst": {
                        "type": "string",
                        "description": "Destination path relative to sandbox root",
                    },
                },
                required_params=["src", "dst"],
                returns="Copy result metadata",
            ),
            self._file_copy,
        )

        self._register_tool(
            ToolDefinition(
                name="file_move",
                description="Move/rename a file within the sandbox",
                category=ToolCategory.FILE,
                parameters={
                    "src": {
                        "type": "string",
                        "description": "Source path relative to sandbox root",
                    },
                    "dst": {
                        "type": "string",
                        "description": "Destination path relative to sandbox root",
                    },
                },
                required_params=["src", "dst"],
                returns="Move result metadata",
            ),
            self._file_move,
        )

        self._register_tool(
            ToolDefinition(
                name="file_read_binary",
                description="Read a file as binary/base64 under the agent file sandbox root",
                category=ToolCategory.FILE,
                parameters={
                    "path": {"type": "string", "description": "Path relative to the sandbox root"},
                },
                required_params=["path"],
                returns="Base64-encoded file content",
            ),
            self._file_read_binary,
        )

    def _register_tool(self, definition: ToolDefinition, implementation: Callable):
        self._tool_registry[definition.name] = definition
        self._tool_implementations[definition.name] = implementation

    # ========================================================================
    # Tool Implementations
    # ========================================================================

    async def _http_get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        r = await self._tool_runner.execute("http_get", url=url, headers=headers, params=params)
        return _from_toolbox_result(r)

    async def _http_post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> ToolResult:
        r = await self._tool_runner.execute("http_post", url=url, data=data, headers=headers)
        return _from_toolbox_result(r)

    async def _http_request(
        self,
        method: str,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> ToolResult:
        r = await self._tool_runner.execute(
            "http_request", method=method, url=url, data=data, headers=headers
        )
        return _from_toolbox_result(r)

    async def _db_query(
        self,
        query: str,
        params: Optional[List[Any]] = None,
    ) -> ToolResult:
        r = await self._database.query(query, *(params or []))
        return _from_toolbox_result(r)

    async def _db_execute(
        self,
        query: str,
        params: Optional[List[Any]] = None,
    ) -> ToolResult:
        r = await self._database.execute(query, *(params or []))
        return _from_toolbox_result(r)

    async def _db_get_tables(self) -> ToolResult:
        r = await self._database.get_tables()
        return _from_toolbox_result(r)

    async def _search_documents(
        self,
        query: str,
        limit: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        try:
            api_bridge = await get_api_bridge()
            result = await api_bridge.call_tool(
                "search_documents",
                {
                    "query": query,
                    "limit": limit,
                    "filter": filter,
                },
            )
            return ToolResult(
                success=True, result=result.result if hasattr(result, "result") else result
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _search_web(
        self,
        query: str,
        num_results: int = 5,
    ) -> ToolResult:
        return ToolResult(
            success=True,
            result={
                "message": "Web search not implemented - integrate with search API",
                "query": query,
            },
        )

    async def _calculate(self, expression: str) -> ToolResult:
        r = await self._tool_runner.execute("calculate", expression=expression)
        return _from_toolbox_result(r)

    async def _statistics(
        self,
        numbers: List[Union[int, float]],
        operations: Optional[List[str]] = None,
    ) -> ToolResult:
        r = await self._tool_runner.execute("statistics", numbers=numbers, operations=operations)
        return _from_toolbox_result(r)

    async def _text_summarize(self, text: str, max_length: int = 200) -> ToolResult:
        r = await self._tool_runner.execute("text_summarize", text=text, max_length=max_length)
        return _from_toolbox_result(r)

    async def _text_extract(self, text: str, fields: List[str]) -> ToolResult:
        r = await self._tool_runner.execute("text_extract", text=text, fields=fields)
        return _from_toolbox_result(r)

    async def _json_parse(self, json_string: str) -> ToolResult:
        r = await self._tool_runner.execute("json_parse", json_string=json_string)
        return _from_toolbox_result(r)

    async def _json_format(self, data: Any, indent: int = 2) -> ToolResult:
        r = await self._tool_runner.execute("json_format", data=data, indent=indent)
        return _from_toolbox_result(r)

    async def _data_transform(self, data: Dict[str, Any], mapping: Dict[str, str]) -> ToolResult:
        r = await self._tool_runner.execute("data_transform", data=data, mapping=mapping)
        return _from_toolbox_result(r)

    async def _cache_get(self, key: str) -> ToolResult:
        try:
            entry = self._cache_manager.get(key)
            if entry is None:
                return ToolResult(success=True, result=None, metadata={"found": False})
            return ToolResult(
                success=True,
                result=entry.value if hasattr(entry, "value") else entry,
                metadata={"found": True},
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _cache_set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> ToolResult:
        try:
            self._cache_manager.set(key, value, ttl=ttl_seconds)
            return ToolResult(success=True, result="cached")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _cache_delete(self, key: str) -> ToolResult:
        try:
            deleted = self._cache_manager.delete(key)
            return ToolResult(success=True, result=deleted, metadata={"deleted": deleted})
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _device_info(self, policy: str = "auto") -> ToolResult:
        try:
            info: Dict[str, Any] = {
                "toolbox_device": get_device(),
            }
            if HAS_GPU_UTILS:
                detect_result = detect()
                info["backend"] = detect_result.backend
                info["confidence"] = detect_result.confidence
                info["warnings"] = detect_result.warnings

                torch_result = resolve_torch_device(policy=policy)  # type: ignore[arg-type]
                info["torch_device"] = torch_result.device
                info["torch_available"] = torch_result.torch_available
                info["notes"] = torch_result.notes
            else:
                info["note"] = "deepiri-gpu-utils not installed; using toolbox get_device()"
            return ToolResult(success=True, result=info)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _confidence_score(self, score: float, source: str = "model_prediction") -> ToolResult:
        try:
            result = self._confidence_calculator.evaluate(score=score, source=source)
            if hasattr(result, "to_dict"):
                return ToolResult(success=True, result=result.to_dict())
            return ToolResult(success=True, result=str(result))
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _monitor_record(self, operation: str, data: Dict[str, Any]) -> ToolResult:
        try:
            self._monitor.record(operation, data)
            return ToolResult(success=True, result="recorded")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _batch_process(
        self,
        items: List[Any],
        processor_description: str = "",
    ) -> ToolResult:
        try:
            result = ToolResult(
                success=True,
                result={
                    "total_items": len(items),
                    "message": f"Batch processor ready. Items: {len(items)}. Description: {processor_description}. "
                    "Processing function must be provided by agent logic.",
                },
            )
            return result
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _log_event(
        self,
        event: str,
        level: str = "info",
        data: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        try:
            log_data = data or {}
            log_data["event"] = event
            if level == "error":
                self._structured_logger.error(log_data)
            elif level == "warning":
                self._structured_logger.warning(log_data)
            else:
                self._structured_logger.info(log_data)
            return ToolResult(success=True, result="logged")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _call_external_api(
        self,
        api_name: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        try:
            api_bridge = await get_api_bridge()
            result = await api_bridge.call_tool(f"{api_name}_{endpoint}", params or {})
            return ToolResult(
                success=True, result=result.result if hasattr(result, "result") else result
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _get_current_time(
        self, timezone: str = "UTC", format: str = "%Y-%m-%d %H:%M:%S"
    ) -> ToolResult:
        r = await self._tool_runner.execute("current_time", timezone=timezone, format=format)
        return _from_toolbox_result(r)

    async def _file_read(self, path: str, encoding: str = "utf-8") -> ToolResult:
        r = await self._file_toolbox.read_text(path, encoding=encoding)
        return _from_toolbox_result(r)

    async def _file_write(self, path: str, content: str, encoding: str = "utf-8") -> ToolResult:
        r = await self._file_toolbox.write_text(path, content, encoding=encoding)
        return _from_toolbox_result(r)

    async def _file_list_dir(self, path: str = ".") -> ToolResult:
        r = await self._file_toolbox.list_dir(path)
        return _from_toolbox_result(r)

    async def _file_stat(self, path: str) -> ToolResult:
        r = await self._file_toolbox.stat(path)
        return _from_toolbox_result(r)

    async def _file_delete(self, path: str) -> ToolResult:
        r = await self._file_toolbox.delete(path)
        return _from_toolbox_result(r)

    async def _file_copy(self, src: str, dst: str) -> ToolResult:
        r = await self._file_toolbox.copy(src, dst)
        return _from_toolbox_result(r)

    async def _file_move(self, src: str, dst: str) -> ToolResult:
        r = await self._file_toolbox.move(src, dst)
        return _from_toolbox_result(r)

    async def _file_read_binary(self, path: str) -> ToolResult:
        r = await self._file_toolbox.read_binary(path)
        return _from_toolbox_result(r)

    # ========================================================================
    # Public API
    # ========================================================================

    def list_tools(self, category: Optional[ToolCategory] = None) -> List[ToolDefinition]:
        tools = list(self._tool_registry.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tool_registry.get(name)

    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        if tool_name not in self._tool_implementations:
            return ToolResult(success=False, error=f"Tool not found: {tool_name}")

        implementation = self._tool_implementations[tool_name]

        try:
            result = await implementation(**kwargs)
            return result
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def format_for_prompt(self) -> str:
        lines = ["Available Tools:"]

        for category in ToolCategory:
            category_tools = [t for t in self._tool_registry.values() if t.category == category]
            if not category_tools:
                continue

            lines.append(f"\n## {category.value.upper()}")
            for tool in category_tools:
                lines.append(f"- **{tool.name}**: {tool.description}")
                if tool.required_params:
                    lines.append(f"  Required: {', '.join(tool.required_params)}")

        return "\n".join(lines)


# ============================================================================
# Registration Helper
# ============================================================================


async def register_api_tools(agent, session_id: Optional[str] = None) -> ComprehensiveAPITools:
    api_tools = ComprehensiveAPITools(session_id=session_id)

    for tool_def in api_tools.list_tools():

        async def make_tool_func(name: str):
            async def tool_func(**kwargs) -> Dict[str, Any]:
                result = await api_tools.execute(name, **kwargs)
                return {
                    "success": result.success,
                    "result": result.result,
                    "error": result.error,
                }

            return tool_func

        agent.register_tool(
            tool_def.name,
            await make_tool_func(tool_def.name),
            tool_def.description,
        )

    return api_tools
