"""
Comprehensive API Tools for Agents
Full suite of API integrations for agent tasks
"""
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from diri_agent_toolbox import AsyncHttpToolbox, ToolRunner
from diri_agent_toolbox.files import SandboxedFileToolbox
from diri_agent_toolbox.models import ToolResult as ToolboxToolResult

from ...integrations.api_bridge import get_api_bridge
from ...database.postgres import get_postgres_manager
from ...logging_config import get_logger
from ...settings import settings

logger = get_logger("cyrex.agent.tools.api")


class APIMethod(str, Enum):
    """HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class ToolCategory(str, Enum):
    """Tool categories"""
    DATABASE = "database"
    HTTP = "http"
    FILE = "file"
    SEARCH = "search"
    MATH = "math"
    TEXT = "text"
    DATA = "data"
    EXTERNAL_API = "external_api"
    INTERNAL_API = "internal_api"


@dataclass
class ToolDefinition:
    """Definition of an agent tool"""
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
    """Result from tool execution"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


def _from_toolbox_result(tb: ToolboxToolResult) -> ToolResult:
    """Map diri-agent-toolbox ToolResult to Cyrex ToolResult dataclass."""
    return ToolResult(
        success=tb.success,
        result=tb.result,
        error=tb.error,
        execution_time_ms=tb.execution_time_ms,
        metadata=dict(tb.metadata) if tb.metadata else {},
    )


class ComprehensiveAPITools:
    """
    Comprehensive suite of API tools for agent use
    Includes database, HTTP, file, and utility tools
    """
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        # Portable HTTP/data/math via diri-agent-toolbox (matches Division Doc extraction).
        self._http_toolbox = AsyncHttpToolbox(timeout=30.0, block_private_hosts=False)
        # Sandboxed file I/O confined to the configured root (paths outside are rejected).
        sandbox_root = Path(settings.AGENT_FILE_SANDBOX_ROOT)
        sandbox_root.mkdir(parents=True, exist_ok=True)
        self._file_toolbox = SandboxedFileToolbox(root_dir=sandbox_root)
        self._tool_runner = ToolRunner(http=self._http_toolbox, files=self._file_toolbox)
        self._tool_registry: Dict[str, ToolDefinition] = {}
        self._tool_implementations: Dict[str, Callable] = {}
        self.logger = logger
        
        # Register all tools
        self._register_all_tools()
    
    async def close(self):
        """Close resources"""
        await self._http_toolbox.aclose()
    
    def _register_all_tools(self):
        """Register all available tools"""
        
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
            self._http_get
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
            self._http_post
        )
        
        self._register_tool(
            ToolDefinition(
                name="http_request",
                description="Make a custom HTTP request with any method",
                category=ToolCategory.HTTP,
                parameters={
                    "method": {"type": "string", "description": "HTTP method (GET, POST, PUT, DELETE, PATCH)"},
                    "url": {"type": "string", "description": "URL to request"},
                    "data": {"type": "object", "description": "Request body"},
                    "headers": {"type": "object", "description": "Headers"},
                },
                required_params=["method", "url"],
                returns="Response data",
            ),
            self._http_request
        )
        
        # ====================================================================
        # Database Tools
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
            self._db_query
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
            self._db_execute
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
            self._db_get_tables
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
            self._search_documents
        )
        
        self._register_tool(
            ToolDefinition(
                name="search_web",
                description="Search the web for information",
                category=ToolCategory.SEARCH,
                parameters={
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "description": "Number of results (default 5)"},
                },
                required_params=["query"],
                returns="Search results with titles and snippets",
            ),
            self._search_web
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
            self._calculate
        )
        
        self._register_tool(
            ToolDefinition(
                name="statistics",
                description="Calculate statistics for a list of numbers",
                category=ToolCategory.MATH,
                parameters={
                    "numbers": {"type": "array", "description": "List of numbers"},
                    "operations": {"type": "array", "description": "Stats to calculate: mean, median, std, min, max, sum"},
                },
                required_params=["numbers"],
                returns="Dictionary of statistics",
            ),
            self._statistics
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
            self._text_summarize
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
            self._text_extract
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
            self._json_parse
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
            self._json_format
        )
        
        self._register_tool(
            ToolDefinition(
                name="data_transform",
                description="Transform data using a mapping",
                category=ToolCategory.DATA,
                parameters={
                    "data": {"type": "object", "description": "Data to transform"},
                    "mapping": {"type": "object", "description": "Field mapping (new_field: old_field or expression)"},
                },
                required_params=["data", "mapping"],
                returns="Transformed data",
            ),
            self._data_transform
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
            self._call_external_api
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
            self._get_current_time
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
            self._file_read
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
            self._file_write
        )

        self._register_tool(
            ToolDefinition(
                name="file_list_dir",
                description="List directory entries under the agent file sandbox root",
                category=ToolCategory.FILE,
                parameters={
                    "path": {"type": "string", "description": "Directory relative to the sandbox root (default '.')"},
                },
                required_params=[],
                returns="List of directory entries",
            ),
            self._file_list_dir
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
            self._file_stat
        )

    def _register_tool(self, definition: ToolDefinition, implementation: Callable):
        """Register a tool with its implementation"""
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
        """HTTP GET request (diri-agent-toolbox AsyncHttpToolbox)."""
        r = await self._tool_runner.execute(
            "http_get", url=url, headers=headers, params=params
        )
        return _from_toolbox_result(r)
    
    async def _http_post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> ToolResult:
        """HTTP POST request (diri-agent-toolbox)."""
        r = await self._tool_runner.execute(
            "http_post", url=url, data=data, headers=headers
        )
        return _from_toolbox_result(r)
    
    async def _http_request(
        self,
        method: str,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> ToolResult:
        """Generic HTTP request (diri-agent-toolbox)."""
        r = await self._tool_runner.execute(
            "http_request", method=method, url=url, data=data, headers=headers
        )
        return _from_toolbox_result(r)
    
    async def _db_query(
        self,
        query: str,
        params: Optional[List[Any]] = None,
    ) -> ToolResult:
        """Database SELECT query"""
        start = datetime.utcnow()
        try:
            # Security check - only allow SELECT
            if not query.strip().upper().startswith("SELECT"):
                return ToolResult(success=False, error="Only SELECT queries are allowed")
            
            postgres = await get_postgres_manager()
            rows = await postgres.fetch(query, *(params or []))
            
            result = [dict(row) for row in rows]
            
            return ToolResult(
                success=True,
                result=result,
                execution_time_ms=(datetime.utcnow() - start).total_seconds() * 1000,
                metadata={"row_count": len(result)},
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _db_execute(
        self,
        query: str,
        params: Optional[List[Any]] = None,
    ) -> ToolResult:
        """Database write operation"""
        start = datetime.utcnow()
        try:
            postgres = await get_postgres_manager()
            result = await postgres.execute(query, *(params or []))
            
            return ToolResult(
                success=True,
                result=result,
                execution_time_ms=(datetime.utcnow() - start).total_seconds() * 1000,
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _db_get_tables(self) -> ToolResult:
        """List database tables"""
        try:
            postgres = await get_postgres_manager()
            rows = await postgres.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            
            tables = [row['table_name'] for row in rows]
            return ToolResult(success=True, result=tables)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _search_documents(
        self,
        query: str,
        limit: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Search documents in vector store"""
        try:
            api_bridge = await get_api_bridge()
            result = await api_bridge.call_tool("search_documents", {
                "query": query,
                "limit": limit,
                "filter": filter,
            })
            return ToolResult(success=True, result=result.result if hasattr(result, 'result') else result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _search_web(
        self,
        query: str,
        num_results: int = 5,
    ) -> ToolResult:
        """Web search (placeholder - implement with actual API)"""
        # This would integrate with a web search API like Serper, Bing, etc.
        return ToolResult(
            success=True,
            result={
                "message": "Web search not implemented - integrate with search API",
                "query": query,
            }
        )
    
    async def _calculate(self, expression: str) -> ToolResult:
        """Evaluate math expression (diri-agent-toolbox safe AST; not Cyrex eval)."""
        r = await self._tool_runner.execute("calculate", expression=expression)
        return _from_toolbox_result(r)
    
    async def _statistics(
        self,
        numbers: List[Union[int, float]],
        operations: Optional[List[str]] = None,
    ) -> ToolResult:
        """Calculate statistics (diri-agent-toolbox)."""
        r = await self._tool_runner.execute(
            "statistics", numbers=numbers, operations=operations
        )
        return _from_toolbox_result(r)
    
    async def _text_summarize(
        self,
        text: str,
        max_length: int = 200,
    ) -> ToolResult:
        """Summarize text (diri-agent-toolbox extractive helper)."""
        r = await self._tool_runner.execute(
            "text_summarize", text=text, max_length=max_length
        )
        return _from_toolbox_result(r)
    
    async def _text_extract(
        self,
        text: str,
        fields: List[str],
    ) -> ToolResult:
        """Extract fields from text (diri-agent-toolbox)."""
        r = await self._tool_runner.execute(
            "text_extract", text=text, fields=fields
        )
        return _from_toolbox_result(r)
    
    async def _json_parse(self, json_string: str) -> ToolResult:
        """Parse JSON string (diri-agent-toolbox)."""
        r = await self._tool_runner.execute("json_parse", json_string=json_string)
        return _from_toolbox_result(r)
    
    async def _json_format(
        self,
        data: Any,
        indent: int = 2,
    ) -> ToolResult:
        """Format as JSON (diri-agent-toolbox)."""
        r = await self._tool_runner.execute(
            "json_format", data=data, indent=indent
        )
        return _from_toolbox_result(r)
    
    async def _data_transform(
        self,
        data: Dict[str, Any],
        mapping: Dict[str, str],
    ) -> ToolResult:
        """Transform data using mapping (diri-agent-toolbox)."""
        r = await self._tool_runner.execute(
            "data_transform", data=data, mapping=mapping
        )
        return _from_toolbox_result(r)
    
    async def _call_external_api(
        self,
        api_name: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Call external API through bridge"""
        try:
            api_bridge = await get_api_bridge()
            result = await api_bridge.call_tool(f"{api_name}_{endpoint}", params or {})
            return ToolResult(success=True, result=result.result if hasattr(result, 'result') else result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _get_current_time(
        self,
        timezone: str = "UTC",
        format: str = "%Y-%m-%d %H:%M:%S",
    ) -> ToolResult:
        """Get current time (diri-agent-toolbox; respects IANA timezones)."""
        r = await self._tool_runner.execute(
            "current_time", timezone=timezone, format=format
        )
        return _from_toolbox_result(r)

    async def _file_read(self, path: str, encoding: str = "utf-8") -> ToolResult:
        """Read a sandboxed text file (diri-agent-toolbox SandboxedFileToolbox)."""
        r = await self._tool_runner.execute("file_read", path=path, encoding=encoding)
        return _from_toolbox_result(r)

    async def _file_write(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
    ) -> ToolResult:
        """Write a sandboxed text file (diri-agent-toolbox)."""
        r = await self._tool_runner.execute(
            "file_write", path=path, content=content, encoding=encoding
        )
        return _from_toolbox_result(r)

    async def _file_list_dir(self, path: str = ".") -> ToolResult:
        """List a sandboxed directory (diri-agent-toolbox)."""
        r = await self._tool_runner.execute("file_list_dir", path=path)
        return _from_toolbox_result(r)

    async def _file_stat(self, path: str) -> ToolResult:
        """Stat a sandboxed path (diri-agent-toolbox)."""
        r = await self._tool_runner.execute("file_stat", path=path)
        return _from_toolbox_result(r)

    # ========================================================================
    # Public API
    # ========================================================================
    
    def list_tools(self, category: Optional[ToolCategory] = None) -> List[ToolDefinition]:
        """List available tools"""
        tools = list(self._tool_registry.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get tool definition"""
        return self._tool_registry.get(name)
    
    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool"""
        if tool_name not in self._tool_implementations:
            return ToolResult(success=False, error=f"Tool not found: {tool_name}")
        
        implementation = self._tool_implementations[tool_name]
        
        try:
            result = await implementation(**kwargs)
            return result
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def format_for_prompt(self) -> str:
        """Format all tools for inclusion in prompt"""
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
    """Register all API tools with an agent"""
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
            tool_def.description
        )
    
    return api_tools

