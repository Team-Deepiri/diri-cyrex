"""
API Bridge / Tools System
External API integration system with tool calling and request management
"""
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import httpx
import asyncio
from ..core.types import ToolCall
from ..logging_config import get_logger
import json

logger = get_logger("cyrex.api_bridge")


class APIBridge:
    """
    Manages external API integrations and tool calls
    Handles authentication, rate limiting, retries, and response caching
    """
    
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._clients: Dict[str, httpx.AsyncClient] = {}
        self._rate_limits: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self.logger = logger
    
    async def register_tool(
        self,
        tool_name: str,
        api_endpoint: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[Dict[str, str]] = None,
        rate_limit: Optional[Dict[str, int]] = None,
        timeout: int = 30,
        description: str = "",
    ):
        """Register an API tool"""
        async with self._lock:
            self._tools[tool_name] = {
                "name": tool_name,
                "api_endpoint": api_endpoint,
                "method": method.upper(),
                "headers": headers or {},
                "auth": auth or {},
                "rate_limit": rate_limit or {"requests": 100, "window": 60},
                "timeout": timeout,
                "description": description,
            }
            
            # Create HTTP client for this tool
            client_headers = headers.copy() if headers else {}
            if auth:
                if "type" in auth and auth["type"] == "bearer":
                    client_headers["Authorization"] = f"Bearer {auth.get('token', '')}"
                elif "type" in auth and auth["type"] == "api_key":
                    client_headers[auth.get("header", "X-API-Key")] = auth.get("key", "")
            
            self._clients[tool_name] = httpx.AsyncClient(
                headers=client_headers,
                timeout=timeout,
            )
            
            # Initialize rate limit tracking
            self._rate_limits[tool_name] = {
                "requests": [],
                "limit": rate_limit.get("requests", 100) if rate_limit else 100,
                "window": rate_limit.get("window", 60) if rate_limit else 60,
            }
            
            self.logger.info(f"API tool registered: {tool_name}", endpoint=api_endpoint)
    
    async def call_tool(
        self,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> ToolCall:
        """Call an API tool"""
        if tool_name not in self._tools:
            raise ValueError(f"Tool not registered: {tool_name}")
        
        tool_config = self._tools[tool_name]
        tool_call = ToolCall(
            tool_name=tool_name,
            api_endpoint=tool_config["api_endpoint"],
            method=tool_config["method"],
            parameters=parameters or {},
            headers=headers or {},
            timeout=tool_config["timeout"],
        )
        
        # Check rate limit
        await self._check_rate_limit(tool_name)
        
        try:
            client = self._clients[tool_name]
            method = tool_config["method"]
            url = tool_config["api_endpoint"]
            
            # Merge parameters into URL or body
            if method == "GET":
                # Add parameters as query params
                response = await client.get(url, params=parameters)
            elif method == "POST":
                response = await client.post(url, json=parameters, headers=headers)
            elif method == "PUT":
                response = await client.put(url, json=parameters, headers=headers)
            elif method == "DELETE":
                response = await client.delete(url, params=parameters, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Update rate limit tracking
            await self._update_rate_limit(tool_name)
            
            tool_call.status = "completed"
            tool_call.completed_at = datetime.utcnow()
            
            # Parse response
            try:
                tool_call.result = response.json()
            except:
                tool_call.result = response.text
            
            self.logger.info(
                f"Tool call completed: {tool_name}",
                status_code=response.status_code,
                tool_id=tool_call.tool_id
            )
            
        except httpx.TimeoutException:
            tool_call.status = "timeout"
            tool_call.error = "Request timed out"
            self.logger.warning(f"Tool call timeout: {tool_name}")
        except Exception as e:
            tool_call.status = "error"
            tool_call.error = str(e)
            self.logger.error(f"Tool call failed: {tool_name}", error=str(e))
        
        return tool_call
    
    async def _check_rate_limit(self, tool_name: str):
        """Check and enforce rate limits"""
        if tool_name not in self._rate_limits:
            return
        
        rate_limit = self._rate_limits[tool_name]
        now = datetime.utcnow()
        window_start = now.timestamp() - rate_limit["window"]
        
        # Remove old requests outside the window
        rate_limit["requests"] = [
            ts for ts in rate_limit["requests"]
            if ts > window_start
        ]
        
        # Check if limit exceeded
        if len(rate_limit["requests"]) >= rate_limit["limit"]:
            # Calculate wait time
            oldest_request = min(rate_limit["requests"])
            wait_time = rate_limit["window"] - (now.timestamp() - oldest_request) + 1
            if wait_time > 0:
                self.logger.warning(f"Rate limit reached for {tool_name}, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                # Retry after wait
                await self._check_rate_limit(tool_name)
    
    async def _update_rate_limit(self, tool_name: str):
        """Update rate limit tracking"""
        if tool_name in self._rate_limits:
            self._rate_limits[tool_name]["requests"].append(datetime.utcnow().timestamp())
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools"""
        return [
            {
                "name": tool["name"],
                "endpoint": tool["api_endpoint"],
                "method": tool["method"],
                "description": tool["description"],
            }
            for tool in self._tools.values()
        ]
    
    async def close(self):
        """Close all HTTP clients"""
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()


# Global API bridge instance
_api_bridge: Optional[APIBridge] = None


async def get_api_bridge() -> APIBridge:
    """Get or create API bridge singleton"""
    global _api_bridge
    if _api_bridge is None:
        _api_bridge = APIBridge()
    return _api_bridge

