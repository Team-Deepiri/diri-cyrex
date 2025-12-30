"""
API Tools for Agents
Tools that call external APIs via the API bridge
"""
from typing import Dict, Any, Optional
from ...integrations.api_bridge import get_api_bridge
from ...logging_config import get_logger

logger = get_logger("cyrex.agent.tools.api")


async def register_api_tools(agent):
    """Register API tools with an agent"""
    api_bridge = await get_api_bridge()
    
    async def call_api(tool_name: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call an external API tool"""
        try:
            tool_call = await api_bridge.call_tool(tool_name, parameters or {})
            return tool_call.result if tool_call.status == "completed" else {"error": tool_call.error}
        except Exception as e:
            logger.error(f"API tool call failed: {e}")
            return {"error": str(e)}
    
    agent.register_tool("call_api", call_api, "Call an external API tool by name")
    
    # Register specific API tools if available
    tools = await api_bridge.list_tools()
    for tool in tools:
        tool_name = tool["name"]
        async def make_tool_func(name):
            async def tool_func(**kwargs):
                return await call_api(name, kwargs)
            return tool_func
        
        agent.register_tool(
            f"api_{tool_name}",
            await make_tool_func(tool_name),
            f"Call {tool_name} API: {tool.get('description', '')}"
        )

