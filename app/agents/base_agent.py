"""
Base Agent Class
Foundation for all Cyrex agents with invoke methods, prompt, and tool support
"""
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
from datetime import datetime
import asyncio
from ..core.types import AgentConfig, AgentRole, AgentStatus, MemoryType
from ..core.memory_manager import get_memory_manager
from ..core.session_manager import get_session_manager
from ..core.enhanced_guardrails import get_enhanced_guardrails
from ..integrations.local_llm import LocalLLMProvider, get_local_llm
from ..integrations.api_bridge import get_api_bridge
from ..logging_config import get_logger
import json

logger = get_logger("cyrex.agent.base")


class AgentResponse:
    """Agent response structure"""
    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        confidence: float = 1.0,
    ):
        self.content = content
        self.metadata = metadata or {}
        self.tool_calls = tool_calls or []
        self.confidence = confidence
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "tool_calls": self.tool_calls,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseAgent(ABC):
    """
    Base agent class with invoke methods, prompt management, and tool integration
    All agents inherit from this class
    """
    
    def __init__(
        self,
        agent_config: AgentConfig,
        llm_provider: Optional[LocalLLMProvider] = None,
        session_id: Optional[str] = None,
    ):
        self.config = agent_config
        self.agent_id = agent_config.agent_id
        self.role = agent_config.role
        self.name = agent_config.name
        self.session_id = session_id
        self.status = AgentStatus.IDLE
        self.logger = logger
        
        # Initialize LLM provider
        if llm_provider:
            self.llm = llm_provider
        else:
            # Initialize with Ollama by default
            self.llm = get_local_llm(
                backend="ollama",
                model_name=agent_config.model_config.get("model", "llama3:8b"),
                temperature=agent_config.temperature,
                max_tokens=agent_config.max_tokens,
            )
        
        # Initialize components
        self._memory_manager = None
        self._session_manager = None
        self._guardrails = None
        self._api_bridge = None
        
        # Prompt template (can be overridden by subclasses)
        self.prompt_template = agent_config.system_prompt or self._default_prompt_template()
        
        # Tool registry
        self._tools: Dict[str, Callable] = {}
        self._register_default_tools()
    
    async def _initialize_components(self):
        """Initialize async components"""
        if not self._memory_manager:
            self._memory_manager = await get_memory_manager()
        if not self._session_manager:
            self._session_manager = await get_session_manager()
        if not self._guardrails:
            self._guardrails = await get_enhanced_guardrails()
        if not self._api_bridge:
            self._api_bridge = await get_api_bridge()
    
    def _default_prompt_template(self) -> str:
        """Default prompt template"""
        return f"""You are {self.name}, a {self.role.value} agent.

Your capabilities: {', '.join(self.config.capabilities)}

Instructions:
- Be helpful, accurate, and concise
- Use tools when appropriate
- Remember context from previous interactions
- Follow safety guidelines

Current task: {{task}}
Context: {{context}}
"""
    
    def _register_default_tools(self):
        """Register default tools available to all agents"""
        self._tools["search_memory"] = self._tool_search_memory
        self._tools["store_memory"] = self._tool_store_memory
        self._tools["get_context"] = self._tool_get_context
    
    async def invoke(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
        use_tools: bool = True,
    ) -> AgentResponse:
        """
        Main invoke method - processes input and returns response
        
        Args:
            input_text: User input or task description
            context: Additional context dictionary
            use_tools: Whether to allow tool usage
        
        Returns:
            AgentResponse with content, metadata, and tool calls
        """
        await self._initialize_components()
        
        self.status = AgentStatus.PROCESSING
        
        try:
            # Build context from memory
            full_context = await self._build_context(context or {}, query=input_text)
            
            # Safety check
            guardrail_result = await self._guardrails.check(input_text, full_context)
            if not guardrail_result["safe"]:
                return AgentResponse(
                    content="I cannot process this request due to safety guidelines.",
                    metadata={"guardrail_action": guardrail_result["action"]},
                    confidence=0.0,
                )
            
            # Build prompt
            prompt = self._build_prompt(input_text, full_context)
            
            # Invoke LLM
            if use_tools and self._tools:
                response = await self._invoke_with_tools(prompt, full_context)
            else:
                response = await self._invoke_simple(prompt)
            
            # Store interaction in memory
            await self._store_interaction(input_text, response.content, full_context)
            
            self.status = AgentStatus.COMPLETED
            return response
            
        except Exception as e:
            self.logger.error(f"Agent invoke failed: {e}", exc_info=True)
            self.status = AgentStatus.ERROR
            return AgentResponse(
                content=f"An error occurred: {str(e)}",
                metadata={"error": str(e)},
                confidence=0.0,
            )
        finally:
            self.status = AgentStatus.IDLE
    
    async def _invoke_simple(self, prompt: str) -> AgentResponse:
        """Simple invoke without tools"""
        if not self.llm:
            raise RuntimeError("LLM provider not initialized")
        
        response_text = await asyncio.to_thread(
            self.llm.invoke,
            prompt
        )
        
        return AgentResponse(
            content=response_text,
            metadata={"method": "simple"},
            confidence=0.8,
        )
    
    async def _invoke_with_tools(self, prompt: str, context: Dict[str, Any]) -> AgentResponse:
        """Invoke with tool support"""
        if not self.llm:
            raise RuntimeError("LLM provider not initialized")
        
        # Enhanced prompt with tool descriptions
        tool_descriptions = self._format_tool_descriptions()
        enhanced_prompt = f"""{prompt}

Available tools:
{tool_descriptions}

You can use tools by including [TOOL:tool_name:parameters_json] in your response.
"""
        
        response_text = await asyncio.to_thread(
            self.llm.invoke,
            enhanced_prompt
        )
        
        # Parse tool calls from response
        tool_calls = self._parse_tool_calls(response_text)
        
        # Execute tool calls
        tool_results = []
        for tool_call in tool_calls:
            try:
                result = await self._execute_tool(tool_call)
                tool_results.append(result)
            except Exception as e:
                self.logger.error(f"Tool execution failed: {e}")
                tool_results.append({"error": str(e)})
        
        # If tools were called, potentially get updated response
        if tool_calls:
            followup_prompt = f"""{enhanced_prompt}

Tool results:
{json.dumps(tool_results, indent=2)}

Provide a final response incorporating the tool results.
"""
            final_response = await asyncio.to_thread(
                self.llm.invoke,
                followup_prompt
            )
            response_text = final_response
        
        return AgentResponse(
            content=response_text,
            metadata={"method": "with_tools", "tool_calls": len(tool_calls)},
            tool_calls=tool_calls,
            confidence=0.9 if tool_calls else 0.8,
        )
    
    def _build_prompt(self, input_text: str, context: Dict[str, Any]) -> str:
        """Build prompt from template"""
        return self.prompt_template.format(
            task=input_text,
            context=json.dumps(context, indent=2),
            agent_name=self.name,
            role=self.role.value,
        )
    
    async def _build_context(self, additional_context: Dict[str, Any], query: Optional[str] = None) -> Dict[str, Any]:
        """Build context from memory and session"""
        context = additional_context.copy()
        
        # Get session context
        if self.session_id:
            session = await self._session_manager.get_session(self.session_id)
            if session:
                context.update(session.context)
        
        # Get relevant memories
        if self._memory_manager:
            memories = await self._memory_manager.build_context(
                session_id=self.session_id,
                query=query,
            )
            context["memories"] = memories
        
        return context
    
    async def _store_interaction(self, input_text: str, output_text: str, context: Dict[str, Any]):
        """Store interaction in memory"""
        if not self._memory_manager:
            return
        
        # Store as episodic memory
        await self._memory_manager.store_memory(
            content=f"User: {input_text}\nAgent: {output_text}",
            memory_type=MemoryType.EPISODIC,
            session_id=self.session_id,
            importance=0.7,
            metadata={"agent_id": self.agent_id, "role": self.role.value},
        )
    
    def _format_tool_descriptions(self) -> str:
        """Format tool descriptions for prompt"""
        descriptions = []
        for tool_name, tool_func in self._tools.items():
            descriptions.append(f"- {tool_name}: {tool_func.__doc__ or 'No description'}")
        return "\n".join(descriptions)
    
    def _parse_tool_calls(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse tool calls from response text"""
        tool_calls = []
        import re
        
        # Look for [TOOL:name:params] pattern
        pattern = r'\[TOOL:([^:]+):([^\]]+)\]'
        matches = re.findall(pattern, response_text)
        
        for tool_name, params_str in matches:
            try:
                params = json.loads(params_str)
                tool_calls.append({
                    "tool": tool_name,
                    "parameters": params,
                })
            except json.JSONDecodeError:
                self.logger.warning(f"Invalid tool call parameters: {params_str}")
        
        return tool_calls
    
    async def _execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call"""
        tool_name = tool_call.get("tool")
        parameters = tool_call.get("parameters", {})
        
        if tool_name not in self._tools:
            return {"error": f"Tool not found: {tool_name}"}
        
        tool_func = self._tools[tool_name]
        
        try:
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**parameters)
            else:
                result = tool_func(**parameters)
            return {"tool": tool_name, "result": result}
        except Exception as e:
            return {"tool": tool_name, "error": str(e)}
    
    # Default tool implementations
    async def _tool_search_memory(self, query: str, limit: int = 5) -> List[str]:
        """Search memory for relevant information"""
        if not self._memory_manager:
            return []
        memories = await self._memory_manager.search_memories(
            query=query,
            session_id=self.session_id,
            limit=limit,
        )
        return [m.content for m in memories]
    
    async def _tool_store_memory(self, content: str, importance: float = 0.5) -> str:
        """Store information in memory"""
        if not self._memory_manager:
            return "Memory manager not available"
        memory_id = await self._memory_manager.store_memory(
            content=content,
            memory_type=MemoryType.LONG_TERM,
            session_id=self.session_id,
            importance=importance,
        )
        return f"Stored memory: {memory_id}"
    
    async def _tool_get_context(self) -> Dict[str, Any]:
        """Get current context"""
        return await self._build_context({})
    
    def register_tool(self, name: str, tool_func: Callable, description: str = ""):
        """Register a custom tool"""
        tool_func.__doc__ = description or tool_func.__doc__
        self._tools[name] = tool_func
        self.logger.debug(f"Tool registered: {name}")
    
    @abstractmethod
    async def process(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task - must be implemented by subclasses"""
        pass

