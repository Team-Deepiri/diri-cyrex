"""
Agent State Processor
LangChain-style state processing for agent invocations
Processes current state and generates response based on LLM
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
import asyncio
from ..core.types import AgentConfig, AgentStatus, MemoryType
from ..core.memory_manager import get_memory_manager
from ..core.session_manager import get_session_manager
from ..core.prompt_templates import get_prompt_template_manager, PromptTemplate
from ..core.event_registry import get_event_registry
from ..core.event_handler import get_event_handler
from ..integrations.local_llm import LocalLLMProvider
from ..logging_config import get_logger

logger = get_logger("cyrex.agent_state_processor")


class StateKey(str, Enum):
    """Keys for agent state dictionary"""
    INPUT = "input"
    OUTPUT = "output"
    MESSAGES = "messages"
    MEMORY = "memory"
    CONTEXT = "context"
    TOOL_CALLS = "tool_calls"
    TOOL_RESULTS = "tool_results"
    METADATA = "metadata"
    SESSION_ID = "session_id"
    AGENT_ID = "agent_id"
    STATUS = "status"
    ITERATION = "iteration"
    ERROR = "error"


@dataclass
class AgentState:
    """Agent state structure for LangChain-style processing"""
    input: str = ""
    output: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    status: AgentStatus = AgentStatus.IDLE
    iteration: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["status"] = self.status.value if isinstance(self.status, Enum) else self.status
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentState':
        """Create from dictionary"""
        if "status" in data and isinstance(data["status"], str):
            data["status"] = AgentStatus(data["status"])
        return cls(**data)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the conversation"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if metadata:
            message["metadata"] = metadata
        self.messages.append(message)
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """Get messages in LLM format"""
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.messages
        ]


class AgentStateProcessor:
    """
    Processes agent state and generates responses using LLM
    Similar to LangChain's state processing approach
    """
    
    def __init__(
        self,
        agent_config: AgentConfig,
        llm_provider: LocalLLMProvider,
        session_id: Optional[str] = None,
    ):
        self.config = agent_config
        self.agent_id = agent_config.agent_id
        self.llm = llm_provider
        self.session_id = session_id
        self.logger = logger
        
        # Initialize components
        self._memory_manager = None
        self._session_manager = None
        self._event_handler = None
        self._event_registry = None
        self._prompt_manager = None
    
    async def _initialize_components(self):
        """Initialize async components"""
        if not self._memory_manager:
            self._memory_manager = await get_memory_manager()
        if not self._session_manager:
            self._session_manager = await get_session_manager()
        if not self._event_handler:
            self._event_handler = await get_event_handler()
        if not self._event_registry:
            self._event_registry = get_event_registry()
        if not self._prompt_manager:
            self._prompt_manager = get_prompt_template_manager()
    
    async def process(
        self,
        input_text: str,
        initial_state: Optional[AgentState] = None,
        max_iterations: int = 10,
        tools: Optional[Dict[str, Callable]] = None,
    ) -> AgentState:
        """
        Process input through state machine and generate response
        
        Args:
            input_text: User input
            initial_state: Optional initial state
            max_iterations: Maximum iterations for tool calling loop
            tools: Available tools for the agent
        
        Returns:
            Final agent state with response
        """
        await self._initialize_components()
        
        # Initialize state
        state = initial_state or AgentState(
            input=input_text,
            agent_id=self.agent_id,
            session_id=self.session_id,
            status=AgentStatus.PROCESSING,
        )
        
        # Emit event
        await self._event_handler.emit(
            event_type="agent.invoked",
            payload={
                "agent_id": self.agent_id,
                "input_text": input_text,
                "session_id": self.session_id,
            },
            source="agent_state_processor",
        )
        
        try:
            # Build initial context
            state.context = await self._build_context(state)
            
            # Add system message
            system_prompt = await self._build_system_prompt(state)
            state.add_message("system", system_prompt)
            
            # Add user input
            state.add_message("user", input_text)
            
            # Process through iterations (for tool calling)
            for iteration in range(max_iterations):
                state.iteration = iteration
                
                # Generate LLM response
                llm_response = await self._invoke_llm(state)
                
                # Parse response for tool calls
                tool_calls = self._parse_tool_calls(llm_response)
                
                if not tool_calls:
                    # No tool calls, final response
                    state.output = llm_response
                    state.add_message("assistant", llm_response)
                    break
                
                # Execute tools
                state.tool_calls.extend(tool_calls)
                tool_results = await self._execute_tools(tool_calls, tools or {})
                state.tool_results.extend(tool_results)
                
                # Add tool results to messages
                for result in tool_results:
                    state.add_message("tool", json.dumps(result))
                
                # Continue loop to get final response with tool results
            
            # Final response if we have tool results
            if state.tool_calls and not state.output:
                final_response = await self._invoke_llm(state)
                state.output = final_response
                state.add_message("assistant", final_response)
            
            # Store in memory
            await self._store_interaction(state)
            
            # Update status
            state.status = AgentStatus.COMPLETED
            
            # Emit completion event
            await self._event_handler.emit(
                event_type="agent.response",
                payload={
                    "agent_id": self.agent_id,
                    "response_content": state.output,
                    "tool_calls": state.tool_calls,
                    "iteration": state.iteration,
                },
                source="agent_state_processor",
            )
            
            return state
        
        except Exception as e:
            self.logger.error(f"State processing failed: {e}", exc_info=True)
            state.status = AgentStatus.ERROR
            state.error = str(e)
            
            # Emit error event
            await self._event_handler.emit(
                event_type="error.occurred",
                payload={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "component": "agent_state_processor",
                    "agent_id": self.agent_id,
                },
                source="agent_state_processor",
                metadata={"priority": "high"},
            )
            
            return state
    
    async def _build_context(self, state: AgentState) -> Dict[str, Any]:
        """Build context from memory and session"""
        context = {}
        
        # Get session context
        if self.session_id and self._session_manager:
            session = await self._session_manager.get_session(self.session_id)
            if session:
                context.update(session.context)
        
        # Get relevant memories
        if self._memory_manager:
            memories = await self._memory_manager.build_context(
                session_id=self.session_id,
                query=state.input,
            )
            context["memories"] = memories
            state.memory = memories
        
        return context
    
    async def _build_system_prompt(self, state: AgentState) -> str:
        """Build system prompt from template"""
        # Try to get template from prompt manager
        template = self._prompt_manager.get_template("conversation")
        
        if template:
            variables = {
                "agent_name": self.config.name,
                "personality": "Professional and helpful",
                "user_preferences": json.dumps(state.context.get("user_preferences", {})),
                "conversation_history": self._format_conversation_history(state.messages[:-1]),
                "user_message": state.input,
            }
            rendered = template.render(variables)
            return rendered["system"]
        
        # Fallback to default
        return f"""You are {self.config.name}, a {self.config.role.value} agent.

Your capabilities: {', '.join(self.config.capabilities)}

Guidelines:
- Be helpful, accurate, and concise
- Use tools when appropriate
- Remember context from previous interactions
- Follow safety guidelines
"""
    
    def _format_conversation_history(self, messages: List[Dict[str, Any]]) -> str:
        """Format conversation history for prompt"""
        if not messages:
            return "No previous context"
        
        formatted = []
        for msg in messages[-5:]:  # Last 5 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted)
    
    async def _invoke_llm(self, state: AgentState) -> str:
        """Invoke LLM with current state"""
        # Build messages for LLM
        messages = state.get_messages_for_llm()
        
        # Add tool descriptions if tools are available
        if state.tool_calls or state.context.get("tools"):
            tool_descriptions = self._format_tool_descriptions(state)
            if tool_descriptions:
                messages[-1]["content"] += f"\n\nAvailable tools:\n{tool_descriptions}"
        
        # Convert to prompt string (for simple LLM interface)
        prompt = self._messages_to_prompt(messages)
        
        # Invoke LLM
        response = await asyncio.to_thread(self.llm.invoke, prompt)
        
        return response
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to prompt string"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "tool":
                prompt_parts.append(f"Tool Result: {content}")
        
        return "\n\n".join(prompt_parts)
    
    def _format_tool_descriptions(self, state: AgentState) -> str:
        """Format tool descriptions for prompt"""
        tools = state.context.get("tools", {})
        if not tools:
            return ""
        
        descriptions = []
        for tool_name, tool_func in tools.items():
            desc = getattr(tool_func, "__doc__", f"Tool: {tool_name}")
            descriptions.append(f"- {tool_name}: {desc}")
        
        return "\n".join(descriptions)
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from LLM response"""
        tool_calls = []
        import re
        
        # Look for [TOOL:name:params] pattern
        pattern = r'\[TOOL:([^:]+):([^\]]+)\]'
        matches = re.findall(pattern, response)
        
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
    
    async def _execute_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        tools: Dict[str, Callable]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls"""
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool")
            parameters = tool_call.get("parameters", {})
            
            # Emit tool called event
            await self._event_handler.emit(
                event_type="tool.called",
                payload={
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "agent_id": self.agent_id,
                },
                source="agent_state_processor",
            )
            
            if tool_name not in tools:
                result = {"tool": tool_name, "error": f"Tool not found: {tool_name}"}
                results.append(result)
                continue
            
            tool_func = tools[tool_name]
            start_time = datetime.utcnow()
            
            try:
                if asyncio.iscoroutinefunction(tool_func):
                    tool_result = await tool_func(**parameters)
                else:
                    tool_result = tool_func(**parameters)
                
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                result = {
                    "tool": tool_name,
                    "result": tool_result,
                    "execution_time_ms": execution_time,
                }
                
                # Emit tool completed event
                await self._event_handler.emit(
                    event_type="tool.completed",
                    payload=result,
                    source="agent_state_processor",
                )
                
            except Exception as e:
                self.logger.error(f"Tool execution failed: {e}", exc_info=True)
                result = {
                    "tool": tool_name,
                    "error": str(e),
                    "execution_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                }
            
            results.append(result)
        
        return results
    
    async def _store_interaction(self, state: AgentState):
        """Store interaction in memory"""
        if not self._memory_manager:
            return
        
        # Store as episodic memory
        await self._memory_manager.store_memory(
            content=f"User: {state.input}\nAgent: {state.output}",
            memory_type=MemoryType.EPISODIC,
            session_id=self.session_id,
            importance=0.7,
            metadata={
                "agent_id": self.agent_id,
                "tool_calls": state.tool_calls,
                "iteration": state.iteration,
            },
        )

