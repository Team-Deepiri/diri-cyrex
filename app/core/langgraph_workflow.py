"""
LangGraph Multi-Agent Workflow
Comprehensive graph-based agent routing for the entire diri-cyrex app
Supports all agent roles, Language Intelligence, Cyrex Guard, and custom workflows
"""
from typing import TypedDict, Annotated, Sequence, Optional, Dict, Any, List, Union
from datetime import datetime
import operator
import asyncio
import os
import re
from ..logging_config import get_logger
from ..core.types import AgentRole, MemoryType, AgentStatus
from ..agents.agent_factory import AgentFactory
from ..agents.base_agent import AgentResponse
from ..settings import settings

logger = get_logger("cyrex.langgraph.workflow")

# LangGraph imports with graceful fallback
HAS_LANGGRAPH = False
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.graph.message import add_messages
    from langgraph.checkpoint.redis import RedisSaver
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    HAS_LANGGRAPH = True
except ImportError as e:
    logger.warning(f"LangGraph not available: {e}, using fallback implementation")
    StateGraph = None
    END = None
    START = None
    RedisSaver = None
    # Create dummy classes for type hints
    class BaseMessage:
        pass
    class HumanMessage:
        def __init__(self, content):
            self.content = content
    class AIMessage:
        def __init__(self, content):
            self.content = content
    class SystemMessage:
        def __init__(self, content):
            self.content = content

# Redis client for checkpointing
try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    logger.warning("Redis async client not available")
    HAS_REDIS = False
    redis = None

# Lazy imports for optional components
_memory_manager = None
_session_manager = None
_monitor = None
_rag_pipeline = None
_event_registry = None
_state_processor = None

async def _get_memory_manager():
    """Lazy load memory manager"""
    global _memory_manager
    if _memory_manager is None:
        try:
            from ..core.memory_manager import get_memory_manager
            _memory_manager = await get_memory_manager()
        except Exception as e:
            logger.debug(f"Memory manager not available: {e}")
    return _memory_manager

async def _get_session_manager():
    """Lazy load session manager"""
    global _session_manager
    if _session_manager is None:
        try:
            from ..core.session_manager import get_session_manager
            _session_manager = await get_session_manager()
        except Exception as e:
            logger.debug(f"Session manager not available: {e}")
    return _session_manager

async def _get_monitor():
    """Lazy load system monitor"""
    global _monitor
    if _monitor is None:
        try:
            from ..core.monitoring import get_monitor
            _monitor = get_monitor()
        except Exception as e:
            logger.debug(f"Monitor not available: {e}")
    return _monitor

async def _get_rag_pipeline():
    """Lazy load RAG pipeline"""
    global _rag_pipeline
    if _rag_pipeline is None:
        try:
            from ..integrations.rag_pipeline import RAGPipeline
            _rag_pipeline = RAGPipeline()
        except Exception as e:
            logger.debug(f"RAG pipeline not available: {e}")
    return _rag_pipeline

async def _get_event_registry():
    """Lazy load event registry"""
    global _event_registry
    if _event_registry is None:
        try:
            from ..core.event_registry import get_event_registry
            _event_registry = await get_event_registry()
        except Exception as e:
            logger.debug(f"Event registry not available: {e}")
    return _event_registry

async def _get_state_processor():
    """Lazy load state processor"""
    global _state_processor
    if _state_processor is None:
        try:
            from ..core.agent_state_processor import AgentStateProcessor
            _state_processor = AgentStateProcessor()
        except Exception as e:
            logger.debug(f"State processor not available: {e}")
    return _state_processor


# State definition for LangGraph workflow
if HAS_LANGGRAPH:
    class AgentState(TypedDict):
        """State structure for multi-agent workflow"""
        messages: Annotated[Sequence[BaseMessage], operator.add]
        current_agent: str
        task_description: str
        task_type: Optional[str]  # Detected task type (lease, contract, fraud, code, etc.)
        plan: Optional[str]
        code: Optional[str]
        quality_check: Optional[str]
        result: Optional[Dict[str, Any]]  # Final result (lease abstraction, contract analysis, etc.)
        context: Dict[str, Any]
        workflow_id: str
        workflow_type: str  # workflow type (standard, lease, contract, fraud, etc.)
        session_id: Optional[str]
        user_id: Optional[str]
        metadata: Dict[str, Any]
        errors: List[str]
        tool_calls: List[Dict[str, Any]]
        agent_history: List[Dict[str, Any]]  # History of agents used
else:
    # Fallback TypedDict without Annotated for type checking
    class AgentState(TypedDict):
        """State structure for multi-agent workflow"""
        messages: List[Any]
        current_agent: str
        task_description: str
        task_type: Optional[str]
        plan: Optional[str]
        code: Optional[str]
        quality_check: Optional[str]
        result: Optional[Dict[str, Any]]
        context: Dict[str, Any]
        workflow_id: str
        workflow_type: str
        session_id: Optional[str]
        user_id: Optional[str]
        metadata: Dict[str, Any]
        errors: List[str]
        tool_calls: List[Dict[str, Any]]
        agent_history: List[Dict[str, Any]]


class LangGraphMultiAgentWorkflow:
    """
    LangGraph-based multi-agent workflow
    Comprehensive workflow supporting all diri-cyrex agents and services:
    - Standard workflow: task → plan → code → quality
    - Language Intelligence: lease abstraction, contract intelligence
    - Cyrex Guard: vendor fraud detection
    - Custom workflows based on task detection
    """
    
    def __init__(
        self,
        llm_provider=None,
        tool_registry=None,
        vector_store=None,
        redis_client=None,
        memory_manager=None,
        session_manager=None,
        monitor=None,
        rag_pipeline=None,
        event_registry=None,
        state_processor=None,
    ):
        self.llm_provider = llm_provider
        self.tool_registry = tool_registry
        self.vector_store = vector_store
        self.redis_client = redis_client
        self.memory_manager = memory_manager
        self.session_manager = session_manager
        self.monitor = monitor
        self.rag_pipeline = rag_pipeline
        self.event_registry = event_registry
        self.state_processor = state_processor
        self.logger = logger
        self.graph = None
        self.checkpointer = None
        
        # Agent cache (lazy loading)
        self._agents: Dict[str, Any] = {}
        
        # Model configuration from settings
        self.default_model = getattr(settings, "LOCAL_LLM_MODEL", "llama3:8b")
        self.default_temperature = getattr(settings, "AI_TEMPERATURE", 0.7)
        self.default_max_tokens = getattr(settings, "AI_MAX_TOKENS", 2000)
        
        if HAS_LANGGRAPH:
            self._build_graph()
        else:
            logger.warning("LangGraph not available, workflow will use fallback mode")
    
    async def _get_agent(self, role: AgentRole, **kwargs) -> Any:
        """Get or create agent by role"""
        role_key = role.value
        if role_key not in self._agents:
            self._agents[role_key] = await AgentFactory.create_agent(
                role=role,
                model_name=kwargs.get("model_name", self.default_model),
                temperature=kwargs.get("temperature", self.default_temperature),
                max_tokens=kwargs.get("max_tokens", self.default_max_tokens),
                **{k: v for k, v in kwargs.items() if k not in ["model_name", "temperature", "max_tokens"]}
            )
        return self._agents[role_key]
    
    def _detect_task_type(self, task_description: str) -> str:
        """Detect task type from description"""
        task_lower = task_description.lower()
        
        # Language Intelligence tasks
        if any(keyword in task_lower for keyword in ["lease", "rental", "tenant", "landlord", "property"]):
            return "lease"
        if any(keyword in task_lower for keyword in ["contract", "agreement", "clause", "obligation"]):
            return "contract"
        
        # Cyrex Guard tasks
        if any(keyword in task_lower for keyword in ["invoice", "vendor", "fraud", "pricing", "benchmark"]):
            return "fraud"
        
        # Code generation tasks
        if any(keyword in task_lower for keyword in ["code", "program", "function", "script", "implement"]):
            return "code"
        
        # Standard workflow
        return "standard"
    
    def _build_graph(self):
        """Build the LangGraph state machine with dynamic routing"""
        if not HAS_LANGGRAPH:
            return
        
        try:
            workflow = StateGraph(AgentState)
            
            # Add all agent nodes
            workflow.add_node("task_router", self._task_router_node)
            workflow.add_node("task_agent", self._task_agent_node)
            workflow.add_node("plan_agent", self._plan_agent_node)
            workflow.add_node("code_agent", self._code_agent_node)
            workflow.add_node("qa_agent", self._qa_agent_node)
            
            # Language Intelligence nodes
            workflow.add_node("lease_processor", self._lease_processor_node)
            workflow.add_node("contract_processor", self._contract_processor_node)
            
            # Cyrex Guard nodes
            workflow.add_node("fraud_agent", self._fraud_agent_node)
            
            # Set entry point
            workflow.set_entry_point("task_router")
            
            # Task router routes to appropriate workflow
            workflow.add_conditional_edges(
                "task_router",
                self._route_from_task_router,
                {
                    "standard": "task_agent",
                    "lease": "lease_processor",
                    "contract": "contract_processor",
                    "fraud": "fraud_agent",
                    "end": END,
                }
            )
            
            # Standard workflow edges
            workflow.add_conditional_edges(
                "task_agent",
                self._should_continue_after_task,
                {
                    "plan_agent": "plan_agent",
                    "end": END,
                }
            )
            
            workflow.add_conditional_edges(
                "plan_agent",
                self._should_continue_after_plan,
                {
                    "code_agent": "code_agent",
                    "end": END,
                }
            )
            
            workflow.add_conditional_edges(
                "code_agent",
                self._should_continue_after_code,
                {
                    "qa_agent": "qa_agent",
                    "end": END,
                }
            )
            
            # All specialized nodes end
            workflow.add_edge("lease_processor", END)
            workflow.add_edge("contract_processor", END)
            workflow.add_edge("fraud_agent", END)
            workflow.add_edge("qa_agent", END)
            
            # Compile with checkpointing if Redis available
            if HAS_REDIS and self.redis_client:
                try:
                    self.checkpointer = RedisSaver(redis_client=self.redis_client)
                    self.graph = workflow.compile(checkpointer=self.checkpointer)
                    logger.info("LangGraph workflow compiled with Redis checkpointing")
                except Exception as e:
                    logger.warning(f"Redis checkpointing failed: {e}, compiling without checkpointing")
                    self.graph = workflow.compile()
            else:
                self.graph = workflow.compile()
                logger.info("LangGraph workflow compiled without checkpointing")
                
        except Exception as e:
            logger.error(f"Failed to build LangGraph workflow: {e}", exc_info=True)
            self.graph = None
    
    async def _task_router_node(self, state: AgentState) -> AgentState:
        """Task router node - detects task type and routes accordingly"""
        workflow_id = state.get("workflow_id", "unknown")
        
        try:
            task_description = state.get("task_description", "")
            if not task_description and state.get("messages"):
                last_message = state["messages"][-1]
                if hasattr(last_message, 'content'):
                    task_description = last_message.content
                elif isinstance(last_message, dict):
                    task_description = last_message.get("content", "")
            
            if not task_description:
                task_description = "No task provided"
            
            # Detect task type
            task_type = self._detect_task_type(task_description)
            workflow_type = task_type
            
            state["task_description"] = task_description
            state["task_type"] = task_type
            state["workflow_type"] = workflow_type
            state["current_agent"] = "task_router"
            
            # Use RAG for context if available
            rag = self.rag_pipeline or await _get_rag_pipeline()
            if rag:
                try:
                    similar_tasks = rag.retrieve(task_description, top_k=3)
                    if similar_tasks:
                        state["metadata"]["rag_context"] = [task.get("challenge_text", "") for task in similar_tasks]
                except Exception as e:
                    logger.debug(f"RAG retrieval failed: {e}")
            
            # Publish event
            event_reg = self.event_registry or await _get_event_registry()
            if event_reg:
                try:
                    await event_reg.publish_event(
                        event_type="workflow.started",
                        payload={
                            "workflow_id": workflow_id,
                            "task_type": task_type,
                            "workflow_type": workflow_type,
                        }
                    )
                except Exception:
                    pass
            
            self.logger.info(f"Task routed to {task_type} workflow", workflow_id=workflow_id, task_type=task_type)
            return state
            
        except Exception as e:
            self.logger.error(f"Task router failed: {e}", exc_info=True, workflow_id=workflow_id)
            state["errors"].append(f"Task routing error: {str(e)}")
            state["workflow_type"] = "standard"  # Fallback to standard
            return state
    
    def _route_from_task_router(self, state: AgentState) -> str:
        """Route from task router based on detected task type"""
        task_type = state.get("task_type", "standard")
        return task_type if task_type in ["standard", "lease", "contract", "fraud"] else "standard"
    
    async def _task_agent_node(self, state: AgentState) -> AgentState:
        """Task decomposition agent node"""
        workflow_id = state.get("workflow_id", "unknown")
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Task agent processing", workflow_id=workflow_id)
            
            task_description = state.get("task_description", "")
            
            # Store task in memory
            memory_mgr = self.memory_manager or await _get_memory_manager()
            if memory_mgr:
                try:
                    await memory_mgr.store_memory(
                        content=f"Task: {task_description}",
                        memory_type=MemoryType.EPISODIC,
                        session_id=state.get("session_id"),
                        user_id=state.get("user_id"),
                        importance=0.8,
                        metadata={"workflow_id": workflow_id, "agent": "task_agent"},
                    )
                except Exception:
                    pass
            
            # Get task agent
            task_agent = await self._get_agent(AgentRole.TASK_DECOMPOSER)
            
            # Build context
            context = state.get("context", {}).copy()
            context.update({
                "workflow_id": workflow_id,
                "session_id": state.get("session_id"),
                "user_id": state.get("user_id"),
                "rag_context": state.get("metadata", {}).get("rag_context", []),
            })
            
            # Invoke task agent
            response: AgentResponse = await task_agent.invoke(
                input_text=f"Break down this task into steps: {task_description}",
                context=context,
                use_tools=True,
            )
            
            # Update state
            state["current_agent"] = "task_agent"
            if HAS_LANGGRAPH:
                state["messages"].append(AIMessage(content=response.content))
            else:
                state["messages"].append({"role": "assistant", "content": response.content})
            
            state["metadata"]["task_breakdown"] = response.content
            state["agent_history"].append({
                "agent": "task_agent",
                "role": AgentRole.TASK_DECOMPOSER.value,
                "response": response.content[:200],
                "timestamp": datetime.now().isoformat(),
            })
            
            if response.tool_calls:
                state["tool_calls"].extend(response.tool_calls)
            
            # Record metrics
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            monitor = self.monitor or await _get_monitor()
            if monitor:
                try:
                    monitor.record_request(
                        request_id=workflow_id,
                        user_id=state.get("user_id"),
                        duration_ms=duration_ms,
                        tokens_used=len(response.content.split()),
                        model=self.default_model,
                    )
                except Exception:
                    pass
            
            self.logger.info(f"Task agent completed in {duration_ms:.2f}ms", workflow_id=workflow_id)
            return state
            
        except Exception as e:
            self.logger.error(f"Task agent node failed: {e}", exc_info=True, workflow_id=workflow_id)
            error_msg = f"Error in task decomposition: {str(e)}"
            if HAS_LANGGRAPH:
                state["messages"].append(AIMessage(content=error_msg))
            else:
                state["messages"].append({"role": "assistant", "content": error_msg})
            state["errors"].append(error_msg)
            return state
    
    async def _plan_agent_node(self, state: AgentState) -> AgentState:
        """Planning agent node"""
        workflow_id = state.get("workflow_id", "unknown")
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Plan agent processing", workflow_id=workflow_id)
            
            task_description = state.get("task_description", "")
            task_breakdown = state.get("metadata", {}).get("task_breakdown", "")
            
            plan_agent = await self._get_agent(AgentRole.TIME_OPTIMIZER)
            
            context = state.get("context", {}).copy()
            context.update({
                "workflow_id": workflow_id,
                "task_breakdown": task_breakdown,
                "session_id": state.get("session_id"),
                "user_id": state.get("user_id"),
            })
            
            prompt = f"""Create an execution plan for this task:

Task: {task_description}

Breakdown: {task_breakdown}

Provide a detailed step-by-step plan with clear milestones."""
            
            response: AgentResponse = await plan_agent.invoke(
                input_text=prompt,
                context=context,
                use_tools=True,
            )
            
            state["plan"] = response.content
            state["current_agent"] = "plan_agent"
            if HAS_LANGGRAPH:
                state["messages"].append(AIMessage(content=f"Plan: {response.content}"))
            else:
                state["messages"].append({"role": "assistant", "content": f"Plan: {response.content}"})
            
            state["metadata"]["plan"] = response.content
            state["agent_history"].append({
                "agent": "plan_agent",
                "role": AgentRole.TIME_OPTIMIZER.value,
                "response": response.content[:200],
                "timestamp": datetime.now().isoformat(),
            })
            
            if response.tool_calls:
                state["tool_calls"].extend(response.tool_calls)
            
            # Store plan in memory
            memory_mgr = self.memory_manager or await _get_memory_manager()
            if memory_mgr:
                try:
                    await memory_mgr.store_memory(
                        content=f"Plan for {task_description}: {response.content}",
                        memory_type=MemoryType.EPISODIC,
                        session_id=state.get("session_id"),
                        user_id=state.get("user_id"),
                        importance=0.9,
                        metadata={"workflow_id": workflow_id, "agent": "plan_agent"},
                    )
                except Exception:
                    pass
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.info(f"Plan agent completed in {duration_ms:.2f}ms", workflow_id=workflow_id)
            return state
            
        except Exception as e:
            self.logger.error(f"Plan agent node failed: {e}", exc_info=True, workflow_id=workflow_id)
            error_msg = f"Error in planning: {str(e)}"
            if HAS_LANGGRAPH:
                state["messages"].append(AIMessage(content=error_msg))
            else:
                state["messages"].append({"role": "assistant", "content": error_msg})
            state["errors"].append(error_msg)
            return state
    
    async def _code_agent_node(self, state: AgentState) -> AgentState:
        """Code generation agent node"""
        workflow_id = state.get("workflow_id", "unknown")
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Code agent processing", workflow_id=workflow_id)
            
            task_description = state.get("task_description", "")
            plan = state.get("plan", "")
            task_breakdown = state.get("metadata", {}).get("task_breakdown", "")
            
            code_agent = await self._get_agent(AgentRole.CREATIVE_SPARKER)
            
            context = state.get("context", {}).copy()
            context.update({
                "workflow_id": workflow_id,
                "plan": plan,
                "task_breakdown": task_breakdown,
                "session_id": state.get("session_id"),
                "user_id": state.get("user_id"),
            })
            
            prompt = f"""Generate code based on this plan:

Task: {task_description}

Task Breakdown: {task_breakdown}

Plan: {plan}

Provide complete, working code with proper error handling and documentation."""
            
            response: AgentResponse = await code_agent.invoke(
                input_text=prompt,
                context=context,
                use_tools=True,
            )
            
            state["code"] = response.content
            state["current_agent"] = "code_agent"
            if HAS_LANGGRAPH:
                state["messages"].append(AIMessage(content=f"Code: {response.content}"))
            else:
                state["messages"].append({"role": "assistant", "content": f"Code: {response.content}"})
            
            state["metadata"]["code"] = response.content
            state["agent_history"].append({
                "agent": "code_agent",
                "role": AgentRole.CREATIVE_SPARKER.value,
                "response": response.content[:200],
                "timestamp": datetime.now().isoformat(),
            })
            
            if response.tool_calls:
                state["tool_calls"].extend(response.tool_calls)
            
            # Store code in memory
            memory_mgr = self.memory_manager or await _get_memory_manager()
            if memory_mgr:
                try:
                    await memory_mgr.store_memory(
                        content=f"Code for {task_description}",
                        memory_type=MemoryType.LONG_TERM,
                        session_id=state.get("session_id"),
                        user_id=state.get("user_id"),
                        importance=0.95,
                        metadata={
                            "workflow_id": workflow_id,
                            "agent": "code_agent",
                            "code_preview": response.content[:200],
                        },
                    )
                except Exception:
                    pass
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.info(f"Code agent completed in {duration_ms:.2f}ms", workflow_id=workflow_id)
            return state
            
        except Exception as e:
            self.logger.error(f"Code agent node failed: {e}", exc_info=True, workflow_id=workflow_id)
            error_msg = f"Error in code generation: {str(e)}"
            if HAS_LANGGRAPH:
                state["messages"].append(AIMessage(content=error_msg))
            else:
                state["messages"].append({"role": "assistant", "content": error_msg})
            state["errors"].append(error_msg)
            return state
    
    async def _qa_agent_node(self, state: AgentState) -> AgentState:
        """Quality assurance agent node"""
        workflow_id = state.get("workflow_id", "unknown")
        start_time = datetime.now()
        
        try:
            self.logger.info(f"QA agent processing", workflow_id=workflow_id)
            
            task_description = state.get("task_description", "")
            plan = state.get("plan", "")
            code = state.get("code", "")
            
            if not code:
                self.logger.warning("No code to review", workflow_id=workflow_id)
                state["quality_check"] = "No code generated to review"
                return state
            
            qa_agent = await self._get_agent(AgentRole.QUALITY_ASSURANCE, temperature=self.default_temperature * 0.5)
            
            context = state.get("context", {}).copy()
            context.update({
                "workflow_id": workflow_id,
                "plan": plan,
                "session_id": state.get("session_id"),
                "user_id": state.get("user_id"),
            })
            
            prompt = f"""Review and validate this code:

Task: {task_description}

Plan: {plan}

Code:
{code}

Check for correctness, error handling, code quality, and alignment with the plan."""
            
            response: AgentResponse = await qa_agent.invoke(
                input_text=prompt,
                context=context,
                use_tools=True,
            )
            
            state["quality_check"] = response.content
            state["current_agent"] = "qa_agent"
            if HAS_LANGGRAPH:
                state["messages"].append(AIMessage(content=f"Quality Check: {response.content}"))
            else:
                state["messages"].append({"role": "assistant", "content": f"Quality Check: {response.content}"})
            
            state["metadata"]["quality_check"] = response.content
            state["agent_history"].append({
                "agent": "qa_agent",
                "role": AgentRole.QUALITY_ASSURANCE.value,
                "response": response.content[:200],
                "timestamp": datetime.now().isoformat(),
            })
            
            if response.tool_calls:
                state["tool_calls"].extend(response.tool_calls)
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.info(f"QA agent completed in {duration_ms:.2f}ms", workflow_id=workflow_id)
            return state
            
        except Exception as e:
            self.logger.error(f"QA agent node failed: {e}", exc_info=True, workflow_id=workflow_id)
            error_msg = f"Error in quality assurance: {str(e)}"
            if HAS_LANGGRAPH:
                state["messages"].append(AIMessage(content=error_msg))
            else:
                state["messages"].append({"role": "assistant", "content": error_msg})
            state["errors"].append(error_msg)
            return state
    
    async def _lease_processor_node(self, state: AgentState) -> AgentState:
        """Lease abstraction processor node"""
        workflow_id = state.get("workflow_id", "unknown")
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Lease processor processing", workflow_id=workflow_id)
            
            # Extract lease information from context or task
            context = state.get("context", {})
            document_text = context.get("document_text", "")
            document_url = context.get("document_url", "")
            lease_id = context.get("lease_id", workflow_id)
            
            # Import and use LeaseProcessor
            from ..services.document_processors.lease_processor import LeaseProcessor
            
            processor = LeaseProcessor()
            result = await processor.process(
                document_text=document_text or state.get("task_description", ""),
                document_url=document_url,
                lease_id=lease_id,
            )
            
            state["result"] = result
            state["current_agent"] = "lease_processor"
            if HAS_LANGGRAPH:
                state["messages"].append(AIMessage(content=f"Lease abstraction completed: {result.get('confidence', 0):.2%} confidence"))
            else:
                state["messages"].append({"role": "assistant", "content": f"Lease abstraction completed: {result.get('confidence', 0):.2%} confidence"})
            
            state["agent_history"].append({
                "agent": "lease_processor",
                "role": "lease_abstraction",
                "result": "completed",
                "confidence": result.get("confidence", 0),
                "timestamp": datetime.now().isoformat(),
            })
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.info(f"Lease processor completed in {duration_ms:.2f}ms", workflow_id=workflow_id)
            return state
            
        except Exception as e:
            self.logger.error(f"Lease processor node failed: {e}", exc_info=True, workflow_id=workflow_id)
            error_msg = f"Error in lease processing: {str(e)}"
            if HAS_LANGGRAPH:
                state["messages"].append(AIMessage(content=error_msg))
            else:
                state["messages"].append({"role": "assistant", "content": error_msg})
            state["errors"].append(error_msg)
            return state
    
    async def _contract_processor_node(self, state: AgentState) -> AgentState:
        """Contract intelligence processor node"""
        workflow_id = state.get("workflow_id", "unknown")
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Contract processor processing", workflow_id=workflow_id)
            
            context = state.get("context", {})
            document_text = context.get("document_text", "")
            document_url = context.get("document_url", "")
            contract_id = context.get("contract_id", workflow_id)
            
            from ..services.document_processors.contract_processor import ContractProcessor
            
            processor = ContractProcessor()
            result = await processor.process(
                document_text=document_text or state.get("task_description", ""),
                document_url=document_url,
                contract_number=context.get("contract_number"),
                party_a=context.get("party_a"),
                party_b=context.get("party_b"),
            )
            
            state["result"] = result
            state["current_agent"] = "contract_processor"
            if HAS_LANGGRAPH:
                state["messages"].append(AIMessage(content=f"Contract processing completed: {result.get('confidence', 0):.2%} confidence"))
            else:
                state["messages"].append({"role": "assistant", "content": f"Contract processing completed: {result.get('confidence', 0):.2%} confidence"})
            
            state["agent_history"].append({
                "agent": "contract_processor",
                "role": "contract_intelligence",
                "result": "completed",
                "confidence": result.get("confidence", 0),
                "timestamp": datetime.now().isoformat(),
            })
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.info(f"Contract processor completed in {duration_ms:.2f}ms", workflow_id=workflow_id)
            return state
            
        except Exception as e:
            self.logger.error(f"Contract processor node failed: {e}", exc_info=True, workflow_id=workflow_id)
            error_msg = f"Error in contract processing: {str(e)}"
            if HAS_LANGGRAPH:
                state["messages"].append(AIMessage(content=error_msg))
            else:
                state["messages"].append({"role": "assistant", "content": error_msg})
            state["errors"].append(error_msg)
            return state
    
    async def _fraud_agent_node(self, state: AgentState) -> AgentState:
        """Vendor fraud detection agent node"""
        workflow_id = state.get("workflow_id", "unknown")
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Fraud agent processing", workflow_id=workflow_id)
            
            fraud_agent = await self._get_agent(AgentRole.VENDOR_INTELLIGENCE)
            
            context = state.get("context", {}).copy()
            context.update({
                "workflow_id": workflow_id,
                "session_id": state.get("session_id"),
                "user_id": state.get("user_id"),
            })
            
            task_description = state.get("task_description", "")
            
            response: AgentResponse = await fraud_agent.invoke(
                input_text=f"Analyze this for vendor fraud: {task_description}",
                context=context,
                use_tools=True,
            )
            
            state["result"] = {
                "analysis": response.content,
                "agent": "vendor_fraud",
            }
            state["current_agent"] = "fraud_agent"
            if HAS_LANGGRAPH:
                state["messages"].append(AIMessage(content=response.content))
            else:
                state["messages"].append({"role": "assistant", "content": response.content})
            
            state["agent_history"].append({
                "agent": "fraud_agent",
                "role": AgentRole.VENDOR_INTELLIGENCE.value,
                "response": response.content[:200],
                "timestamp": datetime.now().isoformat(),
            })
            
            if response.tool_calls:
                state["tool_calls"].extend(response.tool_calls)
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.info(f"Fraud agent completed in {duration_ms:.2f}ms", workflow_id=workflow_id)
            return state
            
        except Exception as e:
            self.logger.error(f"Fraud agent node failed: {e}", exc_info=True, workflow_id=workflow_id)
            error_msg = f"Error in fraud detection: {str(e)}"
            if HAS_LANGGRAPH:
                state["messages"].append(AIMessage(content=error_msg))
            else:
                state["messages"].append({"role": "assistant", "content": error_msg})
            state["errors"].append(error_msg)
            return state
    
    def _should_continue_after_task(self, state: AgentState) -> str:
        """Conditional routing after task agent"""
        if state.get("metadata", {}).get("task_breakdown"):
            return "plan_agent"
        return "end"
    
    def _should_continue_after_plan(self, state: AgentState) -> str:
        """Conditional routing after plan agent"""
        if state.get("plan"):
            return "code_agent"
        return "end"
    
    def _should_continue_after_code(self, state: AgentState) -> str:
        """Conditional routing after code agent"""
        if state.get("code"):
            return "qa_agent"
        return "end"
    
    async def ainvoke(
        self,
        input_data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the workflow
        
        Args:
            input_data: Input data with 'messages' or 'task_description'
            config: LangGraph config (e.g., {"configurable": {"thread_id": "..."}})
        
        Returns:
            Final state after workflow execution
        """
        if not self.graph:
            return await self._fallback_execute(input_data)
        
        try:
            # Prepare initial state
            messages = input_data.get("messages", [])
            if not messages:
                task_description = input_data.get("task_description", "")
                if task_description:
                    if HAS_LANGGRAPH:
                        messages = [HumanMessage(content=task_description)]
                    else:
                        messages = [{"role": "user", "content": task_description}]
                else:
                    if HAS_LANGGRAPH:
                        messages = [HumanMessage(content="Process this task")]
                    else:
                        messages = [{"role": "user", "content": "Process this task"}]
            
            workflow_id = input_data.get("workflow_id", f"workflow_{datetime.now().timestamp()}")
            session_id = input_data.get("session_id")
            user_id = input_data.get("user_id")
            
            # Initialize session
            session_mgr = self.session_manager or await _get_session_manager()
            if session_mgr and session_id:
                try:
                    session = await session_mgr.get_session(session_id)
                    if not session:
                        await session_mgr.create_session(
                            session_id=session_id,
                            user_id=user_id,
                            context=input_data.get("context", {}),
                        )
                except Exception as e:
                    self.logger.debug(f"Session initialization failed: {e}")
            
            initial_state: AgentState = {
                "messages": messages,
                "current_agent": "task_router",
                "task_description": input_data.get("task_description", ""),
                "task_type": None,
                "plan": None,
                "code": None,
                "quality_check": None,
                "result": None,
                "context": input_data.get("context", {}),
                "workflow_id": workflow_id,
                "workflow_type": "standard",
                "session_id": session_id,
                "user_id": user_id,
                "metadata": {},
                "errors": [],
                "tool_calls": [],
                "agent_history": [],
            }
            
            # Execute graph
            if config:
                result = await self.graph.ainvoke(initial_state, config=config)
            else:
                result = await self.graph.ainvoke(initial_state)
            
            # Publish completion event
            event_reg = self.event_registry or await _get_event_registry()
            if event_reg:
                try:
                    await event_reg.publish_event(
                        event_type="workflow.completed",
                        payload={
                            "workflow_id": workflow_id,
                            "workflow_type": result.get("workflow_type", "standard"),
                            "success": len(result.get("errors", [])) == 0,
                        }
                    )
                except Exception:
                    pass
            
            self.logger.info("LangGraph workflow executed successfully", 
                           workflow_id=result.get("workflow_id"))
            return result
            
        except Exception as e:
            self.logger.error(f"LangGraph workflow execution failed: {e}", exc_info=True)
            raise
    
    async def _fallback_execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback execution without LangGraph"""
        self.logger.warning("Using fallback execution (LangGraph not available)")
        
        workflow_id = input_data.get("workflow_id", f"workflow_{datetime.now().timestamp()}")
        task_description = input_data.get("task_description", "")
        context = input_data.get("context", {})
        session_id = input_data.get("session_id")
        user_id = input_data.get("user_id")
        
        # Detect task type
        task_type = self._detect_task_type(task_description)
        
        state = {
            "workflow_id": workflow_id,
            "task_description": task_description,
            "task_type": task_type,
            "workflow_type": task_type,
            "plan": None,
            "code": None,
            "quality_check": None,
            "result": None,
            "context": context,
            "session_id": session_id,
            "user_id": user_id,
            "metadata": {},
            "errors": [],
            "tool_calls": [],
            "messages": [],
            "agent_history": [],
        }
        
        # Route to appropriate handler
        try:
            if task_type == "lease":
                # Use lease processor
                from ..services.document_processors.lease_processor import LeaseProcessor
                processor = LeaseProcessor()
                result = await processor.process(
                    document_text=context.get("document_text", task_description),
                    document_url=context.get("document_url", ""),
                    lease_id=context.get("lease_id", workflow_id),
                )
                state["result"] = result
                
            elif task_type == "contract":
                # Use contract processor
                from ..services.document_processors.contract_processor import ContractProcessor
                processor = ContractProcessor()
                result = await processor.process(
                    document_text=context.get("document_text", task_description),
                    document_url=context.get("document_url", ""),
                )
                state["result"] = result
                
            elif task_type == "fraud":
                # Use fraud agent
                fraud_agent = await self._get_agent(AgentRole.VENDOR_INTELLIGENCE)
                response = await fraud_agent.invoke(
                    input_text=f"Analyze for vendor fraud: {task_description}",
                    context=context,
                    use_tools=True,
                )
                state["result"] = {"analysis": response.content}
                state["messages"].append({"role": "assistant", "content": response.content})
                
            else:
                # Standard workflow
                task_agent = await self._get_agent(AgentRole.TASK_DECOMPOSER)
                task_response = await task_agent.invoke(
                    input_text=f"Break down this task: {task_description}",
                    context=context,
                    use_tools=True,
                )
                state["metadata"]["task_breakdown"] = task_response.content
                state["messages"].append({"role": "assistant", "content": task_response.content})
                
                plan_agent = await self._get_agent(AgentRole.TIME_OPTIMIZER)
                plan_response = await plan_agent.invoke(
                    input_text=f"Create a plan for: {task_description}\n\nBreakdown: {task_response.content}",
                    context=context,
                    use_tools=True,
                )
                state["plan"] = plan_response.content
                state["messages"].append({"role": "assistant", "content": f"Plan: {plan_response.content}"})
                
                code_agent = await self._get_agent(AgentRole.CREATIVE_SPARKER)
                code_response = await code_agent.invoke(
                    input_text=f"Generate code for: {task_description}\n\nPlan: {plan_response.content}",
                    context=context,
                    use_tools=True,
                )
                state["code"] = code_response.content
                state["messages"].append({"role": "assistant", "content": f"Code: {code_response.content}"})
        
        except Exception as e:
            self.logger.error(f"Fallback execution failed: {e}", exc_info=True)
            state["errors"].append(str(e))
            state["metadata"]["error"] = str(e)
        
        return state


# Global workflow instance
_langgraph_workflow: Optional[LangGraphMultiAgentWorkflow] = None


async def get_langgraph_workflow(
    llm_provider=None,
    tool_registry=None,
    vector_store=None,
    redis_client=None,
    memory_manager=None,
    session_manager=None,
    monitor=None,
    rag_pipeline=None,
    event_registry=None,
    state_processor=None,
) -> Optional[LangGraphMultiAgentWorkflow]:
    """Get or create LangGraph workflow singleton"""
    global _langgraph_workflow
    
    if _langgraph_workflow is None:
        # Get Redis client if not provided
        if redis_client is None and HAS_REDIS:
            try:
                redis_url = os.getenv("REDIS_URL")
                if not redis_url:
                    redis_host = getattr(settings, "REDIS_HOST", "localhost")
                    redis_port = getattr(settings, "REDIS_PORT", 6379)
                    redis_password = getattr(settings, "REDIS_PASSWORD", None)
                    redis_db = getattr(settings, "REDIS_DB", 0)
                    
                    if redis_password:
                        redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}"
                    else:
                        redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
                
                redis_client = redis.from_url(redis_url, decode_responses=True)
                logger.info(f"Redis client created: {redis_url.split('@')[-1] if '@' in redis_url else redis_url}")
            except Exception as e:
                logger.warning(f"Failed to create Redis client: {e}")
                redis_client = None
        
        # Lazy load optional components if not provided
        if memory_manager is None:
            memory_manager = await _get_memory_manager()
        
        if session_manager is None:
            session_manager = await _get_session_manager()
        
        if monitor is None:
            monitor = await _get_monitor()
        
        if rag_pipeline is None:
            rag_pipeline = await _get_rag_pipeline()
        
        if event_registry is None:
            event_registry = await _get_event_registry()
        
        if state_processor is None:
            state_processor = await _get_state_processor()
        
        _langgraph_workflow = LangGraphMultiAgentWorkflow(
            llm_provider=llm_provider,
            tool_registry=tool_registry,
            vector_store=vector_store,
            redis_client=redis_client,
            memory_manager=memory_manager,
            session_manager=session_manager,
            monitor=monitor,
            rag_pipeline=rag_pipeline,
            event_registry=event_registry,
            state_processor=state_processor,
        )
    
    return _langgraph_workflow
