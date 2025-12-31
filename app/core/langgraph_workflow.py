"""
LangGraph Multi-Agent Workflow
Graph-based agent routing for task → plan → code flow
Replaces custom MultiAgentCoordinator with LangGraph state machine
"""
from typing import TypedDict, Annotated, Sequence, Optional, Dict, Any, List
from datetime import datetime
import operator
import asyncio
from ..logging_config import get_logger
from ..core.types import AgentRole
from ..agents.agent_factory import AgentFactory
from ..agents.base_agent import AgentResponse

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


# State definition for LangGraph workflow
if HAS_LANGGRAPH:
    class AgentState(TypedDict):
        """State structure for multi-agent workflow"""
        messages: Annotated[Sequence[BaseMessage], operator.add]
        current_agent: str
        task_description: str
        plan: Optional[str]
        code: Optional[str]
        context: Dict[str, Any]
        workflow_id: str
        metadata: Dict[str, Any]
else:
    # Fallback TypedDict without Annotated for type checking
    class AgentState(TypedDict):
        """State structure for multi-agent workflow"""
        messages: List[Any]
        current_agent: str
        task_description: str
        plan: Optional[str]
        code: Optional[str]
        context: Dict[str, Any]
        workflow_id: str
        metadata: Dict[str, Any]


class LangGraphMultiAgentWorkflow:
    """
    LangGraph-based multi-agent workflow
    Implements task → plan → code flow with graph-based routing
    """
    
    def __init__(
        self,
        llm_provider=None,
        tool_registry=None,
        vector_store=None,
        redis_client=None,
    ):
        self.llm_provider = llm_provider
        self.tool_registry = tool_registry
        self.vector_store = vector_store
        self.redis_client = redis_client
        self.logger = logger
        self.graph = None
        self.checkpointer = None
        
        # Initialize agents (lazy loading)
        self._task_agent = None
        self._plan_agent = None
        self._code_agent = None
        
        if HAS_LANGGRAPH:
            self._build_graph()
        else:
            logger.warning("LangGraph not available, workflow will use fallback mode")
    
    async def _get_task_agent(self):
        """Get or create task decomposer agent"""
        if not self._task_agent:
            self._task_agent = await AgentFactory.create_agent(
                role=AgentRole.TASK_DECOMPOSER,
                model_name="llama3:8b",
                temperature=0.7,
            )
        return self._task_agent
    
    async def _get_plan_agent(self):
        """Get or create planning agent (using time optimizer as planner)"""
        if not self._plan_agent:
            self._plan_agent = await AgentFactory.create_agent(
                role=AgentRole.TIME_OPTIMIZER,
                model_name="llama3:8b",
                temperature=0.7,
            )
        return self._plan_agent
    
    async def _get_code_agent(self):
        """Get or create code generation agent (using creative sparker for code)"""
        if not self._code_agent:
            self._code_agent = await AgentFactory.create_agent(
                role=AgentRole.CREATIVE_SPARKER,
                model_name="llama3:8b",
                temperature=0.7,
            )
        return self._code_agent
    
    def _build_graph(self):
        """Build the LangGraph state machine"""
        if not HAS_LANGGRAPH:
            return
        
        try:
            workflow = StateGraph(AgentState)
            
            # Add agent nodes
            workflow.add_node("task_agent", self._task_agent_node)
            workflow.add_node("plan_agent", self._plan_agent_node)
            workflow.add_node("code_agent", self._code_agent_node)
            
            # Set entry point
            workflow.set_entry_point("task_agent")
            
            # Add conditional routing
            workflow.add_conditional_edges(
                "task_agent",
                self._should_continue,
                {
                    "plan_agent": "plan_agent",
                    "code_agent": "code_agent",
                    "end": END,
                }
            )
            
            # Add edges
            workflow.add_edge("plan_agent", "code_agent")
            workflow.add_edge("code_agent", END)
            
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
    
    async def _task_agent_node(self, state: AgentState) -> AgentState:
        """Task decomposition agent node"""
        try:
            self.logger.info("Task agent processing", workflow_id=state.get("workflow_id"))
            
            # Get task description from messages or state
            task_description = state.get("task_description", "")
            if not task_description and state.get("messages"):
                last_message = state["messages"][-1]
                if isinstance(last_message, HumanMessage):
                    task_description = last_message.content
            
            if not task_description:
                task_description = "No task provided"
            
            # Get task agent
            task_agent = await self._get_task_agent()
            
            # Invoke task agent
            response: AgentResponse = await task_agent.invoke(
                input_text=f"Break down this task into steps: {task_description}",
                context=state.get("context", {}),
            )
            
            # Update state
            state["task_description"] = task_description
            state["current_agent"] = "task_agent"
            state["messages"].append(AIMessage(content=response.content))
            state["metadata"]["task_breakdown"] = response.content
            
            self.logger.info("Task agent completed", workflow_id=state.get("workflow_id"))
            return state
            
        except Exception as e:
            self.logger.error(f"Task agent node failed: {e}", exc_info=True)
            state["messages"].append(AIMessage(content=f"Error in task decomposition: {str(e)}"))
            return state
    
    async def _plan_agent_node(self, state: AgentState) -> AgentState:
        """Planning agent node"""
        try:
            self.logger.info("Plan agent processing", workflow_id=state.get("workflow_id"))
            
            task_description = state.get("task_description", "")
            task_breakdown = state.get("metadata", {}).get("task_breakdown", "")
            
            # Get plan agent
            plan_agent = await self._get_plan_agent()
            
            # Invoke plan agent
            prompt = f"""Create an execution plan for this task:

Task: {task_description}

Breakdown: {task_breakdown}

Provide a detailed step-by-step plan."""
            
            response: AgentResponse = await plan_agent.invoke(
                input_text=prompt,
                context=state.get("context", {}),
            )
            
            # Update state
            state["plan"] = response.content
            state["current_agent"] = "plan_agent"
            state["messages"].append(AIMessage(content=f"Plan: {response.content}"))
            state["metadata"]["plan"] = response.content
            
            self.logger.info("Plan agent completed", workflow_id=state.get("workflow_id"))
            return state
            
        except Exception as e:
            self.logger.error(f"Plan agent node failed: {e}", exc_info=True)
            state["messages"].append(AIMessage(content=f"Error in planning: {str(e)}"))
            return state
    
    async def _code_agent_node(self, state: AgentState) -> AgentState:
        """Code generation agent node"""
        try:
            self.logger.info("Code agent processing", workflow_id=state.get("workflow_id"))
            
            task_description = state.get("task_description", "")
            plan = state.get("plan", "")
            
            # Get code agent
            code_agent = await self._get_code_agent()
            
            # Invoke code agent
            prompt = f"""Generate code based on this plan:

Task: {task_description}

Plan: {plan}

Provide complete, working code."""
            
            response: AgentResponse = await code_agent.invoke(
                input_text=prompt,
                context=state.get("context", {}),
            )
            
            # Update state
            state["code"] = response.content
            state["current_agent"] = "code_agent"
            state["messages"].append(AIMessage(content=f"Code: {response.content}"))
            state["metadata"]["code"] = response.content
            
            self.logger.info("Code agent completed", workflow_id=state.get("workflow_id"))
            return state
            
        except Exception as e:
            self.logger.error(f"Code agent node failed: {e}", exc_info=True)
            state["messages"].append(AIMessage(content=f"Error in code generation: {str(e)}"))
            return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Conditional routing based on state"""
        # If no plan, go to plan agent
        if not state.get("plan"):
            return "plan_agent"
        # If plan exists but no code, go to code agent
        elif not state.get("code"):
            return "code_agent"
        # Otherwise, end
        else:
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
            # Fallback execution
            return await self._fallback_execute(input_data)
        
        try:
            # Prepare initial state
            messages = input_data.get("messages", [])
            if not messages:
                # Create HumanMessage from task_description if provided
                task_description = input_data.get("task_description", "")
                if task_description:
                    messages = [HumanMessage(content=task_description)]
                else:
                    messages = [HumanMessage(content="Process this task")]
            
            initial_state: AgentState = {
                "messages": messages,
                "current_agent": "task_agent",
                "task_description": input_data.get("task_description", ""),
                "plan": None,
                "code": None,
                "context": input_data.get("context", {}),
                "workflow_id": input_data.get("workflow_id", f"workflow_{datetime.now().timestamp()}"),
                "metadata": {},
            }
            
            # Execute graph
            if config:
                result = await self.graph.ainvoke(initial_state, config=config)
            else:
                result = await self.graph.ainvoke(initial_state)
            
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
        
        state = {
            "workflow_id": workflow_id,
            "task_description": task_description,
            "plan": None,
            "code": None,
            "context": context,
            "metadata": {},
        }
        
        # Sequential execution
        try:
            # Task agent
            task_agent = await self._get_task_agent()
            task_response = await task_agent.invoke(
                input_text=f"Break down this task: {task_description}",
                context=context,
            )
            state["metadata"]["task_breakdown"] = task_response.content
            
            # Plan agent
            plan_agent = await self._get_plan_agent()
            plan_response = await plan_agent.invoke(
                input_text=f"Create a plan for: {task_description}\n\nBreakdown: {task_response.content}",
                context=context,
            )
            state["plan"] = plan_response.content
            state["metadata"]["plan"] = plan_response.content
            
            # Code agent
            code_agent = await self._get_code_agent()
            code_response = await code_agent.invoke(
                input_text=f"Generate code for: {task_description}\n\nPlan: {plan_response.content}",
                context=context,
            )
            state["code"] = code_response.content
            state["metadata"]["code"] = code_response.content
            
        except Exception as e:
            self.logger.error(f"Fallback execution failed: {e}", exc_info=True)
            state["metadata"]["error"] = str(e)
        
        return state


# Global workflow instance
_langgraph_workflow: Optional[LangGraphMultiAgentWorkflow] = None


async def get_langgraph_workflow(
    llm_provider=None,
    tool_registry=None,
    vector_store=None,
    redis_client=None,
) -> Optional[LangGraphMultiAgentWorkflow]:
    """Get or create LangGraph workflow singleton"""
    global _langgraph_workflow
    
    if _langgraph_workflow is None:
        # Get Redis client if not provided
        if redis_client is None and HAS_REDIS:
            try:
                from ..settings import settings
                redis_url = getattr(settings, "REDIS_URL", "redis://redis:6379")
                redis_client = redis.from_url(redis_url, decode_responses=True)
            except Exception as e:
                logger.warning(f"Failed to create Redis client: {e}")
                redis_client = None
        
        _langgraph_workflow = LangGraphMultiAgentWorkflow(
            llm_provider=llm_provider,
            tool_registry=tool_registry,
            vector_store=vector_store,
            redis_client=redis_client,
        )
    
    return _langgraph_workflow

