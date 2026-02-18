"""
Workflow Orchestrator
Main orchestration engine that coordinates all components
Integrates LangChain, local LLMs, RAG, tools, and state management
"""
from typing import Dict, List, Optional, Any, Iterator, Union, AsyncIterator
import json
import time
import asyncio
import hashlib
from datetime import datetime
from collections import OrderedDict
from ..logging_config import get_logger

logger = get_logger("cyrex.orchestrator")

# Thread-safe LRU cache for LLM instances and agent executors
class LRUCache:
    """Thread-safe LRU cache with size limit and statistics"""
    def __init__(self, max_size: int = 10):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache (thread-safe)"""
        async with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    async def put(self, key: str, value: Any) -> None:
        """Put item in cache with LRU eviction (thread-safe)"""
        async with self.lock:
            if key in self.cache:
                # Update existing item
                self.cache.move_to_end(key)
            else:
                # Add new item
                if len(self.cache) >= self.max_size:
                    # Evict least recently used
                    evicted_key, _ = self.cache.popitem(last=False)
                    self.evictions += 1
                    logger.debug(f"Evicted LRU cache entry: {evicted_key}")
            self.cache[key] = value
    
    async def clear(self) -> None:
        """Clear all cache entries (thread-safe)"""
        async with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": f"{hit_rate:.1f}%",
            "keys": list(self.cache.keys())
        }


# LangChain imports with graceful fallbacks
HAS_LANGCHAIN = False
HAS_AGENTS = False
try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.runnables.utils import Input, Output
    from langchain_core.exceptions import OutputParserException
    HAS_LANGCHAIN = True
except ImportError as e:
    logger.warning(f"LangChain core not available: {e}")
    ChatPromptTemplate = None
    MessagesPlaceholder = None
    JsonOutputParser = None
    StrOutputParser = None
    RunnablePassthrough = None
    RunnableLambda = None
    Input = None
    Output = None
    OutputParserException = Exception  # Fallback

# Agent imports with graceful fallbacks (LangChain 0.2.x compatible)
try:
    # Try multiple import paths for LangChain 0.2.x
    AgentExecutor = None
    create_openai_functions_agent = None
    create_react_agent = None
    
    # Try langchain-classic first (for deprecated AgentExecutor)
    try:
        from langchain_classic.agents import AgentExecutor
        logger.debug("Found AgentExecutor in langchain-classic")
    except ImportError:
        # Try langchain.agents.agent (some 0.2.x versions)
        try:
            from langchain.agents.agent import AgentExecutor
            logger.debug("Found AgentExecutor in langchain.agents.agent")
        except ImportError:
            # Try direct langchain.agents (older versions)
            try:
                from langchain.agents import AgentExecutor
                logger.debug("Found AgentExecutor in langchain.agents")
            except ImportError:
                pass
    
    # Try to import agent creation functions
    # In LangChain 1.x, create_react_agent is in langchain_classic.agents.react.agent
    try:
        from langchain_classic.agents.react.agent import create_react_agent
        logger.info("Successfully imported create_react_agent from langchain_classic.agents.react.agent")
    except ImportError:
        try:
            from langchain_experimental.agents import create_react_agent
            logger.info("Successfully imported create_react_agent from langchain_experimental.agents")
        except ImportError:
            try:
                from langchain.agents import create_openai_functions_agent, create_react_agent
                logger.info("Successfully imported create_react_agent from langchain.agents")
            except ImportError as e:
                logger.debug(f"Failed to import from langchain.agents: {e}")
    # Try alternative paths
    try:
        from langchain.agents.openai_functions import create_openai_functions_agent
        logger.debug("Successfully imported create_openai_functions_agent from langchain.agents.openai_functions")
    except ImportError as e2:
        logger.debug(f"Failed to import from langchain.agents.openai_functions: {e2}")
    try:
        from langchain.agents.react import create_react_agent
        logger.info("Successfully imported create_react_agent from langchain.agents.react")
    except ImportError as e3:
        logger.warning(f"Failed to import create_react_agent from all locations: {e3}")
        logger.warning("Install langchain-experimental: pip install langchain-experimental")
    
    # Check if we have at least AgentExecutor
    if AgentExecutor is not None:
        HAS_AGENTS = True
        logger.debug("LangChain agents available")
    else:
        raise ImportError("AgentExecutor not found in any expected location")
        
except ImportError as e:
    logger.warning(f"LangChain agents not available: {e}")
    create_openai_functions_agent = None
    create_react_agent = None
    AgentExecutor = None
    HAS_AGENTS = False

# LangChain hub for ReAct prompts (optional)
try:
    from langchain import hub
    HAS_HUB = True
except ImportError:
    logger.debug("langchain-hub not available, will use fallback ReAct prompt")
    hub = None
    HAS_HUB = False

from .execution_engine import TaskExecutionEngine, get_execution_engine
from .state_manager import WorkflowStateManager, get_state_manager
from .tool_registry import ToolRegistry, get_tool_registry, ToolCategory
from .guardrails import SafetyGuardrails, get_guardrails
from .queue_manager import TaskQueueManager, get_queue_manager, TaskPriority
from .monitoring import SystemMonitor, get_monitor
from .prompt_manager import PromptVersionManager, get_prompt_manager
from ..integrations.local_llm import LocalLLMProvider, get_local_llm, LLMBackend
from ..integrations.openai_wrapper import OpenAIProvider, get_openai_provider
from ..integrations.milvus_store import MilvusVectorStore, get_milvus_store
from ..integrations.rag_bridge import RAGBridge, get_rag_bridge
from ..services.knowledge_retrieval_engine import KnowledgeRetrievalEngine
from ..settings import settings
from ..integrations.streaming.event_publisher import CyrexEventPublisher


class WorkflowOrchestrator:
    """
    Main orchestration engine for Deepiri AI workflows
    Coordinates LLM, RAG, tools, state, and execution
    """
    
    def __init__(
        self,
        llm_provider: Optional[Union[LocalLLMProvider, OpenAIProvider]] = None,
        vector_store: Optional[MilvusVectorStore] = None,
        execution_engine: Optional[TaskExecutionEngine] = None,
        state_manager: Optional[WorkflowStateManager] = None,
        tool_registry: Optional[ToolRegistry] = None,
        guardrails: Optional[SafetyGuardrails] = None,
        queue_manager: Optional[TaskQueueManager] = None,
        monitor: Optional[SystemMonitor] = None,
        prompt_manager: Optional[PromptVersionManager] = None,
        rag_bridge: Optional[RAGBridge] = None,
    ):
        self.logger = logger
        
        # Initialize LLM provider - prefer OpenAI, fallback to local LLM
        if llm_provider:
            self.llm_provider = llm_provider
        else:
            self.llm_provider = None
            
            # Try OpenAI first if API key is available
            if settings.OPENAI_API_KEY:
                try:
                    # Create OpenAI wrapper that matches LocalLLMProvider interface
                    from ..integrations.openai_wrapper import get_openai_provider
                    self.llm_provider = get_openai_provider()
                    if self.llm_provider:
                        self.logger.info(f"Using OpenAI: {settings.OPENAI_MODEL}")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize OpenAI: {e}, will try local LLM")
                    self.llm_provider = None
            
            # Fallback to local LLM if OpenAI not configured or failed
            if not self.llm_provider:
                try:
                    backend = LLMBackend(settings.LOCAL_LLM_BACKEND)
                    self.llm_provider = get_local_llm(
                        backend=backend.value,
                        model_name=settings.LOCAL_LLM_MODEL,
                        base_url=settings.OLLAMA_BASE_URL if backend == LLMBackend.OLLAMA else None,
                    )
                    if self.llm_provider:
                        self.logger.info(f"Using local LLM: {backend.value}")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize local LLM: {e}")
                    self.llm_provider = None
        
        # Initialize vector store (Milvus)
        if vector_store:
            self.vector_store = vector_store
        else:
            try:
                self.vector_store = get_milvus_store(
                    collection_name="deepiri_knowledge",
                )
            except Exception as e:
                self.logger.warning(f"Milvus not available: {e}")
                self.vector_store = None
        
        # Initialize knowledge engine for RAG bridge
        knowledge_engine = None
        if not self.vector_store:
            try:
                knowledge_engine = KnowledgeRetrievalEngine(
                    vector_store_type="milvus" if settings.MILVUS_HOST else "chroma",
                )
            except Exception as e:
                self.logger.warning(f"Knowledge engine not available: {e}")
        
        # Core components
        self.execution_engine = execution_engine or get_execution_engine()
        self.state_manager = state_manager or get_state_manager()
        self.tool_registry = tool_registry or get_tool_registry()
        self.guardrails = guardrails or get_guardrails()
        self.queue_manager = queue_manager or get_queue_manager()
        self.monitor = monitor or get_monitor()
        self.prompt_manager = prompt_manager or get_prompt_manager()
        self.rag_bridge = rag_bridge or get_rag_bridge(knowledge_engine, self.vector_store)
        
        # LangGraph multi-agent workflow (optional, for complex multi-step workflows)
        self.langgraph_workflow = None
        try:
            from .langgraph_workflow import get_langgraph_workflow
            self._langgraph_workflow_getter = get_langgraph_workflow
        except ImportError:
            self.logger.debug("LangGraph multi-agent workflow not available")
            self._langgraph_workflow_getter = None

        # Response cache for common queries (latency optimization)
        self._response_cache: OrderedDict[str, tuple] = OrderedDict()
        self._response_cache_max_size = 1000
        self._response_cache_ttl = 3600.0  # 1 hour TTL
        self._response_cache_lock = asyncio.Lock()
        
        # LangGraph tool-calling agent (primary execution path)
        # This replaces the old AgentExecutor entirely
        from . import langgraph_agent
        self._langgraph_agent_available = langgraph_agent.is_available()
        if self._langgraph_agent_available:
            self.logger.info("LangGraph tool-calling agent available (native tool calls, no ReAct parsing)")
        else:
            self.logger.warning("LangGraph tool-calling agent NOT available - falling back to AgentExecutor")

        # LRU caches: model instances and compiled agent graphs
        self._graph_cache = LRUCache(max_size=10)
        self._llm_cache = LRUCache(max_size=10)
        self._agent_executor_cache = LRUCache(max_size=20)
        self._current_model: Optional[str] = None
        self._cache_lock = asyncio.Lock()
        
        # Agent tools initialization flag
        self._agent_tools_initialized = False
        
        # Register default tools
        self._register_default_tools()
        
        # Initialize chains
        self._setup_chains()
    
    def _register_default_tools(self):
        """Register default tools for workflow execution"""
        import asyncio
        try:
            # Register knowledge_retrieval tool if RAG bridge is available
            if self.rag_bridge:
                from langchain_core.tools import Tool
                
                async def knowledge_retrieval_func_async(input_data: Dict[str, Any]) -> str:
                    """Retrieve knowledge from vector store using RAG"""
                    query = input_data.get("query", "")
                    if not query:
                        return "Error: 'query' parameter is required"
                    
                    try:
                        # Use RAG bridge to retrieve documents
                        results = await self.rag_bridge.retrieve(query, top_k=5)
                        if results and len(results) > 0:
                            # Format results as string - handle Document objects
                            formatted_parts = []
                            for i, doc in enumerate(results[:5]):
                                if hasattr(doc, 'page_content'):
                                    content = doc.page_content
                                    metadata = getattr(doc, 'metadata', {})
                                    formatted_parts.append(
                                        f"Document {i+1}:\n{content}\n"
                                        f"(Source: {metadata.get('source', 'unknown')})"
                                    )
                                elif isinstance(doc, dict):
                                    formatted_parts.append(
                                        f"Document {i+1}:\n{doc.get('content', doc.get('text', ''))}"
                                    )
                                else:
                                    formatted_parts.append(f"Document {i+1}:\n{str(doc)}")
                            return "\n\n".join(formatted_parts)
                        else:
                            return "No relevant documents found for the query."
                    except Exception as e:
                        self.logger.error(f"Knowledge retrieval failed: {e}", exc_info=True)
                        return f"Error retrieving knowledge: {str(e)}"
                
                # Create sync wrapper for async function
                # Try to capture the running event loop (may not exist during __init__)
                try:
                    _orch_loop = asyncio.get_running_loop()
                except RuntimeError:
                    _orch_loop = None

                def knowledge_retrieval_func(input_data):
                    """Sync wrapper for knowledge retrieval"""
                    # Handle both dict and string inputs
                    if isinstance(input_data, str):
                        input_data = {"query": input_data}
                    elif not isinstance(input_data, dict):
                        input_data = {"query": str(input_data)}
                    
                    # Schedule on the main event loop if available
                    if _orch_loop is not None and _orch_loop.is_running():
                        future = asyncio.run_coroutine_threadsafe(
                            knowledge_retrieval_func_async(input_data), _orch_loop
                        )
                        return future.result(timeout=30)
                    else:
                        loop = asyncio.new_event_loop()
                        try:
                            return loop.run_until_complete(knowledge_retrieval_func_async(input_data))
                        finally:
                            loop.close()
                
                # Create tool from function
                knowledge_tool = Tool(
                    name="knowledge_retrieval",
                    description="Retrieve relevant knowledge/documents from the vector store using semantic search. Input should be a dict with 'query' key containing the search query, or a string query.",
                    func=knowledge_retrieval_func
                )
                
                self.tool_registry.register_tool(
                    knowledge_tool,
                    category=ToolCategory.DATA,
                )
                self.logger.info("Registered knowledge_retrieval tool")
        except Exception as e:
            self.logger.warning(f"Failed to register default tools: {e}", exc_info=True)
    
    async def initialize_agent_tools(self):
        """
        Initialize all agent tools from app/agents/tools
        Call this after orchestrator is created to register all available tools
        """
        if self._agent_tools_initialized:
            self.logger.debug("Agent tools already initialized")
            return
        
        try:
            from .agent_integration import register_all_agent_tools
            stats = await register_all_agent_tools(self.tool_registry)
            self._agent_tools_initialized = True
            self.logger.info(f"Agent tools initialized: {stats}")
            return stats
        except Exception as e:
            self.logger.error(f"Failed to initialize agent tools: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _setup_chains(self):
        """Setup LangChain chains for different workflows"""
        if not self.llm_provider:
            self.logger.warning("LLM provider not available, chains not initialized")
            self.rag_chain = None
            self.agent_executor = None
            return
        
        try:
            llm = self.llm_provider.get_llm()
            
            # Cache the initial LLM instance (sync during init)
            initial_model = self.llm_provider.config.model_name if hasattr(self.llm_provider, 'config') else 'default'
            # Direct cache access during initialization (no async needed yet)
            self._llm_cache.cache[initial_model] = llm
            self._current_model = initial_model
            self.logger.info(f"Cached initial LLM instance for {initial_model}")
            
            # RAG chain with proper document formatting
            if self.vector_store:
                # MilvusVectorStore has similarity_search, not get_retriever; wrap for LCEL
                def _retriever_fn(input_val):
                    query = (
                        input_val.get("question", input_val)
                        if isinstance(input_val, dict)
                        else input_val
                    )
                    if not isinstance(query, str):
                        query = str(query)
                    return self.vector_store.similarity_search(query, k=4)

                retriever = RunnableLambda(_retriever_fn)
                
                # Format documents function - must be wrapped in RunnableLambda for pipe operator
                def format_docs(docs):
                    return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                                      for i, doc in enumerate(docs)])
                
                # Wrap format_docs in RunnableLambda to make it compatible with pipe operator
                # The pipe operator (|) requires both operands to be Runnable objects
                if RunnableLambda:
                    format_docs_runnable = RunnableLambda(format_docs)
                else:
                    # Fallback: create a simple lambda wrapper if RunnableLambda not available
                    # This shouldn't happen if LangChain is properly installed
                    raise RuntimeError("RunnableLambda not available - LangChain core must be installed")
                
                self.rag_chain = (
                    {"context": retriever | format_docs_runnable, "question": RunnablePassthrough()}
                    | ChatPromptTemplate.from_messages([
                        ("system", "Use the following context to answer the question:\n\n{context}"),
                        ("human", "{question}")
                    ])
                    | llm
                    | StrOutputParser()
                )
            else:
                self.rag_chain = None
            
            # Agent executor for tool usage
            tools = self.tool_registry.get_tools()
            if tools and HAS_AGENTS:
                self.agent_executor = self._create_agent_executor(llm, tools)
                
                # Cache the initial agent executor (sync during init)
                if self.agent_executor:
                    tools_hash = hash(tuple(sorted([t.name for t in tools])))
                    cache_key = f"{initial_model}:{tools_hash}"
                    # Direct cache access during initialization (no async needed yet)
                    self._agent_executor_cache.cache[cache_key] = self.agent_executor
                    self.logger.info(f"Cached initial agent executor for {initial_model}")
            else:
                self.agent_executor = None
                if tools:
                    self.logger.warning(f"Tools available but agents not available or no tools registered")
                    
        except Exception as e:
            self.logger.warning(f"Failed to setup chains: {e}", exc_info=True)
            self.rag_chain = None
            self.agent_executor = None
    
    def _create_agent_executor(self, llm, tools):
        """Create agent executor based on LLM type"""
        try:
            # Check if LLM supports function calling (OpenAI-compatible)
            is_openai_compatible = (
                hasattr(llm, 'bind_tools') or 
                hasattr(llm, '_is_openai') or
                'openai' in str(type(llm)).lower() or
                'chatopenai' in str(type(llm)).lower()
            )
            
            if is_openai_compatible and create_openai_functions_agent:
                # Use OpenAI functions agent
                try:
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", """You are a helpful AI assistant with access to tools.
When you need to use a tool, use it. Always provide clear, helpful responses.
If a tool fails, explain what happened and suggest alternatives."""),
                        ("human", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ])
                    
                    agent = create_openai_functions_agent(llm, tools, prompt)
                    executor = AgentExecutor(
                        agent=agent,
                        tools=tools,
                        verbose=self.logger.level <= 10,  # Verbose if debug
                        max_iterations=10,
                        handle_parsing_errors=True,
                        return_intermediate_steps=True,
                    )
                    
                    self.logger.info(f"Created OpenAI functions agent with {len(tools)} tools")
                    return executor
                    
                except ImportError as e:
                    self.logger.warning(f"OpenAI agent creation failed: {e}, trying ReAct")
                    # Fall through to ReAct
                except Exception as e:
                    self.logger.warning(f"OpenAI agent creation error: {e}, trying ReAct")
                    # Fall through to ReAct
            
            # Use ReAct agent for local LLMs or if OpenAI agent failed
            if create_react_agent:
                try:
                    # Try to pull ReAct prompt from hub with timeout, fallback to default
                    # NOTE: For langchain_classic, we should use the default prompt from the hub if available
                    react_prompt = None
                    if HAS_HUB and hub:
                        # Use thread-based timeout to prevent hanging on network requests
                        import threading
                        import queue
                        
                        result_queue = queue.Queue()
                        def pull_prompt():
                            try:
                                prompt = hub.pull("hwchase17/react")
                                result_queue.put(prompt)
                            except Exception as e:
                                self.logger.debug(f"Failed to pull prompt from hub: {e}")
                                result_queue.put(None)
                        
                        thread = threading.Thread(target=pull_prompt, daemon=True)
                        thread.start()
                        thread.join(timeout=2.0)  # 2 second timeout for hub pull
                        
                        if thread.is_alive():
                            self.logger.warning("LangChain Hub pull timed out, using fallback prompt")
                            react_prompt = None
                        elif not result_queue.empty():
                            react_prompt = result_queue.get()
                            if react_prompt:
                                self.logger.info("Using ReAct prompt from LangChain Hub - this should handle agent_scratchpad correctly")
                    
                    # If we still don't have a prompt, try to use custom prompts from prompts folder
                    if not react_prompt:
                        # Try to use vendor fraud prompt if available, otherwise use default
                        # Import ReAct prompt from prompts folder
                        from app.agents.prompts import REACT_AGENT_SYSTEM_PROMPT
                        from langchain_core.prompts import PromptTemplate
                        
                        # Use PromptTemplate instead of ChatPromptTemplate to reduce overhead
                        # The delay is caused by ChatPromptTemplate processing multiple messages
                        react_prompt = PromptTemplate.from_template(
                            REACT_AGENT_SYSTEM_PROMPT + "\n\n{agent_scratchpad}\n\nQuestion: {input}"
                        )
                        self.logger.info("Using ReAct agent system prompt (optimized with PromptTemplate)")
                    
                    # Create ReAct agent
                    self.logger.info(f"Creating ReAct agent with {len(tools)} tools: {[t.name for t in tools]}")
                    agent = create_react_agent(llm, tools, react_prompt)
                    self.logger.info("ReAct agent created successfully")
                    
                    # Custom error handler for parsing errors (helps with models that don't follow format perfectly)
                    def handle_parsing_error(error: Exception) -> str:
                        """Handle parsing errors gracefully"""
                        error_str = str(error)
                        self.logger.warning(f"Agent parsing error: {error_str}")
                        # Return a message that encourages the agent to retry with proper format
                        return "I encountered an error parsing my response. Let me try again with the correct format:\n\nThought: I need to use the correct format.\nAction: [tool_name]\nAction Input: [JSON parameters]"
                    
                    executor = AgentExecutor(
                        agent=agent,
                        tools=tools,
                        verbose=False,  # Disable verbose to reduce overhead
                        max_iterations=10,  # Increased to 10 to allow multi-step tool usage
                        handle_parsing_errors=handle_parsing_error,  # Use custom handler
                        max_execution_time=120.0,  # Increased to 120s to allow model loading (13s) + generation
                        return_intermediate_steps=True,
                    )
                    
                    self.logger.info(f"Created ReAct agent executor with {len(tools)} tools")
                    return executor
                    
                except ImportError as e:
                    self.logger.error(f"Failed to create ReAct agent: {e}")
                    return None
                except Exception as e:
                    self.logger.error(f"ReAct agent creation error: {e}", exc_info=True)
                    return None
            else:
                if not HAS_AGENTS:
                    self.logger.error("LangChain agents not available - AgentExecutor import failed")
                if create_react_agent is None:
                    self.logger.error("create_react_agent is None - import failed. Check LangChain installation.")
                    self.logger.error("Try: pip install langchain langchain-community")
                else:
                    self.logger.warning("ReAct agent creation not available (unknown reason)")
                return None
                
        except Exception as e:
            self.logger.error(f"Agent executor creation failed: {e}", exc_info=True)
            return None
    
    async def process_request(
        self,
        user_input: str,
        user_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        use_rag: bool = True,
        use_tools: bool = True,
        use_langgraph: bool = False,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        stream: bool = False,  # NEW: Enable streaming response to client
        **kwargs
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Process a user request through the full orchestration pipeline
        
        Args:
            user_input: User's input/request
            user_id: Optional user identifier
            workflow_id: Optional workflow ID for state tracking
            use_rag: Whether to use RAG for context
            use_tools: Whether to allow tool usage
            use_langgraph: Whether to use LangGraph multi-agent workflow
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt to prepend
            model: Optional model name to use (overrides default)
            **kwargs: Additional parameters
        
        Returns:
            Response dictionary with result, metadata, and state
        """
        start_time = datetime.now()
        request_id = workflow_id or f"req_{datetime.now().timestamp()}"
        
        # Detailed timing instrumentation
        timings = {
            "total_start": time.time(),
            "rag_ms": 0.0,
            "llm_first_token_ms": 0.0,
            "llm_total_ms": 0.0,
            "cache_hit": False,
        }
        
        # Check response cache first (latency optimization)
        cache_key = self._make_cache_key(user_input, use_tools, use_rag)
        cached_response = await self._get_cached_response(cache_key)
        if cached_response:
            timings["cache_hit"] = True
            total_time = (time.time() - timings["total_start"]) * 1000
            self.logger.info(f"Cache hit for request (latency: {total_time:.1f}ms)")
            return {
                **cached_response,
                "from_cache": True,
                "latency_ms": total_time,
                "timings": timings,
            }
        
        # Track model for LangGraph agent
        # Track model for LangGraph agent (model selection is handled by the graph cache)
        if model:
            self._current_model = model
        
        try:
            # Safety check
            safety_result = self.guardrails.check_prompt(user_input, user_id)
            if self.guardrails.should_block(safety_result):
                return {
                    "success": False,
                    "error": f"Request blocked by safety check: {safety_result.message}",
                    "safety_score": safety_result.score,
                    "request_id": request_id,
                }
            
            # Use LangGraph workflow if requested
            if use_langgraph and self._langgraph_workflow_getter:
                try:
                    # Initialize LangGraph workflow if not already done
                    if not self.langgraph_workflow:
                        self.langgraph_workflow = await self._langgraph_workflow_getter(
                            llm_provider=self.llm_provider,
                            tool_registry=self.tool_registry,
                            vector_store=self.vector_store,
                        )
                    
                    if self.langgraph_workflow:
                        # Execute LangGraph workflow
                        config = {"configurable": {"thread_id": request_id}}
                        result = await self.langgraph_workflow.ainvoke(
                            {
                                "task_description": user_input,
                                "workflow_id": request_id,
                                "context": {"user_id": user_id, **kwargs},
                            },
                            config=config,
                        )
                        
                        # Format result for consistency
                        return {
                            "success": True,
                            "result": result.get("code") or result.get("plan") or result.get("metadata", {}).get("task_breakdown", ""),
                            "metadata": {
                                "workflow_id": result.get("workflow_id"),
                                "task_breakdown": result.get("metadata", {}).get("task_breakdown"),
                                "plan": result.get("plan"),
                                "code": result.get("code"),
                                "method": "langgraph",
                            },
                            "request_id": request_id,
                        }
                except Exception as e:
                    self.logger.warning(f"LangGraph workflow failed: {e}, falling back to sequential", exc_info=True)
                    # Fall through to sequential processing
            
            # OPTIMIZATION: Parallel RAG + LLM Preparation
            # Start RAG retrieval and LLM preparation in parallel to eliminate blocking
            context = ""
            context_docs = []  # Initialize to avoid undefined variable
            rag_task = None
            if use_rag and self.vector_store and user_input and user_input.strip():
                # Start RAG retrieval asynchronously (don't block)
                rag_start = time.time()
                rag_task = asyncio.create_task(
                    self.vector_store.asimilarity_search(user_input, k=4)
                )
                self.logger.debug("RAG retrieval started in parallel with LLM preparation")
            
            # Check LLM provider
            if not self.llm_provider:
                return {
                    "success": False,
                    "error": "LLM provider not initialized. Please configure local LLM (Ollama) or set OPENAI_API_KEY.",
                    "request_id": request_id,
                }
            
            # Await RAG task if it was started (now we need the context)
            if rag_task is not None:
                try:
                    context_docs = await rag_task
                    rag_duration = (time.time() - rag_start) * 1000
                    self.logger.debug(f"RAG retrieval completed in {rag_duration:.0f}ms (was running in parallel)")
                    if context_docs:
                        context = "\n\n".join([doc.page_content for doc in context_docs])
                except Exception as e:
                    self.logger.warning(f"RAG retrieval failed: {e}", exc_info=True)
                    context_docs = []
                    context = ""
            
            # Build prompt
            if context:
                prompt = self.prompt_manager.get_prompt(
                    "rag_qa",
                    question=user_input,
                    context=context,
                )
            else:
                prompt = self.prompt_manager.get_prompt(
                    "general_qa",
                    question=user_input,
                )
            
            # Generate response
            agent_intermediate_steps = []
            response = None

            if use_tools and self._langgraph_agent_available:
                # PRIMARY PATH: LangGraph with native tool calling
                # No ReAct parsing, no AgentExecutor, no agent_scratchpad
                from . import langgraph_agent
                t0 = time.time()
                self.logger.info(f"LangGraph agent: model={model or self._current_model}, tools requested")
                    
                # Resolve model name
                agent_model = model or self._current_model or getattr(
                    getattr(self.llm_provider, 'config', None), 'model_name', 'mistral:7b'
                )

                # Resolve Ollama base URL from the existing provider
                agent_base_url = getattr(self.llm_provider, 'base_url', None)
                if agent_base_url is None:
                    agent_base_url = getattr(
                        getattr(self.llm_provider, 'config', None), 'base_url', None
                    )

                # Get tools from registry
                lc_tools = self.tool_registry.get_tools() if self.tool_registry else []
                if not lc_tools:
                    self.logger.warning("No tools registered, falling back to direct LLM")
                else:
                    # Cache key: model + sorted tool names
                    tool_key = ",".join(sorted(t.name for t in lc_tools))
                    cache_key = f"{agent_model}:{tool_key}"

                    # Latency optimization: reduce num_ctx, set keep_alive, limit max_tokens aggressively
                    from .latency_optimizer import get_optimized_params
                    opt = get_optimized_params(
                        has_tools=bool(lc_tools),  # Tools are available if registered
                        current_max_tokens=max_tokens or 2048
                    )

                    # Always use optimized max_tokens (optimizer now always sets it)
                    optimized_max_tokens = opt.get("max_tokens", 256)  # Fallback to 256 if not set

                    # Get or build graph+PDGE (cached by model+tools, reused across requests)
                    agent_bundle = await self._graph_cache.get(cache_key)
                    if agent_bundle is None:
                        self.logger.info(
                            f"Building new PDGE LangGraph agent for {cache_key} "
                            f"(max_tokens={optimized_max_tokens}, num_ctx={opt['num_ctx']}, keep_alive={opt['keep_alive']})"
                        )
                        # build_agent returns (compiled_graph, pdge_engine)
                        agent_bundle = langgraph_agent.build_agent(
                            model_name=agent_model,
                            tools=lc_tools,
                            base_url=agent_base_url,
                            temperature=kwargs.get("temperature", 0.7),
                            max_tokens=optimized_max_tokens,
                            num_ctx=opt["num_ctx"],
                            keep_alive=opt["keep_alive"],  # Ensures model stays loaded
                        )
                        await self._graph_cache.put(cache_key, agent_bundle)
                        self.logger.info(f"Agent built and cached with keep_alive={opt['keep_alive']} (model will stay loaded)")
                    else:
                        self.logger.debug(f"Reusing cached PDGE LangGraph agent ({cache_key}) - model should already be loaded")
                    
                    # Invoke (pass the full bundle -- invoke unpacks it)
                    input_text = user_input
                    if context:
                        input_text = f"Context:\n{context}\n\n{user_input}"

                    try:
                        # Use streaming by default for faster response (industry best practice)
                        # Streaming starts generation immediately, reducing time-to-first-token
                        llm_start = time.time()
                        stream_result = langgraph_agent.invoke(
                            agent_bundle, input_text,
                            system_prompt=system_prompt,
                            timeout=120.0,
                            stream=True,  # Enable streaming for low latency
                        )
                        
                        # Collect streamed chunks into full response
                        # Even when collecting, streaming is faster than ainvoke because
                        # generation starts immediately rather than waiting for full completion
                        from .streaming_coordinator import StreamChunk
                        response_text = ""
                        tool_calls = []
                        intermediate_steps = []
                        pdge_stats = {}
                        total_latency = 0.0
                        first_token_time = None
                        stream_start = time.time()
                        stream_failed = False
                        
                        try:
                            async for chunk in stream_result:
                                if isinstance(chunk, StreamChunk):
                                    if chunk.type == "token":
                                        if first_token_time is None:
                                            first_token_time = chunk.timestamp_ms
                                            timings["llm_first_token_ms"] = first_token_time
                                            self.logger.info(f"First token received in {first_token_time:.0f}ms (streaming)")
                                        response_text += chunk.content
                                    elif chunk.type == "tool_start":
                                        tool_calls.append({
                                            "tool": chunk.metadata.get("tool_call", {}).get("name", "unknown"),
                                            "input": chunk.metadata.get("tool_call", {}).get("arguments", {}),
                                        })
                                    elif chunk.type == "tool_result":
                                        intermediate_steps.append({
                                            "tool": chunk.metadata.get("tool_name", "unknown"),
                                            "output": chunk.content,
                                        })
                                    elif chunk.type == "error":
                                        self.logger.error(f"Streaming error: {chunk.content}")
                                    total_latency = chunk.timestamp_ms
                            
                            # Get PDGE stats from the agent bundle if available
                            if isinstance(agent_bundle, tuple) and len(agent_bundle) >= 2:
                                pdge_engine = agent_bundle[1]
                                if pdge_engine and hasattr(pdge_engine, 'stats'):
                                    try:
                                        pdge_stats = pdge_engine.stats
                                    except Exception:
                                        pass
                            
                            total_latency = (time.time() - stream_start) * 1000
                            timings["llm_total_ms"] = total_latency
                            
                        except Exception as stream_err:
                            self.logger.error(f"Streaming collection failed: {stream_err}, falling back to non-streaming", exc_info=True)
                            stream_failed = True
                        
                        # Build result dict compatible with non-streaming format
                        if stream_failed:
                            # Fallback to non-streaming if streaming fails
                            result = await langgraph_agent.invoke(
                            agent_bundle, input_text,
                            system_prompt=system_prompt,
                            timeout=120.0,
                            stream=False,  # Fallback to non-streaming
                            )
                        else:
                            result = {
                                "response": response_text,
                                "tool_calls": tool_calls,
                                "intermediate_steps": intermediate_steps,
                                "latency_ms": total_latency,
                                "timed_out": False,
                                "pdge_stats": pdge_stats,
                                "first_token_ms": first_token_time or 0.0,
                            }

                        pdge_stats = result.get("pdge_stats", {})
                        first_token = result.get("first_token_ms", 0.0)
                        self.logger.info(
                            f"LangGraph+PDGE completed in {result['latency_ms']:.0f}ms "
                            f"(first_token={first_token:.0f}ms), "
                            f"tool_calls={len(result['tool_calls'])}, timed_out={result['timed_out']}, "
                            f"pdge={pdge_stats}"
                        )

                        if result["timed_out"]:
                            self.logger.warning("LangGraph timed out, falling back to direct LLM")
                        else:
                            response = result["response"]
                            agent_intermediate_steps = result["intermediate_steps"]

                            # Log tool calls
                            for tc in result["tool_calls"]:
                                self.logger.info(f"Tool called: {tc['tool']}({tc['input']})")
                    
                    except Exception as lang_err:
                        err_msg = str(lang_err)
                        if "does not support tools" in err_msg:
                            self.logger.warning(
                                f"Model {agent_model} does not support native tool calling, "
                                f"falling back to direct LLM (no tools)"
                            )
                            # Invalidate the cached graph for this model
                            await self._graph_cache.put(cache_key, None)
                        else:
                            raise

            # Fallback: direct LLM call (no tools, or tools unavailable, or timeout)
            if response is None:
                if use_tools and not self._langgraph_agent_available:
                    self.logger.warning("LangGraph agent not available, using direct LLM (no tools)")
                    response = await self.llm_provider.ainvoke(prompt)
            
            # Safety check on output
            output_safety = self.guardrails.check_output(response)
            if self.guardrails.should_block(output_safety):
                return {
                    "success": False,
                    "error": f"Response blocked by safety check: {output_safety.message}",
                    "safety_score": output_safety.score,
                    "request_id": request_id,
                }
            
            # Track metrics
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            if self.llm_provider:
                if hasattr(self.llm_provider, 'config'):
                    model_name = self.llm_provider.config.model_name
                elif hasattr(self.llm_provider, 'model'):
                    model_name = self.llm_provider.model
                else:
                    model_name = "unknown"
            else:
                model_name = "unknown"
            self.monitor.record_request(
                request_id=request_id,
                user_id=user_id,
                duration_ms=duration_ms,
                tokens_used=len(response.split()),  # Approximate
                model=model_name,
            )
            
            # Publish inference event to Synapse
            try:
                if not hasattr(self, '_event_publisher'):
                    self._event_publisher = CyrexEventPublisher()
                    await self._event_publisher.connect()
                
                # Get model version (default to "latest" if not available)
                model_version = getattr(self.llm_provider, 'version', 'latest') if self.llm_provider else 'latest'
                
                # Create InferenceEvent and publish it
                try:
                    from deepiri_modelkit.contracts.events import InferenceEvent
                    inference_event = InferenceEvent(
                        event="inference-complete",
                        source="cyrex",
                        model_name=model_name or "unknown",
                        version=model_version or "latest",
                        user_id=user_id or "anonymous",
                        request_id=request_id or "unknown",
                        latency_ms=duration_ms or 0,
                        tokens_used=len(response.split()) if response else 0,
                        cost=0.0,  # Cost not available for local models - use 0.0 instead of None
                        confidence=output_safety.score if hasattr(output_safety, 'score') and output_safety.score is not None else 1.0
                )
                    await self._event_publisher.publish_inference_event(inference_event)
                except Exception as e:
                    self.logger.warning(f"Failed to publish inference event: {e}", exc_info=True)
            except Exception as e:
                self.logger.warning(f"Failed to publish inference event: {e}", exc_info=True)
            
            # Auto-capture interaction for pipeline (Helox training + Cyrex runtime)
            try:
                from .pipeline_auto_capture import get_auto_capture
                auto_capture = await get_auto_capture()
                await auto_capture.capture_interaction(
                    user_input=user_input,
                    response=response,
                    user_id=user_id,
                    session_id=kwargs.get("session_id"),
                    model_name=model_name,
                    duration_ms=duration_ms,
                    context_sources=len(context_docs),
                    safety_scores={
                        "input_score": safety_result.score,
                        "output_score": output_safety.score,
                    },
                    intermediate_steps=agent_intermediate_steps,
                )
            except Exception as e:
                self.logger.debug(f"Auto-capture failed (non-critical): {e}")

            # Cache response for future requests (latency optimization)
            if response:
                await self._cache_response(cache_key, {
                    "success": True,
                    "response": response,
                    "request_id": request_id,
                })
            
            # Calculate total time
            total_time = (time.time() - timings["total_start"]) * 1000
            timings["total_ms"] = total_time
            
            # PHASE 2.1: If streaming requested, yield chunks instead of returning dict
            if stream:
                # Yield streaming chunks to client
                async def stream_chunks():
                    # Yield initial metadata
                    yield {
                        "type": "start",
                        "request_id": request_id,
                        "timings": timings,
                    }
                    
                    # If we have a stream result, yield tokens as they come
                    # Otherwise, yield the full response
                    if response:
                        # Split response into words for streaming effect
                        words = response.split()
                        for i, word in enumerate(words):
                            yield {
                                "type": "token",
                                "content": word + (" " if i < len(words) - 1 else ""),
                                "request_id": request_id,
                            }
                            await asyncio.sleep(0.01)  # Small delay for smooth streaming
                    
                    # Yield final metadata
                    yield {
                        "type": "done",
                        "request_id": request_id,
                        "response": response,
                        "context_sources": len(context_docs) if 'context_docs' in locals() else 0,
                        "duration_ms": duration_ms,
                        "timings": timings,
                        "intermediate_steps": agent_intermediate_steps,
                        "safety_checks": {
                            "input_score": safety_result.score,
                            "output_score": output_safety.score,
                        },
                    }
                
                return stream_chunks()
            
            # Non-streaming: return dict as before
            return {
                "success": True,
                "response": response,
                "request_id": request_id,
                "context_sources": len(context_docs) if 'context_docs' in locals() else 0,
                "duration_ms": duration_ms,
                "timings": timings,  # Include detailed timings
                "intermediate_steps": agent_intermediate_steps,  # Include tool call steps
                "safety_checks": {
                    "input_score": safety_result.score,
                    "output_score": output_safety.score,
                },
            }
        
        except Exception as e:
            self.logger.error(f"Request processing failed: {e}", exc_info=True)
            self.monitor.record_error(request_id, str(e))

            # Auto-capture error recovery for pipeline
            try:
                from .pipeline_auto_capture import get_auto_capture
                auto_capture = await get_auto_capture()
                await auto_capture.capture_error_recovery(
                    error_message=str(e),
                    recovery_response="Request failed — error logged for training.",
                    user_id=user_id,
                    session_id=kwargs.get("session_id"),
                    original_input=user_input,
                )
            except Exception:
                pass  # auto-capture failure is never fatal
            
            return {
                "success": False,
                "error": str(e),
                "request_id": request_id,
            }
        finally:
            pass  # LangGraph agents are cached per model, no restoration needed
    
    async def execute_workflow(
        self,
        workflow_id: str,
        steps: List[Dict[str, Any]],
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a multi-step workflow"""
        return await self.execution_engine.execute_workflow(
            workflow_id,
            steps,
            initial_state,
        )
    
    async def stream_response(
        self,
        user_input: str,
        user_id: Optional[str] = None,
        **kwargs
    ) -> Iterator[str]:
        """Stream response from LLM"""
        # Safety check
        safety_result = self.guardrails.check_prompt(user_input, user_id)
        if self.guardrails.should_block(safety_result):
            yield json.dumps({
                "error": f"Request blocked: {safety_result.message}",
                "safety_score": safety_result.score,
            })
            return
        
        # Get context if needed
        context = ""
        if self.vector_store:
            try:
                docs = await self.vector_store.asimilarity_search(user_input, k=2)
                context = "\n\n".join([doc.page_content for doc in docs])
            except:
                pass
        
        # Check LLM provider
        if not self.llm_provider:
            yield json.dumps({
                "error": "LLM provider not initialized",
                "message": "Please configure local LLM (Ollama) or set OPENAI_API_KEY",
            })
            return
        
        # Build prompt
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {user_input}\n\nAnswer:"
        else:
            prompt = user_input
        
        # Stream response
        try:
            for chunk in self.llm_provider.stream(prompt):
                yield chunk
        except Exception as e:
            self.logger.error(f"Streaming failed: {e}", exc_info=True)
            yield json.dumps({"error": str(e)})
    
    async def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status - optimized for speed with timeouts"""
        import asyncio
        
        # Overall timeout for entire status check (5 seconds max)
        try:
            return await asyncio.wait_for(self._get_status_internal(), timeout=5.0)
        except asyncio.TimeoutError:
            self.logger.warning("Status check timed out after 5 seconds")
            return {
                "llm": {"status": "timeout", "error": "Status check timed out"},
                "vector_store": None,
                "tools": {},
                "monitor": {},
                "queue": None,
                "error": "Status check timed out after 5 seconds"
            }
    
    async def _get_status_internal(self) -> Dict[str, Any]:
        """Internal status check method"""
        import asyncio
        
        status = {
            "llm": {"status": "not_initialized"},
            "vector_store": None,
            "tools": {},
            "monitor": {},
            "queue": None,
        }
        
        # Get LLM status (with timeout protection - run in executor to avoid blocking)
        try:
            if self.llm_provider and hasattr(self.llm_provider, 'health_check'):
                try:
                    # Run synchronous health_check in executor with timeout
                    loop = asyncio.get_event_loop()
                    health_check = await asyncio.wait_for(
                        loop.run_in_executor(None, self.llm_provider.health_check),
                        timeout=2.0
                    )
                    status["llm"] = health_check
                except asyncio.TimeoutError:
                    status["llm"] = {"status": "timeout", "error": "Health check timed out"}
                except Exception as e:
                    status["llm"] = {"status": "error", "error": str(e)}
        except Exception as e:
            self.logger.warning(f"Failed to get LLM status: {e}")
        
        # Get vector store status (with timeout protection - run in executor)
        try:
            if self.vector_store:
                if hasattr(self.vector_store, 'health_check'):
                    try:
                        # Run synchronous health_check in executor with timeout
                        loop = asyncio.get_event_loop()
                        health_check = await asyncio.wait_for(
                            loop.run_in_executor(None, self.vector_store.health_check),
                            timeout=2.0
                        )
                        status["vector_store"] = health_check
                    except asyncio.TimeoutError:
                        # Fallback to stats if health_check times out
                        try:
                            stats = await asyncio.wait_for(
                                loop.run_in_executor(None, self.vector_store.stats) if hasattr(self.vector_store, 'stats') else None,
                                timeout=1.0
                            )
                            status["vector_store"] = stats
                        except Exception:
                            status["vector_store"] = None
                    except Exception as vs_error:
                        error_str = str(vs_error)
                        # Handle gRPC channel errors gracefully - log as debug, not warning
                        if "channel" in error_str.lower() or "grpc" in error_str.lower() or "rpc" in error_str.lower():
                            self.logger.debug(f"gRPC channel error during vector store status check (non-fatal): {vs_error}")
                            status["vector_store"] = {
                                "error": "gRPC channel error",
                                "status": "connection_issue",
                                "note": "Milvus connection channel issue - service may still be functional"
                            }
                        else:
                            self.logger.debug(f"Vector store status check failed: {vs_error}")
                            status["vector_store"] = None
                elif hasattr(self.vector_store, 'stats'):
                    try:
                        loop = asyncio.get_event_loop()
                        status["vector_store"] = await asyncio.wait_for(
                            loop.run_in_executor(None, self.vector_store.stats),
                            timeout=1.0
                        )
                    except Exception as stats_error:
                        error_str = str(stats_error)
                        # Handle gRPC channel errors gracefully
                        if "channel" in error_str.lower() or "grpc" in error_str.lower() or "rpc" in error_str.lower():
                            self.logger.debug(f"gRPC channel error during vector store stats (non-fatal): {stats_error}")
                            status["vector_store"] = {
                                "error": "gRPC channel error",
                                "status": "connection_issue"
                            }
                        else:
                            status["vector_store"] = None
        except Exception as e:
            error_str = str(e)
            # Handle gRPC channel errors gracefully
            if "channel" in error_str.lower() or "grpc" in error_str.lower() or "rpc" in error_str.lower():
                self.logger.debug(f"gRPC channel error during vector store status (non-fatal): {e}")
            else:
                self.logger.warning(f"Failed to get vector store status: {e}")
        
        # Get tool stats (fast, no timeout needed)
        try:
            status["tools"] = self.tool_registry.get_tool_stats()
        except Exception as e:
            self.logger.warning(f"Failed to get tool stats: {e}")
        
        # Get monitor stats (fast, no timeout needed)
        try:
            status["monitor"] = self.monitor.get_stats()
        except Exception as e:
            self.logger.warning(f"Failed to get monitor stats: {e}")
        
        # Add queue stats if available (with timeout)
        try:
            if hasattr(self.queue_manager, 'get_queue_stats'):
                try:
                    # Call the method and check if result is async
                    queue_stats_method = self.queue_manager.get_queue_stats
                    if asyncio.iscoroutinefunction(queue_stats_method):
                        queue_stats = await asyncio.wait_for(queue_stats_method(), timeout=1.0)
                    else:
                        # Run synchronous method in executor
                        loop = asyncio.get_event_loop()
                        queue_stats = await asyncio.wait_for(
                            loop.run_in_executor(None, queue_stats_method),
                            timeout=1.0
                        )
                    status["queue"] = queue_stats
                except asyncio.TimeoutError:
                    status["queue"] = None
                except Exception:
                    status["queue"] = None
        except Exception:
            status["queue"] = None
        
        return status
    
    def _make_cache_key(self, user_input: str, use_tools: bool, use_rag: bool) -> str:
        """Create cache key for response caching."""
        normalized = user_input.strip().lower()
        key_str = f"{normalized}:{use_tools}:{use_rag}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    async def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired."""
        async with self._response_cache_lock:
            if cache_key in self._response_cache:
                result, timestamp = self._response_cache[cache_key]
                if (time.time() - timestamp) < self._response_cache_ttl:
                    # Move to end (LRU)
                    self._response_cache.move_to_end(cache_key)
                    return result
                else:
                    # Expired
                    del self._response_cache[cache_key]
        return None
    
    async def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """Cache response for future requests."""
        async with self._response_cache_lock:
            self._response_cache[cache_key] = (response, time.time())
            self._response_cache.move_to_end(cache_key)
            # Evict oldest if over limit
            if len(self._response_cache) > self._response_cache_max_size:
                self._response_cache.popitem(last=False)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring and optimization
        
        Returns:
            Dictionary with LLM and agent executor cache statistics
        """
        async with self._response_cache_lock:
            response_cache_size = len(self._response_cache)
        return {
            "graph_cache": self._graph_cache.get_stats(),
            "llm_cache": self._llm_cache.get_stats(),
            "agent_executor_cache": self._agent_executor_cache.get_stats(),
            "response_cache": {
                "size": response_cache_size,
                "max_size": self._response_cache_max_size,
                "ttl_seconds": self._response_cache_ttl,
            },
            "current_model": self._current_model,
            "langgraph_agent_available": self._langgraph_agent_available,
        }
    
    def _make_cache_key(self, user_input: str, use_tools: bool, use_rag: bool) -> str:
        """Create cache key for response caching."""
        normalized = user_input.strip().lower()
        key_str = f"{normalized}:{use_tools}:{use_rag}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    async def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired."""
        async with self._response_cache_lock:
            if cache_key in self._response_cache:
                result, timestamp = self._response_cache[cache_key]
                if (time.time() - timestamp) < self._response_cache_ttl:
                    # Move to end (LRU)
                    self._response_cache.move_to_end(cache_key)
                    return result
                else:
                    # Expired
                    del self._response_cache[cache_key]
        return None
    
    async def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """Cache response for future requests."""
        async with self._response_cache_lock:
            self._response_cache[cache_key] = (response, time.time())
            self._response_cache.move_to_end(cache_key)
            # Evict oldest if over limit
            if len(self._response_cache) > self._response_cache_max_size:
                self._response_cache.popitem(last=False)
    
    async def clear_caches(self) -> Dict[str, str]:
        """
        Clear all caches (useful for debugging or memory management)
        
        Returns:
            Status message
        """
        await self._graph_cache.clear()
        await self._llm_cache.clear()
        await self._agent_executor_cache.clear()
        async with self._response_cache_lock:
            self._response_cache.clear()
        self._current_model = None
        self.logger.info("All caches cleared")
        return {"status": "success", "message": "All caches cleared"}
    
    async def warm_models(self, model_names: List[str]) -> Dict[str, Any]:
        """
        Pre-warm multiple models by loading them into cache
        This can be called on startup or during idle time to improve first-request latency
        
        Args:
            model_names: List of model names to pre-load
        
        Returns:
            Dictionary with warming results
        """
        results = {}
        for model_name in model_names:
            try:
                # Check if already cached
                cached = await self._llm_cache.get(model_name)
                if cached:
                    results[model_name] = {"status": "already_cached", "time_ms": 0}
                    continue
                
                # Load and cache the model
                start = time.time()
                if self.llm_provider and hasattr(self.llm_provider, 'config'):
                    original = self.llm_provider.config.model_name
                    self.llm_provider.config.model_name = model_name
                    if hasattr(self.llm_provider, '_initialize_llm'):
                        self.llm_provider._initialize_llm()
                    await self._llm_cache.put(model_name, self.llm_provider.llm)
                    # Restore original model
                    self.llm_provider.config.model_name = original
                    if hasattr(self.llm_provider, '_initialize_llm'):
                        self.llm_provider._initialize_llm()
                    
                    duration_ms = (time.time() - start) * 1000
                    results[model_name] = {"status": "warmed", "time_ms": duration_ms}
                    self.logger.info(f"Pre-warmed model {model_name} in {duration_ms:.0f}ms")
                else:
                    results[model_name] = {"status": "failed", "error": "LLM provider not available"}
            except Exception as e:
                results[model_name] = {"status": "failed", "error": str(e)}
                self.logger.error(f"Failed to warm model {model_name}: {e}")
        
        return {
            "warmed": len([r for r in results.values() if r["status"] == "warmed"]),
            "already_cached": len([r for r in results.values() if r["status"] == "already_cached"]),
            "failed": len([r for r in results.values() if r["status"] == "failed"]),
            "details": results,
            "cache_stats": await self.get_cache_stats()
        }


# Singleton orchestrator instance
_orchestrator = None

def get_orchestrator() -> WorkflowOrchestrator:
    """Get global orchestrator instance (singleton)"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = WorkflowOrchestrator()
    return _orchestrator

