"""
Workflow Orchestrator
Main orchestration engine that coordinates all components
Integrates LangChain, local LLMs, RAG, tools, and state management
"""
from typing import Dict, List, Optional, Any, Iterator, Union
import json
import time
from datetime import datetime
from ..logging_config import get_logger

logger = get_logger("cyrex.orchestrator")

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
        
        # Initialize LangGraph workflow (optional)
        self.langgraph_workflow = None
        try:
            from .langgraph_workflow import get_langgraph_workflow
            # Will be initialized lazily when needed
            self._langgraph_workflow_getter = get_langgraph_workflow
        except ImportError:
            self.logger.debug("LangGraph workflow not available")
            self._langgraph_workflow_getter = None
        
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
                def knowledge_retrieval_func(input_data):
                    """Sync wrapper for knowledge retrieval"""
                    # Handle both dict and string inputs
                    if isinstance(input_data, str):
                        input_data = {"query": input_data}
                    elif not isinstance(input_data, dict):
                        input_data = {"query": str(input_data)}
                    
                    # Run async function
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    return loop.run_until_complete(knowledge_retrieval_func_async(input_data))
                
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
    
    def _setup_chains(self):
        """Setup LangChain chains for different workflows"""
        if not self.llm_provider:
            self.logger.warning("LLM provider not available, chains not initialized")
            self.rag_chain = None
            self.agent_executor = None
            return
        
        try:
            llm = self.llm_provider.get_llm()
            
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
                        max_execution_time=30.0,  # Reduced to 30s to fail faster if hanging
                        return_intermediate_steps=True,
                    )
                    
                    self.logger.info(f"✅ Created ReAct agent executor with {len(tools)} tools")
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
        **kwargs
    ) -> Dict[str, Any]:
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
        
        # Temporarily switch model if specified
        original_model = None
        if model and self.llm_provider and hasattr(self.llm_provider, 'config'):
            original_model = self.llm_provider.config.model_name
            self.llm_provider.config.model_name = model
            # Re-initialize LLM with new model
            if hasattr(self.llm_provider, '_initialize_llm'):
                self.llm_provider._initialize_llm()
            self.logger.info(f"Switched model from {original_model} to {model} for this request")
        
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
            
            # Retrieve context if RAG enabled and user_input is not empty
            context_docs = []
            if use_rag and self.vector_store and user_input and user_input.strip():
                try:
                    context_docs = await self.vector_store.asimilarity_search(user_input, k=4)
                    if context_docs:
                        context = "\n\n".join([doc.page_content for doc in context_docs])
                        self.logger.debug(f"Retrieved {len(context_docs)} documents for RAG context")
                    else:
                        # Empty collection - no documents indexed yet (this is fine)
                        context = ""
                        self.logger.debug("No documents found in vector store (collection may be empty - RAG disabled for this request)")
                except Exception as e:
                    self.logger.warning(f"RAG retrieval failed: {e}")
                    context = ""
            else:
                context = ""
            
            # Check LLM provider
            if not self.llm_provider:
                return {
                    "success": False,
                    "error": "LLM provider not initialized. Please configure local LLM (Ollama) or set OPENAI_API_KEY.",
                    "request_id": request_id,
                }
            
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
            
            # Generate response - use agent executor if tools are requested
            agent_intermediate_steps = []  # Initialize to empty list
            if use_tools and self.agent_executor:
                try:
                    # Use agent executor for tool-using requests
                    self.logger.info(f"Using agent executor with tools for request")
                    
                    # Prepare input for agent (keep it SIMPLE to reduce overhead)
                    # The system prompt is already in the ReAct prompt template, no need to repeat it here
                    input_text = user_input
                    if context and not system_prompt:
                        # Only add context if there's no system prompt (to keep input short)
                        input_text = f"Context:\n{context}\n\nQuestion: {user_input}"
                    
                    # Agent input - AgentExecutor will handle agent_scratchpad internally
                    # Passing it explicitly causes expensive processing overhead
                    agent_input = {
                        "input": input_text
                        # Note: agent_scratchpad removed - let AgentExecutor handle it to avoid 35s delay
                    }
                    
                    self.logger.info(f"Agent input prepared with system_prompt={system_prompt is not None}, context={context is not None}")
                    self.logger.debug(f"Agent input: {agent_input}, keys: {list(agent_input.keys())}, agent_scratchpad type: {type(agent_input.get('agent_scratchpad'))}")
                    
                    # Execute agent (using ainvoke instead of deprecated arun)
                    # Set a timeout to prevent hanging
                    import asyncio
                    from datetime import datetime as dt
                    
                    invoke_start = dt.now()
                    self.logger.info(f"⏱️ Starting agent_executor.ainvoke at {invoke_start.strftime('%H:%M:%S.%f')}")
                    
                    # Log input size for diagnostics
                    input_str = str(agent_input.get('input', ''))
                    tool_count = len(self.agent_executor.tools) if hasattr(self.agent_executor, 'tools') else 0
                    self.logger.info(f"⏱️ Input length: {len(input_str)} chars, Tools available: {tool_count}")
                    
                    # Add detailed timing to find bottleneck
                    t0 = time.time()
                    self.logger.info(f"⏱️ [t+0.00s] About to call agent_executor.ainvoke")
                    
                    try:
                        # Wrap the agent executor call to log when it actually starts processing
                        self.logger.info(f"⏱️ [t+{time.time()-t0:.2f}s] Entering agent_executor.ainvoke")
                        result = await asyncio.wait_for(
                            self.agent_executor.ainvoke(agent_input),
                            timeout=60.0  # Allow time for model loading (13s) + generation
                        )
                        self.logger.info(f"⏱️ [t+{time.time()-t0:.2f}s] agent_executor.ainvoke returned")
                        
                        invoke_end = dt.now()
                        invoke_duration = (invoke_end - invoke_start).total_seconds()
                        self.logger.info(f"⏱️ agent_executor.ainvoke completed in {invoke_duration:.2f}s")
                    except asyncio.TimeoutError:
                        self.logger.error("Agent executor timed out after 60 seconds - model may not be following ReAct format or executor is hanging")
                        # Fallback to direct LLM call
                        self.logger.warning("Falling back to direct LLM call without agent executor")
                        response = await self.llm_provider.ainvoke(input_text)
                        return {
                            "success": True,
                            "response": response,
                            "request_id": request_id,
                            "metadata": {
                                "method": "direct_llm_fallback",
                                "reason": "agent_executor_timeout"
                            }
                        }
                    except OutputParserException as e:
                        self.logger.error(f"Agent output parsing failed: {e}")
                        self.logger.error(f"Raw model output: {e.llm_output if hasattr(e, 'llm_output') else 'N/A'}")
                        # Try to extract any usable response
                        error_msg = str(e)
                        if "Final Answer:" in error_msg:
                            # Extract the final answer from the error
                            try:
                                answer = error_msg.split("Final Answer:")[-1].strip()
                                self.logger.info(f"Extracted answer from parsing error: {answer}")
                                return {
                                    "success": True,
                                    "response": answer,
                                    "request_id": request_id,
                                    "metadata": {
                                        "method": "extracted_from_error",
                                        "parsing_error": str(e)
                                    }
                                }
                            except:
                                pass
                        # Fallback to direct LLM
                        self.logger.warning("Falling back to direct LLM call due to parsing error")
                        response = await self.llm_provider.ainvoke(input_text)
                        return {
                            "success": True,
                            "response": response,
                            "request_id": request_id,
                            "metadata": {
                                "method": "direct_llm_fallback",
                                "reason": "parsing_error",
                                "error": str(e)
                            }
                        }
                    except ValueError as e:
                        if "agent_scratchpad" in str(e):
                            self.logger.error(f"Agent executor agent_scratchpad error: {e}")
                            self.logger.error(f"Agent input was: {agent_input}")
                            # The AgentExecutor from langchain_classic has a bug where it converts agent_scratchpad to string
                            # Try to work around it by not passing agent_scratchpad at all and letting it handle it
                            self.logger.warning("Retrying without agent_scratchpad in input - letting AgentExecutor handle it internally")
                            agent_input_no_scratchpad = {"input": input_text}
                            try:
                                result = await self.agent_executor.ainvoke(agent_input_no_scratchpad)
                            except Exception as e2:
                                self.logger.error(f"Retry also failed: {e2}")
                                raise ValueError(f"Agent executor agent_scratchpad format error. This is a known compatibility issue with langchain_classic. Error: {e}")
                        else:
                            raise
                    
                    # Log raw agent result for debugging
                    self.logger.info(f"Agent executor returned: {type(result)} with keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
                    if isinstance(result, dict):
                        self.logger.debug(f"Agent output: {result.get('output', 'NO OUTPUT KEY')}")
                        # Log ALL keys to see what we're getting
                        self.logger.debug(f"Full result keys: {list(result.keys())}")
                        if "intermediate_steps" in result:
                            self.logger.info(f"Found intermediate_steps in result: {len(result['intermediate_steps'])} steps")
                    
                    # Extract response from agent result
                    if isinstance(result, dict):
                        if "output" in result:
                            response = result["output"]
                        elif "answer" in result:
                            response = result["answer"]
                        else:
                            response = str(result)
                    else:
                        response = str(result)
                    
                    # Extract intermediate_steps - CRITICAL for tool call tracking
                    intermediate_steps = result.get("intermediate_steps", []) if isinstance(result, dict) else []
                    # Update agent_intermediate_steps for return value
                    agent_intermediate_steps = intermediate_steps
                    
                    # Log tool usage
                    tool_calls = len(intermediate_steps)
                    if tool_calls > 0:
                        self.logger.info(f"Agent executor completed with {tool_calls} tool calls")
                        for i, step in enumerate(intermediate_steps):
                            if isinstance(step, tuple) and len(step) >= 2:
                                action = step[0]
                                observation = step[1]
                                tool_name = getattr(action, 'tool', 'unknown') if hasattr(action, 'tool') else 'unknown'
                                self.logger.info(f"Tool call {i+1}: {tool_name} -> {str(observation)[:100]}")
                    else:
                        self.logger.warning(f"Agent executor completed but made NO tool calls. Response: {response[:200]}")
                        self.logger.warning(f"Full result keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
                        # Log the full result structure for debugging
                        if isinstance(result, dict):
                            self.logger.debug(f"Result structure: {list(result.keys())}")
                            if "intermediate_steps" in result:
                                self.logger.debug(f"Intermediate steps type: {type(result['intermediate_steps'])}, length: {len(result.get('intermediate_steps', []))}")
                        # This is a known issue with llama3:8b - it may not follow ReAct format well
                        self.logger.warning("NOTE: llama3:8b may not reliably follow ReAct format. Consider using a model better at tool calling (e.g., mistral:7b, qwen2.5:7b, or llama3.1:8b)")
                    
                except Exception as e:
                    self.logger.error(f"Agent executor failed: {e}", exc_info=True)
                    self.logger.warning(f"Falling back to direct LLM (tools will not be available)")
                    # Fallback to direct LLM call
                    # If max_tokens is specified, temporarily update config
                    if max_tokens and hasattr(self.llm_provider, 'config'):
                        original_max = self.llm_provider.config.max_tokens
                        self.llm_provider.config.max_tokens = max_tokens
                        # Reinitialize with new max_tokens
                        if hasattr(self.llm_provider, '_initialize_llm'):
                            self.llm_provider._initialize_llm()
                        response = await self.llm_provider.ainvoke(prompt)
                        # Restore original
                        self.llm_provider.config.max_tokens = original_max
                        if hasattr(self.llm_provider, '_initialize_llm'):
                            self.llm_provider._initialize_llm()
                    else:
                        response = await self.llm_provider.ainvoke(prompt)
            else:
                # Direct LLM call (no tools or agent not available)
                # If max_tokens is specified, temporarily update config
                if max_tokens and hasattr(self.llm_provider, 'config'):
                    original_max = self.llm_provider.config.max_tokens
                    self.llm_provider.config.max_tokens = max_tokens
                    # Reinitialize with new max_tokens
                    if hasattr(self.llm_provider, '_initialize_llm'):
                        self.llm_provider._initialize_llm()
                    response = await self.llm_provider.ainvoke(prompt)
                    # Restore original
                    self.llm_provider.config.max_tokens = original_max
                    if hasattr(self.llm_provider, '_initialize_llm'):
                        self.llm_provider._initialize_llm()
                else:
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
            
            return {
                "success": True,
                "response": response,
                "request_id": request_id,
                "context_sources": len(context_docs),
                "duration_ms": duration_ms,
                "intermediate_steps": agent_intermediate_steps,  # Include tool call steps
                "safety_checks": {
                    "input_score": safety_result.score,
                    "output_score": output_safety.score,
                },
            }
        
        except Exception as e:
            self.logger.error(f"Request processing failed: {e}", exc_info=True)
            self.monitor.record_error(request_id, str(e))
            
            return {
                "success": False,
                "error": str(e),
                "request_id": request_id,
            }
        finally:
            # Restore original model if it was changed
            if original_model and self.llm_provider and hasattr(self.llm_provider, 'config'):
                self.llm_provider.config.model_name = original_model
                if hasattr(self.llm_provider, '_initialize_llm'):
                    self.llm_provider._initialize_llm()
                self.logger.info(f"Restored model to {original_model}")
    
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


# Singleton orchestrator instance
_orchestrator = None

def get_orchestrator() -> WorkflowOrchestrator:
    """Get global orchestrator instance (singleton)"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = WorkflowOrchestrator()
    return _orchestrator

