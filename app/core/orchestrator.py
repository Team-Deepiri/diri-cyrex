"""
Workflow Orchestrator
Main orchestration engine that coordinates all components
Integrates LangChain, local LLMs, RAG, tools, and state management
"""
from typing import Dict, List, Optional, Any, Iterator, Union
import json
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
    try:
        from langchain.agents import create_openai_functions_agent, create_react_agent
    except ImportError:
        # Try alternative paths
        try:
            from langchain.agents.openai_functions import create_openai_functions_agent
        except ImportError:
            pass
        try:
            from langchain.agents.react import create_react_agent
        except ImportError:
            pass
    
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
                from langchain.tools import Tool
                
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
                retriever = self.vector_store.get_retriever(k=4)
                
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
                    
                    if not react_prompt:
                        # Fallback prompt - use this by default to avoid network calls
                        react_prompt = ChatPromptTemplate.from_messages([
                            ("system", """You are a helpful AI assistant with access to tools.
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""),
                            ("human", "{input}"),
                            MessagesPlaceholder(variable_name="agent_scratchpad"),
                        ])
                    
                    agent = create_react_agent(llm, tools, react_prompt)
                    executor = AgentExecutor(
                        agent=agent,
                        tools=tools,
                        verbose=self.logger.level <= 10,
                        max_iterations=10,
                        handle_parsing_errors=True,
                        return_intermediate_steps=True,
                    )
                    
                    self.logger.info(f"Created ReAct agent with {len(tools)} tools")
                    return executor
                    
                except ImportError as e:
                    self.logger.error(f"Failed to create ReAct agent: {e}")
                    return None
                except Exception as e:
                    self.logger.error(f"ReAct agent creation error: {e}", exc_info=True)
                    return None
            else:
                self.logger.warning("ReAct agent creation not available")
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
            **kwargs: Additional parameters
        
        Returns:
            Response dictionary with result, metadata, and state
        """
        start_time = datetime.now()
        request_id = workflow_id or f"req_{datetime.now().timestamp()}"
        
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
            if use_tools and self.agent_executor:
                try:
                    # Use agent executor for tool-using requests
                    self.logger.info(f"Using agent executor with tools for request")
                    
                    # Prepare input for agent (include context if available)
                    agent_input = {
                        "input": f"Context:\n{context}\n\nQuestion: {user_input}\n\nAnswer:" if context else user_input
                    }
                    
                    # Execute agent (using ainvoke instead of deprecated arun)
                    result = await self.agent_executor.ainvoke(agent_input)
                    
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
                    
                    # Log tool usage
                    intermediate_steps = result.get("intermediate_steps", []) if isinstance(result, dict) else []
                    tool_calls = len(intermediate_steps)
                    if tool_calls > 0:
                        self.logger.info(f"Agent executor completed with {tool_calls} tool calls")
                    
                except Exception as e:
                    self.logger.warning(f"Agent executor failed: {e}, falling back to direct LLM", exc_info=True)
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
                
                await self._event_publisher.publish_inference(
                    model_name=model_name,
                    version=model_version,
                    latency_ms=duration_ms,
                    user_id=user_id,
                    request_id=request_id,
                    tokens_used=len(response.split()),
                    confidence=output_safety.score if hasattr(output_safety, 'score') else None
                )
            except Exception as e:
                self.logger.warning(f"Failed to publish inference event: {e}", exc_info=True)
            
            return {
                "success": True,
                "response": response,
                "request_id": request_id,
                "context_sources": len(context_docs),
                "duration_ms": duration_ms,
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

