"""
Workflow Orchestrator
Main orchestration engine that coordinates all components
Integrates LangChain, local LLMs, RAG, tools, and state management
"""
from typing import Dict, List, Optional, Any, Iterator
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.utils import Input, Output
import json
from datetime import datetime
from ..logging_config import get_logger
from .execution_engine import TaskExecutionEngine, get_execution_engine
from .state_manager import WorkflowStateManager, get_state_manager
from .tool_registry import ToolRegistry, get_tool_registry
from .guardrails import SafetyGuardrails, get_guardrails
from .queue_manager import TaskQueueManager, get_queue_manager, TaskPriority
from .monitoring import SystemMonitor, get_monitor
from .prompt_manager import PromptVersionManager, get_prompt_manager
from ..integrations.local_llm import LocalLLMProvider, get_local_llm, LLMBackend
from ..integrations.openai_wrapper import OpenAIProvider, get_openai_provider
from typing import Union
from ..integrations.milvus_store import MilvusVectorStore, get_milvus_store
from ..integrations.rag_bridge import RAGBridge, get_rag_bridge
from ..services.knowledge_retrieval_engine import KnowledgeRetrievalEngine
from ..settings import settings

logger = get_logger("cyrex.orchestrator")


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
        
        # Initialize chains
        self._setup_chains()
    
    def _setup_chains(self):
        """Setup LangChain chains for different workflows"""
        if not self.llm_provider:
            self.logger.warning("LLM provider not available, chains not initialized")
            self.rag_chain = None
            self.tool_chain = None
            return
        
        try:
            llm = self.llm_provider.get_llm()
            
            # RAG chain
            if self.vector_store:
                retriever = self.vector_store.get_retriever(k=4)
                
                self.rag_chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | ChatPromptTemplate.from_messages([
                        ("system", "Use the following context to answer the question:\n\n{context}"),
                        ("human", "{question}")
                    ])
                    | llm
                    | StrOutputParser()
                )
            else:
                self.rag_chain = None
        except Exception as e:
            self.logger.warning(f"Failed to setup chains: {e}")
            self.rag_chain = None
        
        # Tool-using chain
        tools = self.tool_registry.get_tools()
        if tools:
            # Create tool-using chain with LangChain's agent executor
            from langchain.agents import create_openai_functions_agent, AgentExecutor
            from langchain_core.prompts import ChatPromptTemplate
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant with access to tools."),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # Note: This requires OpenAI-compatible LLM
            # For local models, we'll use a simpler approach
            self.tool_chain = None  # Will be set up per-request if needed
        else:
            self.tool_chain = None
    
    async def process_request(
        self,
        user_input: str,
        user_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        use_rag: bool = True,
        use_tools: bool = True,
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
            
            # Retrieve context if RAG enabled
            context_docs = []
            if use_rag and self.vector_store:
                try:
                    context_docs = await self.vector_store.asimilarity_search(user_input, k=4)
                    context = "\n\n".join([doc.page_content for doc in context_docs])
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
            
            # Generate response
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
        """Get orchestrator status"""
        status = {
            "llm": self.llm_provider.health_check() if self.llm_provider else {"status": "not_initialized"},
            "vector_store": self.vector_store.stats() if self.vector_store else None,
            "tools": self.tool_registry.get_tool_stats(),
            "monitor": self.monitor.get_stats(),
        }
        
        # Add queue stats if available
        try:
            if hasattr(self.queue_manager, 'get_queue_stats'):
                status["queue"] = await self.queue_manager.get_queue_stats()
        except:
            status["queue"] = None
        
        return status


def get_orchestrator() -> WorkflowOrchestrator:
    """Get global orchestrator instance"""
    return WorkflowOrchestrator()

