"""
Orchestration API Routes
Exposes the workflow orchestrator via REST API
"""
import os
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, AsyncIterator
import json
from ..core.orchestrator import WorkflowOrchestrator, get_orchestrator
from ..core.state_manager import StateStatus
from ..logging_config import get_logger

logger = get_logger("cyrex.orchestration_api")

router = APIRouter(prefix="/orchestration", tags=["orchestration"])


class ProcessRequestInput(BaseModel):
    """Input for process request"""
    user_input: str = Field(..., min_length=1, description="User's input/request")
    user_id: Optional[str] = Field(None, description="User identifier")
    workflow_id: Optional[str] = Field(None, description="Workflow ID for state tracking")
    use_rag: bool = Field(True, description="Whether to use RAG for context")
    use_tools: bool = Field(True, description="Whether to allow tool usage")
    use_langgraph: bool = Field(False, description="Whether to use LangGraph multi-agent workflow (task → plan → code)")
    force_local_llm: bool = Field(False, description="Force use of local LLM instead of OpenAI")
    llm_backend: Optional[str] = Field(None, description="Local LLM backend (ollama, llama_cpp, transformers)")
    llm_model: Optional[str] = Field(None, description="Local LLM model name (e.g., mistral:7b)")
    llm_base_url: Optional[str] = Field(None, description="Local LLM base URL (e.g., http://ollama:11434)")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate (default: 500 for local LLM, 2000 for OpenAI)")
    request_timeout: Optional[float] = Field(None, description="Request timeout in seconds (default: uses LOCAL_LLM_TIMEOUT setting)")
    stream: bool = Field(False, description="Enable streaming response (returns tokens as they're generated)")
    priority: Optional[str] = Field("normal", description="Request priority: critical, high, normal, low, batch")
    enable_batching: bool = Field(False, description="Enable request batching (groups similar requests together)")


class WorkflowStep(BaseModel):
    """Workflow step definition"""
    name: str
    tool: Optional[str] = None
    input: Dict[str, Any] = Field(default_factory=dict)


class ExecuteWorkflowInput(BaseModel):
    """Input for workflow execution"""
    workflow_id: str
    steps: List[WorkflowStep]
    initial_state: Optional[Dict[str, Any]] = Field(default_factory=dict)


@router.post("/process")
async def process_request(
    input: ProcessRequestInput,
    request: Request,
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator),
):
    """Process a user request through the orchestration pipeline"""
    try:
        # If force_local_llm is True, create a temporary orchestrator with local LLM only
        if input.force_local_llm:
            from ..integrations.local_llm import get_local_llm, LLMBackend
            from ..settings import settings
            
            backend_str = input.llm_backend or settings.LOCAL_LLM_BACKEND
            model_str = input.llm_model or settings.LOCAL_LLM_MODEL
            
            logger.info(f"Force local LLM requested: backend={backend_str}, model={model_str}, use_tools={input.use_tools}")
            
            try:
                backend = LLMBackend(backend_str)
                # Pass max_tokens if provided, otherwise use default (200 for local LLM on CPU - faster responses)
                max_tokens_for_llm = input.max_tokens if input.max_tokens else 200
                # Use provided base_url or fall back to settings
                base_url = input.llm_base_url
                if not base_url and backend == LLMBackend.OLLAMA:
                    base_url = settings.OLLAMA_BASE_URL
                
                logger.info(f"Initializing local LLM: backend={backend.value}, model={model_str}, base_url={base_url}")
                local_llm = get_local_llm(
                    backend=backend.value,
                    model_name=model_str,
                    base_url=base_url,
                    max_tokens=max_tokens_for_llm,
                )
                
                if local_llm:
                    # Create a temporary orchestrator with local LLM only
                    # Note: We'll try to use it even if health check failed - actual invocation will handle errors
                    try:
                        # Import here to avoid circular imports
                        import asyncio
                        
                        logger.info(f"Creating temporary orchestrator with local LLM (use_tools={input.use_tools})")
                        temp_orchestrator = WorkflowOrchestrator(llm_provider=local_llm)
                        
                        logger.info(f"Processing request with local LLM: user_input length={len(input.user_input)}, max_tokens={max_tokens_for_llm}")
                        # Use request timeout if provided, otherwise use settings value
                        request_timeout = input.request_timeout if input.request_timeout is not None else float(settings.LOCAL_LLM_TIMEOUT)
                        logger.info(f"Using timeout: {request_timeout}s for local LLM request")
                        # Add timeout protection for local LLM requests
                        result = await asyncio.wait_for(
                            temp_orchestrator.process_request(
                                user_input=input.user_input,
                                user_id=input.user_id,
                                workflow_id=input.workflow_id,
                                use_rag=input.use_rag,
                                use_tools=input.use_tools,
                                use_langgraph=input.use_langgraph,
                                max_tokens=max_tokens_for_llm,
                            ),
                            timeout=request_timeout
                        )
                        logger.info(f"Local LLM request completed successfully")
                        # Check if result indicates failure
                        if isinstance(result, dict) and result.get("success") is False:
                            error_msg = result.get("error", "Unknown error")
                            raise HTTPException(
                                status_code=500,
                                detail=error_msg
                        )
                        return result
                    except TimeoutError as e:
                        request_timeout = input.request_timeout if input.request_timeout is not None else float(settings.LOCAL_LLM_TIMEOUT)
                        raise HTTPException(
                            status_code=504,
                            detail=f"Local LLM ({backend_str}) request timed out after {request_timeout} seconds. The model may be too slow or unresponsive. Try reducing max_tokens, increasing request_timeout, or using a faster model. Error: {str(e)}"
                        )
                    except Exception as e:
                        error_msg = str(e)
                        error_lower = error_msg.lower()
                        
                        # Check for specific error types
                        if "404" in error_msg or "not found" in error_lower or "model" in error_lower and "not found" in error_lower:
                            raise HTTPException(
                                status_code=404,
                                detail=f"Model '{input.llm_model or 'mistral:7b'}' not found in Ollama. Please pull the model first: docker exec -it deepiri-ollama-dev ollama pull {input.llm_model or 'mistral:7b'}"
                            )
                        elif "Connection" in error_msg or "connect" in error_lower or "timeout" in error_lower:
                            raise HTTPException(
                                status_code=503,
                                detail=f"Local LLM ({backend_str}) connection failed. Make sure Ollama is running at the configured URL. Error: {error_msg}"
                            )
                        else:
                            raise HTTPException(
                                status_code=500,
                                detail=f"Local LLM ({backend_str}) error: {error_msg}"
                            )
                else:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Local LLM ({backend_str}) could not be initialized. Check logs for details."
                    )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid LLM backend: {e}")
        
        # Use default orchestrator (OpenAI preferred, local LLM fallback)
        # PHASE 2.2: Request Queuing and Batching
        queue_manager = get_request_queue_manager()
        
        # Determine request priority
        priority_map = {
            "critical": RequestPriority.CRITICAL,
            "high": RequestPriority.HIGH,
            "normal": RequestPriority.NORMAL,
            "low": RequestPriority.LOW,
            "batch": RequestPriority.BATCH,
        }
        priority = priority_map.get(input.priority.lower(), RequestPriority.NORMAL)
        
        # Override priority if batching is explicitly enabled
        if input.enable_batching:
            priority = RequestPriority.BATCH
        
        # Create queued request
        import uuid
        request_id = input.workflow_id or f"req_{uuid.uuid4().hex[:8]}"
        queued_request = QueuedRequest(
            request_id=request_id,
            user_input=input.user_input,
            user_id=input.user_id,
            workflow_id=input.workflow_id,
            use_rag=input.use_rag,
            use_tools=input.use_tools,
            use_langgraph=input.use_langgraph,
            max_tokens=input.max_tokens,
            model=input.llm_model,
            stream=input.stream,
            priority=priority,
        )
        
        # Define processor function
        async def process_queued_request(req: QueuedRequest) -> Dict[str, Any]:
            """Process a queued request"""
            # Calculate progressive timeout
            base_timeout = 30.0
            if req.use_tools:
                base_timeout += 30.0
            if req.use_rag:
                base_timeout += 20.0
            if len(req.user_input) > 500:
                base_timeout += 20.0
            progressive_timeout = min(base_timeout, 120.0)
            
            # Process request
            if req.stream:
                # Streaming - return async iterator
                stream_result = orchestrator.process_request(
                    user_input=req.user_input,
                    user_id=req.user_id,
                    workflow_id=req.workflow_id,
                    use_rag=req.use_rag,
                    use_tools=req.use_tools,
                    use_langgraph=req.use_langgraph,
                    max_tokens=req.max_tokens,
                    model=req.model,
                    stream=True,
                )
                
                # Collect stream into result
                full_response = ""
                async for chunk in stream_result:
                    if isinstance(chunk, dict):
                        if chunk.get("type") == "token":
                            full_response += chunk.get("content", "")
                
                return {
                    "success": True,
                    "response": full_response,
                    "request_id": req.request_id,
                }
            else:
                # Non-streaming
                result = await asyncio.wait_for(
                    orchestrator.process_request(
                        user_input=req.user_input,
                        user_id=req.user_id,
                        workflow_id=req.workflow_id,
                        use_rag=req.use_rag,
                        use_tools=req.use_tools,
                        use_langgraph=req.use_langgraph,
                        max_tokens=req.max_tokens,
                        model=req.model,
                        stream=False,
                    ),
                    timeout=progressive_timeout
                )
                return result
        
        # Enqueue or process immediately
        queue_result = await queue_manager.enqueue_request(queued_request, process_queued_request)
        
        # If queued, return task info
        if queue_result.get("status") == "queued":
            return {
                "success": True,
                "status": "queued",
                "task_id": queue_result.get("task_id"),
                "request_id": queue_result.get("request_id"),
                "queue_position": queue_result.get("queue_position", 0),
                "message": "Request queued. Use /orchestration/queue/{task_id} to check status.",
            }
        
        # Otherwise, return result directly
        result = queue_result
        if isinstance(result, dict) and result.get("success") is False:
            error_msg = result.get("error", "Unknown error")
            raise HTTPException(
                status_code=500,
                detail=error_msg
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflow")
async def execute_workflow(
    input: ExecuteWorkflowInput,
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator),
):
    """Execute a multi-step workflow"""
    try:
        steps = [{"name": s.name, "tool": s.tool, "input": s.input} for s in input.steps]
        result = await orchestrator.execute_workflow(
            workflow_id=input.workflow_id,
            steps=steps,
            initial_state=input.initial_state,
        )
        return result
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status(
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator),
):
    """Get orchestrator status - with timeout protection"""
    import asyncio
    try:
        # Add an additional timeout wrapper as safety (10 seconds max)
        return await asyncio.wait_for(orchestrator.get_status(), timeout=10.0)
    except asyncio.TimeoutError:
        logger.error("Status endpoint timed out after 10 seconds")
        raise HTTPException(
            status_code=504,
            detail="Status check timed out. Some services may be slow or unavailable."
        )
    except Exception as e:
        logger.error(f"Failed to get status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflow/{workflow_id}")
async def get_workflow_state(
    workflow_id: str,
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator),
):
    """Get workflow state"""
    try:
        state = orchestrator.state_manager.get_state(workflow_id)
        if not state:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return state.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm-services")
async def discover_llm_services():
    """Discover available local LLM services on Docker network"""
    import asyncio
    try:
        from ..utils.docker_scanner import scan_docker_network_async
        
        # Use async version directly - much more efficient
        # Reduced timeout to 30 seconds (should be plenty with 5s per check)
        services = await asyncio.wait_for(
            scan_docker_network_async(),
            timeout=30.0  # 30 second timeout for scanning (5s per check * 7 hostnames = 35s max, but parallel)
        )
        
        return {
            "services": services,
            "count": len(services)
        }
    except asyncio.TimeoutError:
        logger.warning("Docker network scanning timed out after 30 seconds")
        # Return empty list instead of error - scanning is optional
        return {
            "services": [],
            "count": 0,
            "warning": "Service discovery timed out. Services may still be available."
        }
    except Exception as e:
        logger.error(f"Failed to discover LLM services: {e}", exc_info=True)
        # Return empty list instead of error - scanning is optional
        return {
            "services": [],
            "count": 0,
            "error": str(e)
        }


@router.get("/health-comprehensive")
async def get_comprehensive_health(
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator),
):
    """Get comprehensive health status of all runtime services"""
    import asyncio
    import time
    from ..settings import settings
    
    # Initialize default response structure
    health_status = {
        "timestamp": time.time(),
        "version": "3.0.0",
        "services": {},
        "configuration": {},
        "errors": []
    }
    
    try:
        # Get orchestrator status (with timeout)
        orchestrator_status = {}
        try:
            orchestrator_status = await asyncio.wait_for(
                orchestrator.get_status(),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            orchestrator_status = {"error": "Status check timed out", "status": "timeout"}
            health_status["errors"].append("Orchestrator status check timed out")
        except Exception as e:
            error_str = str(e)
            # Handle gRPC channel errors gracefully - these are often non-fatal
            if "channel" in error_str.lower() or "grpc" in error_str.lower() or "rpc" in error_str.lower():
                logger.debug(f"gRPC channel error during status check (non-fatal): {e}")
                # Try to get partial status without vector store
                orchestrator_status = {
                    "error": "gRPC channel error (non-fatal)",
                    "status": "partial",
                    "note": "Vector store status unavailable due to connection issue"
                }
                health_status["errors"].append("Vector store connection issue (gRPC channel error)")
            else:
                logger.warning(f"Failed to get orchestrator status: {e}")
                orchestrator_status = {"error": str(e), "status": "error"}
                health_status["errors"].append(f"Orchestrator status error: {str(e)}")
        
        # Get modelkit status
        modelkit_status = {
            "status": "unknown",
            "loaded": False,
            "models": [],
            "error": None
        }
        try:
            # Check if modelkit is importable
            try:
                import deepiri_modelkit
                modelkit_status["status"] = "available"
                modelkit_status["version"] = getattr(deepiri_modelkit, "__version__", "unknown")
            except ImportError:
                modelkit_status["status"] = "not_installed"
                modelkit_status["error"] = "deepiri_modelkit not installed"
            
            # Check for loaded models
            try:
                from ..integrations.model_loader import get_auto_loader, _auto_loader
                auto_loader = _auto_loader or await get_auto_loader()
                if auto_loader and hasattr(auto_loader, 'list_loaded_models'):
                    loaded_models = auto_loader.list_loaded_models()
                    modelkit_status["loaded"] = len(loaded_models) > 0
                    # Format models for display
                    modelkit_status["models"] = [
                        {
                            "name": model.get("key", "unknown"),
                            "version": model.get("metadata", {}).get("version", "unknown") if isinstance(model.get("metadata"), dict) else "unknown",
                            "loaded_at": model.get("loaded_at", "unknown")
                        }
                        for model in loaded_models
                    ]
            except Exception as e:
                modelkit_status["error"] = f"Failed to check loaded models: {str(e)}"
        except Exception as e:
            modelkit_status["error"] = str(e)
            modelkit_status["status"] = "error"
        
        # Get LLM services
        llm_services = []
        try:
            from ..utils.docker_scanner import scan_docker_network
            loop = asyncio.get_event_loop()
            services = await asyncio.wait_for(
                loop.run_in_executor(None, scan_docker_network),
                timeout=10.0
            )
            llm_services = services
        except Exception as e:
            logger.debug(f"Failed to discover LLM services: {e}")
        
        # Get detailed Milvus status separately
        milvus_status = {
            "status": "unknown",
            "healthy": False,
            "connection": {},
            "collection": {},
            "error": None
        }
        try:
            if orchestrator_status.get("vector_store"):
                vs_status = orchestrator_status["vector_store"]
                milvus_status = {
                    "status": "healthy" if vs_status.get("healthy") else "unhealthy",
                    "healthy": vs_status.get("healthy", False),
                    "connection": {
                        "host": vs_status.get("host") or vs_status.get("connection", {}).get("host", "unknown"),
                        "port": vs_status.get("port") or vs_status.get("connection", {}).get("port", "unknown"),
                        "connected": vs_status.get("connection", {}).get("connected", False) if isinstance(vs_status.get("connection"), dict) else False,
                    },
                    "collection": {
                        "name": vs_status.get("collection_name") or vs_status.get("collection", {}).get("name", "unknown"),
                        "exists": vs_status.get("collection", {}).get("exists", False) if isinstance(vs_status.get("collection"), dict) else False,
                        "num_entities": vs_status.get("num_entities") or vs_status.get("collection", {}).get("num_entities", 0),
                        "loaded": vs_status.get("collection", {}).get("loaded", False) if isinstance(vs_status.get("collection"), dict) else False,
                        "mode": vs_status.get("mode", "unknown"),
                    },
                    "embeddings": vs_status.get("embeddings", {}),
                    "errors": vs_status.get("errors", []),
                }
            else:
                milvus_status["error"] = "Vector store not initialized"
                milvus_status["status"] = "not_initialized"
        except Exception as e:
            milvus_status["error"] = str(e)
            milvus_status["status"] = "error"
        
        # Get Redis status
        redis_status = {
            "status": "unknown",
            "healthy": False,
            "connection": {},
            "error": None
        }
        try:
            from ..utils.cache import get_redis_client
            redis_client = get_redis_client()
            if redis_client:
                try:
                    redis_client.ping()
                    redis_status = {
                        "status": "healthy",
                        "healthy": True,
                        "connection": {
                            "host": getattr(settings, 'REDIS_HOST', 'unknown'),
                            "port": getattr(settings, 'REDIS_PORT', 'unknown'),
                            "db": getattr(settings, 'REDIS_DB', 0),
                            "connected": True,
                        }
                    }
                except Exception as e:
                    redis_status = {
                        "status": "unhealthy",
                        "healthy": False,
                        "connection": {
                            "host": getattr(settings, 'REDIS_HOST', 'unknown'),
                            "port": getattr(settings, 'REDIS_PORT', 'unknown'),
                            "db": getattr(settings, 'REDIS_DB', 0),
                            "connected": False,
                        },
                        "error": str(e)
                    }
            else:
                redis_status = {
                    "status": "not_configured",
                    "healthy": False,
                    "connection": {
                        "host": getattr(settings, 'REDIS_HOST', 'unknown'),
                        "port": getattr(settings, 'REDIS_PORT', 'unknown'),
                        "db": getattr(settings, 'REDIS_DB', 0),
                        "connected": False,
                    },
                    "error": "Redis client not initialized"
                }
        except Exception as e:
            redis_status["error"] = str(e)
            redis_status["status"] = "error"
        
        # Get MLflow status
        mlflow_status = {
            "status": "unknown",
            "healthy": False,
            "connection": {},
            "error": None
        }
        try:
            mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            import httpx
            try:
                # Try to connect to MLflow tracking server
                response = httpx.get(f"{mlflow_tracking_uri}/health", timeout=2.0)
                if response.status_code == 200:
                    mlflow_status = {
                        "status": "healthy",
                        "healthy": True,
                        "connection": {
                            "tracking_uri": mlflow_tracking_uri,
                            "connected": True,
                        }
                    }
                else:
                    mlflow_status = {
                        "status": "unhealthy",
                        "healthy": False,
                        "connection": {
                            "tracking_uri": mlflow_tracking_uri,
                            "connected": False,
                        },
                        "error": f"MLflow returned status code {response.status_code}"
                    }
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                mlflow_status = {
                    "status": "unhealthy",
                    "healthy": False,
                    "connection": {
                        "tracking_uri": mlflow_tracking_uri,
                        "connected": False,
                    },
                    "error": f"Connection failed: {str(e)}"
                }
            except Exception as e:
                mlflow_status = {
                    "status": "error",
                    "healthy": False,
                    "connection": {
                        "tracking_uri": mlflow_tracking_uri,
                        "connected": False,
                    },
                    "error": str(e)
                }
        except ImportError:
            mlflow_status = {
                "status": "not_available",
                "healthy": False,
                "connection": {},
                "error": "httpx not available for MLflow health check"
            }
        except Exception as e:
            mlflow_status["error"] = str(e)
            mlflow_status["status"] = "error"
        
        # Get Synapse status
        synapse_status = {
            "status": "unknown",
            "healthy": False,
            "connection": {},
            "error": None
        }
        try:
            synapse_url = os.getenv("SYNAPSE_URL", "http://localhost:8001")
            import httpx
            try:
                # Try to connect to Synapse health endpoint
                response = httpx.get(f"{synapse_url}/health", timeout=2.0)
                if response.status_code == 200:
                    synapse_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                    synapse_status = {
                        "status": synapse_data.get("status", "healthy") if synapse_data else "healthy",
                        "healthy": synapse_data.get("status") == "healthy" if synapse_data else True,
                        "connection": {
                            "url": synapse_url,
                            "connected": True,
                        }
                    }
                else:
                    synapse_status = {
                        "status": "unhealthy",
                        "healthy": False,
                        "connection": {
                            "url": synapse_url,
                            "connected": False,
                        },
                        "error": f"Synapse returned status code {response.status_code}"
                    }
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                synapse_status = {
                    "status": "unhealthy",
                    "healthy": False,
                    "connection": {
                        "url": synapse_url,
                        "connected": False,
                    },
                    "error": f"Connection failed: {str(e)}"
                }
            except Exception as e:
                synapse_status = {
                    "status": "error",
                    "healthy": False,
                    "connection": {
                        "url": synapse_url,
                        "connected": False,
                    },
                    "error": str(e)
                }
        except ImportError:
            synapse_status = {
                "status": "not_available",
                "healthy": False,
                "connection": {},
                "error": "httpx not available for Synapse health check"
            }
        except Exception as e:
            synapse_status["error"] = str(e)
            synapse_status["status"] = "error"
        
        # Compile comprehensive health status
        health_status["services"] = {
            "orchestrator": orchestrator_status,
            "modelkit": modelkit_status,
            "milvus": milvus_status,
            "redis": redis_status,
            "mlflow": mlflow_status,
            "synapse": synapse_status,
            "llm_services": llm_services,
        }
        
        # Get configuration safely
        try:
            health_status["configuration"] = {
                "local_llm_backend": getattr(settings, 'LOCAL_LLM_BACKEND', 'unknown'),
                "local_llm_model": getattr(settings, 'LOCAL_LLM_MODEL', 'unknown'),
                "ollama_base_url": getattr(settings, 'OLLAMA_BASE_URL', 'unknown'),
                "openai_configured": bool(getattr(settings, 'OPENAI_API_KEY', None)),
                "milvus_host": getattr(settings, 'MILVUS_HOST', 'unknown'),
                "milvus_port": getattr(settings, 'MILVUS_PORT', 'unknown'),
            }
        except Exception as e:
            logger.warning(f"Failed to get configuration: {e}")
            health_status["configuration"] = {"error": str(e)}
            health_status["errors"].append(f"Configuration error: {str(e)}")
        
        return health_status
    except Exception as e:
        logger.error(f"Failed to get comprehensive health: {e}", exc_info=True)
        # Return partial results even on error
        health_status["errors"].append(f"Critical error: {str(e)}")
        return health_status
