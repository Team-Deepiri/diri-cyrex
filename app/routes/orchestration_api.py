"""
Orchestration API Routes
Exposes the workflow orchestrator via REST API
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
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
    force_local_llm: bool = Field(False, description="Force use of local LLM instead of OpenAI")
    llm_backend: Optional[str] = Field(None, description="Local LLM backend (ollama, llama_cpp, transformers)")
    llm_model: Optional[str] = Field(None, description="Local LLM model name (e.g., llama3:8b)")
    llm_base_url: Optional[str] = Field(None, description="Local LLM base URL (e.g., http://ollama:11434)")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate (default: 500 for local LLM, 2000 for OpenAI)")


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
            
            try:
                backend = LLMBackend(backend_str)
                # Pass max_tokens if provided, otherwise use default (200 for local LLM on CPU - faster responses)
                max_tokens_for_llm = input.max_tokens if input.max_tokens else 200
                # Use provided base_url or fall back to settings
                base_url = input.llm_base_url
                if not base_url and backend == LLMBackend.OLLAMA:
                    base_url = settings.OLLAMA_BASE_URL
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
                        temp_orchestrator = WorkflowOrchestrator(llm_provider=local_llm)
                        result = await temp_orchestrator.process_request(
                            user_input=input.user_input,
                            user_id=input.user_id,
                            workflow_id=input.workflow_id,
                            use_rag=input.use_rag,
                            use_tools=input.use_tools,
                            max_tokens=max_tokens_for_llm,
                        )
                        # Check if result indicates failure
                        if isinstance(result, dict) and result.get("success") is False:
                            error_msg = result.get("error", "Unknown error")
                            raise HTTPException(
                                status_code=500,
                                detail=error_msg
                        )
                        return result
                    except TimeoutError as e:
                        raise HTTPException(
                            status_code=504,
                            detail=f"Local LLM ({backend_str}) request timed out after 120 seconds. The model may be too slow or unresponsive. Try reducing max_tokens or using a faster model. Error: {str(e)}"
                        )
                    except Exception as e:
                        error_msg = str(e)
                        error_lower = error_msg.lower()
                        
                        # Check for specific error types
                        if "404" in error_msg or "not found" in error_lower or "model" in error_lower and "not found" in error_lower:
                            raise HTTPException(
                                status_code=404,
                                detail=f"Model '{input.llm_model or 'llama3:8b'}' not found in Ollama. Please pull the model first: docker exec -it deepiri-ollama-ai ollama pull {input.llm_model or 'llama3:8b'}"
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
        result = await orchestrator.process_request(
            user_input=input.user_input,
            user_id=input.user_id,
            workflow_id=input.workflow_id,
            use_rag=input.use_rag,
            use_tools=input.use_tools,
        )
        # Check if result indicates failure
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
        from ..utils.docker_scanner import scan_docker_network
        
        # Run scanning in executor with timeout to prevent hanging
        loop = asyncio.get_event_loop()
        services = await asyncio.wait_for(
            loop.run_in_executor(None, scan_docker_network),
            timeout=60.0  # 60 second timeout for scanning
        )
        
        return {
            "services": services,
            "count": len(services)
        }
    except asyncio.TimeoutError:
        logger.warning("Docker network scanning timed out after 60 seconds")
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
