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
    user_input: str = Field(..., description="User's input/request")
    user_id: Optional[str] = Field(None, description="User identifier")
    workflow_id: Optional[str] = Field(None, description="Workflow ID for state tracking")
    use_rag: bool = Field(True, description="Whether to use RAG for context")
    use_tools: bool = Field(True, description="Whether to allow tool usage")
    force_local_llm: bool = Field(False, description="Force use of local LLM instead of OpenAI")
    llm_backend: Optional[str] = Field(None, description="Local LLM backend (ollama, llama_cpp, transformers)")
    llm_model: Optional[str] = Field(None, description="Local LLM model name (e.g., llama3:8b)")


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
                local_llm = get_local_llm(
                    backend=backend.value,
                    model_name=model_str,
                    base_url=settings.OLLAMA_BASE_URL if backend == LLMBackend.OLLAMA else None,
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
                        )
                        return result
                    except Exception as e:
                        error_msg = str(e)
                        if "Connection" in error_msg or "connect" in error_msg.lower():
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
    """Get orchestrator status"""
    try:
        return await orchestrator.get_status()
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

