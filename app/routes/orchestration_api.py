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
        result = await orchestrator.process_request(
            user_input=input.user_input,
            user_id=input.user_id,
            workflow_id=input.workflow_id,
            use_rag=input.use_rag,
            use_tools=input.use_tools,
        )
        return result
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

