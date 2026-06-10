"""
LangGraph Workflow API Routes
API endpoints for LangGraph workflow execution and testing
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio
import json
import uuid

from ..core.langgraph_workflow import get_langgraph_workflow
from ..logging_config import get_logger

logger = get_logger("cyrex.routes.workflow")

router = APIRouter(prefix="/api/workflow", tags=["LangGraph Workflow"])


# ============================================================================
# Request/Response Models
# ============================================================================

class WorkflowExecuteRequest(BaseModel):
    """Request to execute a LangGraph workflow"""
    task_description: str
    workflow_type: Optional[str] = "standard"  # standard, lease, contract, fraud
    context: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    workflow_id: Optional[str] = None
    stream: bool = False
    config: Optional[Dict[str, Any]] = None  # LangGraph config


class WorkflowStateResponse(BaseModel):
    """Response with workflow state"""
    workflow_id: str
    workflow_type: str
    task_type: Optional[str]
    current_agent: str
    status: str
    result: Optional[Dict[str, Any]] = None
    agent_history: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


# ============================================================================
# Routes
# ============================================================================

@router.post("/execute")
async def execute_workflow(request: WorkflowExecuteRequest):
    """
    Execute a LangGraph workflow
    
    Supports streaming and non-streaming responses
    """
    try:
        workflow_id = request.workflow_id or f"workflow_{datetime.now().timestamp()}"
        
        # Get workflow instance
        workflow = await get_langgraph_workflow()
        if not workflow:
            raise HTTPException(status_code=500, detail="Workflow not available")
        
        # Prepare input data
        input_data = {
            "task_description": request.task_description,
            "context": request.context,
            "session_id": request.session_id,
            "user_id": request.user_id,
            "workflow_id": workflow_id,
        }
        
        if request.stream:
            return StreamingResponse(
                stream_workflow_execution(workflow, input_data, request.config),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming execution
            result = await workflow.ainvoke(input_data, config=request.config)
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "final_state": result,
            }
    
    except Exception as e:
        logger.error(f"Workflow execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def stream_workflow_execution(
    workflow,
    input_data: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
):
    """Stream workflow execution with real-time updates"""
    try:
        workflow_id = input_data.get("workflow_id", "unknown")
        
        # Send initial state
        yield json.dumps({
            "type": "workflow_started",
            "workflow_id": workflow_id,
            "timestamp": datetime.utcnow().isoformat(),
        }) + "\n"
        
        # Execute workflow with state updates
        # Note: LangGraph doesn't natively support streaming state updates
        # We'll simulate by executing and sending periodic updates
        
        result = await workflow.ainvoke(input_data, config=config)
        
        # Send state updates based on agent history
        if result.get("agent_history"):
            for entry in result["agent_history"]:
                yield json.dumps({
                    "type": "agent_start",
                    "agent": entry.get("agent"),
                    "role": entry.get("role"),
                    "timestamp": entry.get("timestamp"),
                }) + "\n"
                
                await asyncio.sleep(0.1)  # Small delay for smooth streaming
                
                yield json.dumps({
                    "type": "agent_complete",
                    "agent": entry.get("agent"),
                    "response": entry.get("response", "")[:200] if entry.get("response") else "",
                    "timestamp": entry.get("timestamp"),
                }) + "\n"
        
        # Send state updates
        yield json.dumps({
            "type": "state_update",
            "state": result,
        }) + "\n"
        
        # Send final state
        yield json.dumps({
            "type": "done",
            "final_state": result,
            "workflow_id": workflow_id,
        }) + "\n"
    
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        yield json.dumps({
            "type": "error",
            "error": str(e),
            "workflow_id": input_data.get("workflow_id", "unknown"),
        }) + "\n"


@router.post("/{workflow_id}/stop")
async def stop_workflow(workflow_id: str) -> Dict[str, Any]:
    """Stop a running workflow"""
    # Note: LangGraph workflows can't be easily stopped mid-execution
    # This is a placeholder for future implementation
    return {
        "success": True,
        "message": "Workflow stop requested",
        "workflow_id": workflow_id,
        "note": "Workflow may complete current step before stopping",
    }


@router.get("/{workflow_id}/state")
async def get_workflow_state(workflow_id: str) -> Dict[str, Any]:
    """Get workflow state by ID"""
    # Note: This would require checkpoint storage to retrieve state
    # For now, return a placeholder
    return {
        "workflow_id": workflow_id,
        "message": "State retrieval requires checkpoint storage",
        "note": "Use Redis checkpointing to enable state retrieval",
    }


@router.get("/types")
async def list_workflow_types() -> List[Dict[str, Any]]:
    """List available workflow types"""
    return [
        {
            "id": "standard",
            "name": "Standard Workflow",
            "description": "Task → Plan → Code → Quality",
            "agents": ["task_agent", "plan_agent", "code_agent", "qa_agent"],
        },
        {
            "id": "lease",
            "name": "Lease Abstraction",
            "description": "Process lease documents and extract structured data",
            "agents": ["lease_processor"],
        },
        {
            "id": "contract",
            "name": "Contract Intelligence",
            "description": "Analyze contracts, track clause evolution, build dependency graphs",
            "agents": ["contract_processor"],
        },
        {
            "id": "fraud",
            "name": "Vendor Fraud Detection",
            "description": "Detect vendor fraud, analyze invoices, benchmark pricing",
            "agents": ["fraud_agent"],
        },
    ]


@router.get("/health")
async def workflow_health() -> Dict[str, Any]:
    """Check workflow system health"""
    try:
        workflow = await get_langgraph_workflow()
        
        return {
            "status": "healthy" if workflow else "degraded",
            "workflow_available": workflow is not None,
            "langgraph_available": workflow.graph is not None if workflow else False,
            "checkpointing_available": workflow.checkpointer is not None if workflow else False,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }

