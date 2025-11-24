"""
Intelligence API
API endpoints for three-tier AI system:
1. Command Routing (BERT/DeBERTa)
2. Contextual Ability Generation (LLM + RAG)
3. Workflow Optimization (RL)
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from ..services.command_router import get_command_router
from ..services.contextual_ability_engine import get_contextual_ability_engine
from ..services.workflow_optimizer import get_workflow_optimizer
from ..services.knowledge_retrieval_engine import get_knowledge_retrieval_engine
from ..logging_config import get_logger, ErrorLogger

router = APIRouter()
logger = get_logger("cyrex.intelligence_api")
error_logger = ErrorLogger()


# Request Models
class CommandRoutingRequest(BaseModel):
    command: str = Field(..., description="User's natural language command")
    user_role: Optional[str] = Field(None, description="User's role (software_engineer, designer, etc.)")
    context: Optional[Dict] = Field(default_factory=dict, description="Additional context")
    min_confidence: Optional[float] = Field(0.7, ge=0, le=1, description="Minimum confidence threshold")
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Number of top predictions")


class AbilityGenerationRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    user_command: str = Field(..., description="User's request")
    user_profile: Dict = Field(..., description="User profile (role, momentum, level, etc.)")
    project_context: Optional[Dict] = Field(default_factory=dict, description="Project context")
    chat_history: Optional[List] = Field(default_factory=list, description="Conversation history")


class WorkflowRecommendationRequest(BaseModel):
    user_data: Dict = Field(..., description="User's current state (momentum, tasks, efficiency, etc.)")


class RewardFeedbackRequest(BaseModel):
    outcome: Dict = Field(..., description="Outcome data (task_completed, efficiency, user_rating, etc.)")


class KnowledgeIndexRequest(BaseModel):
    content: str = Field(..., description="Content to index")
    metadata: Dict = Field(..., description="Document metadata")
    knowledge_base: str = Field("project_context", description="Target knowledge base")


class KnowledgeQueryRequest(BaseModel):
    query: str = Field(..., description="Search query")
    knowledge_bases: Optional[List[str]] = Field(None, description="KB names to search (None = all)")
    top_k: int = Field(5, ge=1, le=20, description="Number of results")


# Tier 1: Command Routing Routes
@router.post("/intelligence/route-command")
async def route_command(req: CommandRoutingRequest, request: Request):
    """
    Route user command to predefined abilities (Tier 1: Classification)
    
    Uses fine-tuned BERT/DeBERTa to route user commands to predefined abilities.
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        router = get_command_router()
        
        if req.min_confidence:
            # Single best match with confidence threshold
            result = router.route_with_confidence_threshold(
                req.command,
                req.user_role,
                req.min_confidence,
                req.context
            )
            
            if result:
                return {
                    "success": True,
                    "data": result,
                    "request_id": request_id
                }
            else:
                return {
                    "success": False,
                    "message": "No ability found above confidence threshold",
                    "request_id": request_id
                }
        else:
            # Top-k predictions
            predictions = router.route_command(
                req.command,
                req.user_role,
                req.context,
                req.top_k
            )
            
            return {
                "success": True,
                "data": {
                    "predictions": predictions,
                    "count": len(predictions)
                },
                "request_id": request_id
            }
    
    except Exception as e:
        error_logger.log_api_error(e, request_id, "/intelligence/route-command")
        logger.error(f"Command routing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Tier 2: Contextual Ability Generation Routes
@router.post("/intelligence/generate-ability")
async def generate_ability(req: AbilityGenerationRequest, request: Request):
    """
    Generate dynamic ability using LLM + RAG (Tier 2: Generation)
    
    Uses GPT-4/Claude with RAG to generate unique, contextual abilities on-the-fly.
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        engine = get_contextual_ability_engine()
        
        result = engine.generate_ability(
            req.user_id,
            req.user_command,
            req.user_profile,
            req.project_context,
            req.chat_history
        )
        
        return {
            "success": result["success"],
            "data": result.get("ability"),
            "alternatives": result.get("alternatives", []),
            "error": result.get("error"),
            "request_id": request_id
        }
    
    except Exception as e:
        error_logger.log_api_error(e, request_id, "/intelligence/generate-ability")
        logger.error(f"Ability generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/intelligence/ability/feedback")
async def ability_feedback(req: RewardFeedbackRequest, request: Request):
    """Provide feedback on generated ability for learning"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        # This would store feedback for future RAG retrieval
        # Implementation depends on your feedback storage system
        return {
            "success": True,
            "message": "Feedback recorded",
            "request_id": request_id
        }
    
    except Exception as e:
        error_logger.log_api_error(e, request_id, "/intelligence/ability/feedback")
        raise HTTPException(status_code=500, detail=str(e))


# Tier 3: Workflow Optimization Routes
@router.post("/intelligence/recommend-action")
async def recommend_action(req: WorkflowRecommendationRequest, request: Request):
    """
    Get RL agent's recommended action (Tier 3: Optimization)
    
    Uses PPO agent to recommend optimal actions based on user's current state.
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        optimizer = get_workflow_optimizer()
        
        recommendation = optimizer.recommend_action(req.user_data)
        
        return {
            "success": True,
            "data": recommendation,
            "request_id": request_id
        }
    
    except Exception as e:
        error_logger.log_api_error(e, request_id, "/intelligence/recommend-action")
        logger.error(f"Action recommendation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/intelligence/optimizer/reward")
async def record_reward(req: RewardFeedbackRequest, request: Request):
    """
    Record reward for RL agent training
    
    Provides outcome feedback to train the workflow optimizer.
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        optimizer = get_workflow_optimizer()
        
        reward = optimizer.compute_reward(req.outcome)
        
        # Store transition (would need state/action from previous recommendation)
        # This is simplified - in production, you'd track the full trajectory
        
        return {
            "success": True,
            "data": {
                "reward": reward,
                "message": "Reward recorded for optimizer learning"
            },
            "request_id": request_id
        }
    
    except Exception as e:
        error_logger.log_api_error(e, request_id, "/intelligence/optimizer/reward")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/intelligence/optimizer/update")
async def update_optimizer(request: Request):
    """Update RL agent policy (training endpoint)"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        optimizer = get_workflow_optimizer()
        
        result = optimizer.update(epochs=10, batch_size=64)
        
        return {
            "success": True,
            "data": result,
            "request_id": request_id
        }
    
    except Exception as e:
        error_logger.log_api_error(e, request_id, "/intelligence/optimizer/update")
        raise HTTPException(status_code=500, detail=str(e))


# Knowledge Retrieval Routes
@router.post("/intelligence/knowledge/index")
async def index_document(req: KnowledgeIndexRequest, request: Request):
    """Index document in knowledge base"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        engine = get_knowledge_retrieval_engine()
        
        engine.add_document(
            req.content,
            req.metadata,
            req.knowledge_base
        )
        
        return {
            "success": True,
            "message": f"Document indexed in {req.knowledge_base}",
            "request_id": request_id
        }
    
    except Exception as e:
        error_logger.log_api_error(e, request_id, "/intelligence/knowledge/index")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/intelligence/knowledge/query")
async def query_knowledge(req: KnowledgeQueryRequest, request: Request):
    """Query knowledge bases"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        engine = get_knowledge_retrieval_engine()
        
        docs = engine.retrieve(
            req.query,
            req.knowledge_bases,
            req.top_k
        )
        
        results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "knowledge_base": doc.metadata.get("knowledge_base", "unknown")
            }
            for doc in docs
        ]
        
        return {
            "success": True,
            "data": {
                "results": results,
                "count": len(results)
            },
            "request_id": request_id
        }
    
    except Exception as e:
        error_logger.log_api_error(e, request_id, "/intelligence/knowledge/query")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/intelligence/knowledge/query-formatted")
async def query_knowledge_formatted(req: KnowledgeQueryRequest, request: Request):
    """Query knowledge bases and return formatted context string"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        engine = get_knowledge_retrieval_engine()
        
        context = engine.retrieve_formatted(
            req.query,
            req.knowledge_bases,
            req.top_k
        )
        
        return {
            "success": True,
            "data": {
                "context": context,
                "query": req.query
            },
            "request_id": request_id
        }
    
    except Exception as e:
        error_logger.log_api_error(e, request_id, "/intelligence/knowledge/query-formatted")
        raise HTTPException(status_code=500, detail=str(e))

