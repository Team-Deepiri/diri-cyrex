"""
Agent Playground API Routes
API endpoints for the Agent Playground UI
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio
import json
import uuid

from ..core.prompt_templates import get_prompt_template_manager, PromptCategory
from ..core.queue_system import get_queue_producer, create_queue_consumer, QueuePriority
from ..core.enhanced_state_manager import get_enhanced_state_manager, WorkflowPhase
from ..core.redis_streams_broker import get_redis_streams_broker, StreamEventType
from ..core.advanced_guardrails import get_advanced_guardrails
from ..integrations.ollama_container import get_ollama_client, ChatMessage, GenerationOptions
from ..integrations.realtime_streaming import get_stream_publisher
from ..agents.base_agent import BaseAgent, AgentResponse
from ..agents.agent_factory import create_agent
from ..agents.tools.enhanced_memory_tools import EnhancedMemoryTools
from ..agents.tools.comprehensive_api_tools import ComprehensiveAPITools
from ..database.agent_tables import initialize_agent_database
from ..logging_config import get_logger

logger = get_logger("cyrex.routes.agent_playground")

router = APIRouter(prefix="/api/agent", tags=["Agent Playground"])


# ============================================================================
# Request/Response Models
# ============================================================================

class AgentConfigRequest(BaseModel):
    """Request to initialize an agent"""
    agent_id: Optional[str] = None
    agent_type: str = "conversational"
    name: str = "Test Agent"
    model: str = "llama3:8b"
    temperature: float = 0.7
    max_tokens: int = 2000
    system_prompt: str = "You are a helpful AI assistant."
    tools: List[str] = Field(default_factory=list)


class AgentInvokeRequest(BaseModel):
    """Request to invoke an agent"""
    instance_id: str
    input: str
    config: Optional[AgentConfigRequest] = None
    stream: bool = False
    context: Dict[str, Any] = Field(default_factory=dict)


class AgentInstanceResponse(BaseModel):
    """Response for agent instance"""
    instance_id: str
    agent_id: str
    status: str
    model: str
    tools: List[str]
    started_at: str


# ============================================================================
# In-Memory Agent Store (for playground)
# ============================================================================

_active_agents: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Routes
# ============================================================================

@router.post("/initialize")
async def initialize_agent(config: AgentConfigRequest) -> Dict[str, Any]:
    """Initialize a new agent instance for the playground"""
    try:
        instance_id = str(uuid.uuid4())
        agent_id = config.agent_id or str(uuid.uuid4())
        
        # Verify Ollama connection
        ollama = await get_ollama_client()
        health = await ollama.health_check()
        
        if health.get("status") != "healthy":
            logger.warning(f"Ollama not healthy: {health}")
            # Continue anyway for testing
        
        # Store agent configuration
        _active_agents[instance_id] = {
            "instance_id": instance_id,
            "agent_id": agent_id,
            "config": config.model_dump(),
            "status": "idle",
            "started_at": datetime.utcnow().isoformat(),
            "conversation": [],
            "metrics": {
                "tokens_used": 0,
                "tool_calls": 0,
                "messages": 0,
            },
        }
        
        # Publish event
        try:
            broker = await get_redis_streams_broker()
            await broker.publish(
                f"agent:{instance_id}",
                StreamEventType.AGENT_STARTED,
                {"agent_id": agent_id, "config": config.model_dump()},
            )
        except Exception as e:
            logger.warning(f"Failed to publish event: {e}")
        
        logger.info(f"Initialized agent instance: {instance_id}")
        
        return {
            "instance_id": instance_id,
            "agent_id": agent_id,
            "status": "idle",
            "model": config.model,
            "tools": config.tools,
            "started_at": _active_agents[instance_id]["started_at"],
        }
    
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/invoke")
async def invoke_agent(request: AgentInvokeRequest):
    """Invoke an agent with input"""
    instance_id = request.instance_id
    
    if instance_id not in _active_agents:
        raise HTTPException(status_code=404, detail="Agent instance not found")
    
    agent_data = _active_agents[instance_id]
    config = AgentConfigRequest(**agent_data["config"])
    
    # Update status
    agent_data["status"] = "processing"
    
    if request.stream:
        return StreamingResponse(
            stream_agent_response(instance_id, request.input, config),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming response
        try:
            response = await generate_agent_response(instance_id, request.input, config)
            agent_data["status"] = "idle"
            return response
        except Exception as e:
            agent_data["status"] = "error"
            raise HTTPException(status_code=500, detail=str(e))


async def stream_agent_response(
    instance_id: str,
    user_input: str,
    config: AgentConfigRequest,
):
    """Stream agent response token by token"""
    try:
        ollama = await get_ollama_client()
        guardrails = await get_advanced_guardrails()
        
        agent_data = _active_agents.get(instance_id, {})
        conversation = agent_data.get("conversation", [])
        
        # Safety check
        safety_result = await guardrails.check_input(user_input)
        if not safety_result.passed:
            yield json.dumps({
                "type": "error",
                "content": f"Input blocked: {safety_result.message}",
            }) + "\n"
            return
        
        # Build messages
        messages = [
            ChatMessage(role="system", content=config.system_prompt),
        ]
        
        # Add conversation history
        for msg in conversation[-10:]:  # Last 10 messages
            messages.append(ChatMessage(
                role=msg["role"],
                content=msg["content"],
            ))
        
        # Add user message
        messages.append(ChatMessage(role="user", content=user_input))
        
        # Add to conversation history
        conversation.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Generate streaming response
        options = GenerationOptions(
            temperature=config.temperature,
            num_predict=config.max_tokens,
        )
        
        full_response = ""
        
        async for token in ollama.chat_stream(messages, model=config.model, options=options):
            full_response += token
            yield json.dumps({"type": "token", "content": token}) + "\n"
            await asyncio.sleep(0.01)  # Small delay for smooth streaming
        
        # Add assistant response to history
        conversation.append({
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Update metrics
        agent_data["metrics"]["tokens_used"] += len(full_response) / 4
        agent_data["metrics"]["messages"] += 2
        agent_data["status"] = "idle"
        
        # Check output safety
        output_result = await guardrails.check_output(full_response)
        if not output_result.passed:
            yield json.dumps({
                "type": "warning",
                "content": f"Output warning: {output_result.message}",
            }) + "\n"
        
        yield json.dumps({"type": "done", "total_tokens": len(full_response)}) + "\n"
    
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"
        
        if instance_id in _active_agents:
            _active_agents[instance_id]["status"] = "error"


async def generate_agent_response(
    instance_id: str,
    user_input: str,
    config: AgentConfigRequest,
) -> Dict[str, Any]:
    """Generate non-streaming agent response"""
    try:
        ollama = await get_ollama_client()
        guardrails = await get_advanced_guardrails()
        
        agent_data = _active_agents.get(instance_id, {})
        conversation = agent_data.get("conversation", [])
        
        # Safety check
        safety_result = await guardrails.check_input(user_input)
        if not safety_result.passed:
            return {
                "success": False,
                "error": f"Input blocked: {safety_result.message}",
            }
        
        # Build messages
        messages = [
            ChatMessage(role="system", content=config.system_prompt),
        ]
        
        for msg in conversation[-10:]:
            messages.append(ChatMessage(role=msg["role"], content=msg["content"]))
        
        messages.append(ChatMessage(role="user", content=user_input))
        
        # Add to conversation
        conversation.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Generate response
        options = GenerationOptions(
            temperature=config.temperature,
            num_predict=config.max_tokens,
        )
        
        result = await ollama.chat(messages, model=config.model, options=options)
        
        # Add to conversation
        conversation.append({
            "role": "assistant",
            "content": result.response,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Update metrics
        agent_data["metrics"]["tokens_used"] += result.eval_count
        agent_data["metrics"]["messages"] += 2
        
        return {
            "success": True,
            "response": result.response,
            "tokens_used": result.eval_count,
            "duration_ms": result.total_duration / 1_000_000,  # Convert ns to ms
        }
    
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


@router.post("/{instance_id}/stop")
async def stop_agent(instance_id: str) -> Dict[str, Any]:
    """Stop an agent instance"""
    if instance_id not in _active_agents:
        raise HTTPException(status_code=404, detail="Agent instance not found")
    
    # Publish stop event
    try:
        broker = await get_redis_streams_broker()
        await broker.publish(
            f"agent:{instance_id}",
            StreamEventType.AGENT_COMPLETED,
            {"reason": "stopped"},
        )
    except Exception as e:
        logger.warning(f"Failed to publish stop event: {e}")
    
    # Remove from active agents
    del _active_agents[instance_id]
    
    logger.info(f"Stopped agent instance: {instance_id}")
    
    return {"success": True, "message": "Agent stopped"}


@router.get("/{instance_id}/status")
async def get_agent_status(instance_id: str) -> Dict[str, Any]:
    """Get agent instance status"""
    if instance_id not in _active_agents:
        raise HTTPException(status_code=404, detail="Agent instance not found")
    
    agent_data = _active_agents[instance_id]
    
    return {
        "instance_id": instance_id,
        "agent_id": agent_data["agent_id"],
        "status": agent_data["status"],
        "started_at": agent_data["started_at"],
        "metrics": agent_data["metrics"],
        "conversation_length": len(agent_data.get("conversation", [])),
    }


@router.get("/instances")
async def list_agent_instances() -> List[Dict[str, Any]]:
    """List all active agent instances"""
    return [
        {
            "instance_id": data["instance_id"],
            "agent_id": data["agent_id"],
            "status": data["status"],
            "model": data["config"]["model"],
            "started_at": data["started_at"],
        }
        for data in _active_agents.values()
    ]


@router.get("/templates")
async def list_prompt_templates(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available prompt templates"""
    manager = get_prompt_template_manager()
    
    cat = PromptCategory(category) if category else None
    templates = manager.list_templates(category=cat)
    
    return [
        {
            "template_id": t.template_id,
            "name": t.name,
            "description": t.description,
            "category": t.category.value,
            "variables": [v.model_dump() for v in t.variables],
        }
        for t in templates
    ]


@router.get("/tools")
async def list_available_tools() -> List[Dict[str, Any]]:
    """List available agent tools"""
    api_tools = ComprehensiveAPITools()
    
    return [tool.to_dict() for tool in api_tools.list_tools()]


@router.get("/models")
async def list_available_models() -> Dict[str, Any]:
    """List available Ollama models - auto-detected from container"""
    try:
        ollama = await get_ollama_client()
        
        # Try to connect if not connected, but don't fail if it times out
        connection_ok = ollama.is_connected
        if not connection_ok:
            try:
                # Try quick connection with timeout
                connection_ok = await asyncio.wait_for(ollama.connect(), timeout=5.0)
                logger.info(f"Ollama connection attempt: {connection_ok}")
            except asyncio.TimeoutError:
                logger.debug("Connection attempt timed out, but will try to list models anyway")
                connection_ok = False
            except Exception as connect_err:
                logger.debug(f"Connection attempt failed: {connect_err}, but will try to list models anyway")
                connection_ok = False
        
        # Try to get models even if connection check failed
        # (Ollama might be working but health check is slow)
        models = []
        try:
            # Use a timeout for listing models too
            models = await asyncio.wait_for(ollama.list_models(), timeout=10.0)
            # If we got models, we're definitely connected
            if models:
                connection_ok = True
                logger.info(f"Successfully listed {len(models)} models from Ollama")
        except asyncio.TimeoutError:
            logger.warning("Model listing timed out")
            # If we have cached models, use those
            if ollama.available_models:
                models = ollama.available_models
                connection_ok = True
                logger.info(f"Using {len(models)} cached models due to timeout")
        except Exception as list_err:
            logger.warning(f"Failed to list models: {list_err}")
            # If we have cached models, use those
            if ollama.available_models:
                models = ollama.available_models
                connection_ok = True
                logger.info(f"Using {len(models)} cached models due to error")
            else:
                # No models available
                models = []
        
        # Determine status: if we have models, we're connected
        status = "connected" if (connection_ok or models) else "disconnected"
        
        result = {
            "status": status,
            "is_connected": bool(connection_ok or models),
            "models": [
                {
                    "name": m.name,
                    "size": m.size,
                    "modified_at": m.modified_at,
                }
                for m in models
            ],
            "model_names": [m.name for m in models],  # Simple list for frontend
        }
        
        logger.debug(f"Models endpoint response: status={status}, models_count={len(models)}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to list Ollama models: {e}", exc_info=True)
        # Even on error, check if we have cached models
        try:
            ollama = await get_ollama_client()
            if ollama.available_models:
                logger.info(f"Returning {len(ollama.available_models)} cached models despite error")
                return {
                    "status": "connected",
                    "is_connected": True,
                    "models": [
                        {
                            "name": m.name,
                            "size": m.size,
                            "modified_at": m.modified_at,
                        }
                        for m in ollama.available_models
                    ],
                    "model_names": [m.name for m in ollama.available_models],
                    "warning": "Using cached models due to error",
                }
        except Exception as cache_err:
            logger.debug(f"Could not get cached models: {cache_err}")
        
        return {
            "status": "error",
            "is_connected": False,
            "error": str(e),
            "models": [],
            "model_names": [],
        }


@router.post("/evaluate")
async def evaluate_agent_response(
    response: str,
    expected: Optional[str] = None,
    criteria: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate an agent response"""
    # Basic evaluation metrics
    evaluation = {
        "length": len(response),
        "word_count": len(response.split()),
    }
    
    # Safety check
    guardrails = await get_advanced_guardrails()
    safety = await guardrails.check_output(response)
    evaluation["safety"] = {
        "passed": safety.passed,
        "risk_level": safety.risk_level.value,
        "message": safety.message,
    }
    
    # Similarity check if expected provided
    if expected:
        # Simple word overlap similarity
        response_words = set(response.lower().split())
        expected_words = set(expected.lower().split())
        overlap = len(response_words & expected_words)
        total = len(response_words | expected_words)
        evaluation["similarity"] = overlap / total if total > 0 else 0
    
    return evaluation

