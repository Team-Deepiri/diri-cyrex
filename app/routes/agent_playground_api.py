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
from ..core.advanced_guardrails import get_advanced_guardrails, GuardrailAction
from ..core.orchestrator import get_orchestrator
from ..integrations.ollama_container import get_ollama_client, ChatMessage, GenerationOptions
from ..integrations.realtime_streaming import get_stream_publisher
from ..agents.base_agent import BaseAgent, AgentResponse
from ..agents.agent_factory import create_agent
from ..agents.tools.enhanced_memory_tools import EnhancedMemoryTools
from ..agents.tools.comprehensive_api_tools import ComprehensiveAPITools
from ..database.agent_tables import initialize_agent_database
from ..database.postgres import get_postgres_manager
from ..logging_config import get_logger

logger = get_logger("cyrex.routes.agent_playground")

router = APIRouter(prefix="/api/agent", tags=["Agent Playground"])

# Initialize database tables on module load (will be called when router is imported)
_initialized = False

async def ensure_database_initialized():
    """Ensure database tables are created"""
    global _initialized
    if not _initialized:
        try:
            await initialize_agent_database()
            _initialized = True
            logger.info("Agent playground database tables initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize agent playground database: {e}")


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
    system_prompt: str = """You are a helpful, intelligent AI assistant. Your goal is to provide accurate, useful, and contextually appropriate responses.

Guidelines:
- Be clear, concise, and direct in your responses
- If you don't know something, admit it rather than guessing
- Ask clarifying questions when the user's request is ambiguous
- Maintain context from the conversation history
- Be professional but friendly in your tone
- Focus on being helpful and solving the user's problem
- If asked to do something you cannot do, explain what you can do instead

Remember: Quality over quantity. Better to give a short, accurate answer than a long, rambling one."""
    tools: List[str] = Field(default_factory=list)


class MultipleAgentsRequest(BaseModel):
    """Request to initialize multiple agents"""
    agents: List[AgentConfigRequest]


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
_group_chats: Dict[str, Dict[str, Any]] = {}  # Group chat ID -> group chat data


# ============================================================================
# Database Functions for Conversation Storage
# ============================================================================

async def save_message_to_db(
    instance_id: str,
    agent_id: str,
    role: str,
    content: str,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    is_error: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a conversation message to the database"""
    try:
        postgres = await get_postgres_manager()
        message_id = str(uuid.uuid4())
        
        await postgres.execute("""
            INSERT INTO cyrex.agent_playground_messages 
            (message_id, instance_id, agent_id, role, content, tool_calls, is_error, metadata, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
        """, 
            message_id,
            instance_id,
            agent_id,
            role,
            content,
            json.dumps(tool_calls) if tool_calls else None,
            is_error,
            json.dumps(metadata) if metadata else json.dumps({}),
        )
        
        logger.debug(f"Saved message {message_id} for instance {instance_id}")
        return message_id
    except Exception as e:
        logger.error(f"Failed to save message to database: {e}", exc_info=True)
        # Don't fail the request if DB save fails
        return str(uuid.uuid4())


async def load_conversation_history(instance_id: str) -> List[Dict[str, Any]]:
    """Load conversation history for an agent instance from the database"""
    try:
        postgres = await get_postgres_manager()
        rows = await postgres.fetch("""
            SELECT message_id, agent_id, role, content, tool_calls, is_error, metadata, created_at
            FROM cyrex.agent_playground_messages
            WHERE instance_id = $1
            ORDER BY created_at ASC
        """, instance_id)
        
        conversation = []
        for row in rows:
            conversation.append({
                "message_id": row["message_id"],
                "role": row["role"],
                "content": row["content"],
                "tool_calls": json.loads(row["tool_calls"]) if row["tool_calls"] else None,
                "is_error": row["is_error"],
                "timestamp": row["created_at"].isoformat() if hasattr(row["created_at"], 'isoformat') else str(row["created_at"]),
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            })
        
        logger.debug(f"Loaded {len(conversation)} messages for instance {instance_id}")
        return conversation
    except Exception as e:
        logger.warning(f"Failed to load conversation history: {e}", exc_info=True)
        return []


# ============================================================================
# Routes
# ============================================================================

@router.post("/initialize")
async def initialize_agent(config: AgentConfigRequest) -> Dict[str, Any]:
    """Initialize a new agent instance for the playground"""
    # Ensure database is initialized
    await ensure_database_initialized()
    
    try:
        instance_id = str(uuid.uuid4())
        agent_id = config.agent_id or str(uuid.uuid4())
        
        # Verify Ollama connection
        ollama = await get_ollama_client()
        health = await ollama.health_check()
        
        if health.get("status") != "healthy":
            logger.warning(f"Ollama not healthy: {health}")
            # Continue anyway for testing
        
        # Load existing conversation history from database
        conversation_history = await load_conversation_history(instance_id)
        
        # Store agent configuration
        _active_agents[instance_id] = {
            "instance_id": instance_id,
            "agent_id": agent_id,
            "config": config.model_dump(),
            "status": "idle",
            "started_at": datetime.utcnow().isoformat(),
            "conversation": conversation_history,
            "metrics": {
                "tokens_used": 0,
                "tool_calls": 0,
                "messages": len(conversation_history),
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


@router.post("/initialize-multiple")
async def initialize_multiple_agents(request: MultipleAgentsRequest) -> Dict[str, Any]:
    """Initialize multiple agent instances for the playground"""
    # Ensure database is initialized
    await ensure_database_initialized()
    
    results = {
        "total": len(request.agents),
        "successful": 0,
        "failed": 0,
        "instances": [],
        "errors": [],
    }
    
    for agent_config in request.agents:
        try:
            instance_id = str(uuid.uuid4())
            agent_id = agent_config.agent_id or str(uuid.uuid4())
            
            # Verify Ollama connection (only check once)
            if results["successful"] == 0 and results["failed"] == 0:
                ollama = await get_ollama_client()
                health = await ollama.health_check()
                
                if health.get("status") != "healthy":
                    logger.warning(f"Ollama not healthy: {health}")
                    # Continue anyway for testing
            
            # Load existing conversation history from database
            conversation_history = await load_conversation_history(instance_id)
            
            # Store agent configuration
            _active_agents[instance_id] = {
                "instance_id": instance_id,
                "agent_id": agent_id,
                "config": agent_config.model_dump(),
                "status": "idle",
                "started_at": datetime.utcnow().isoformat(),
                "conversation": conversation_history,
                "metrics": {
                    "tokens_used": 0,
                    "tool_calls": 0,
                    "messages": len(conversation_history),
                },
            }
            
            # Publish event
            try:
                broker = await get_redis_streams_broker()
                await broker.publish(
                    f"agent:{instance_id}",
                    StreamEventType.AGENT_STARTED,
                    {"agent_id": agent_id, "config": agent_config.model_dump()},
                )
            except Exception as e:
                logger.warning(f"Failed to publish event: {e}")
            
            logger.info(f"Initialized agent instance: {instance_id} ({agent_config.name})")
            
            results["instances"].append({
                "instance_id": instance_id,
                "agent_id": agent_id,
                "name": agent_config.name,
                "status": "idle",
                "model": agent_config.model,
                "tools": agent_config.tools,
                "started_at": _active_agents[instance_id]["started_at"],
            })
            results["successful"] += 1
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to initialize agent '{agent_config.name}': {e}", exc_info=True)
            results["errors"].append({
                "agent_name": agent_config.name,
                "error": error_msg,
            })
            results["failed"] += 1
    
    return results


@router.post("/invoke")
async def invoke_agent(request: AgentInvokeRequest):
    """Invoke an agent with input"""
    instance_id = request.instance_id
    
    if instance_id not in _active_agents:
        available_ids = list(_active_agents.keys())[:5]  # Show first 5 for debugging
        logger.warning(f"Agent instance not found: {instance_id}. Available instances: {available_ids}")
        raise HTTPException(
            status_code=404, 
            detail=f"Agent instance '{instance_id}' not found. Please initialize the agent first."
        )
    
    agent_data = _active_agents[instance_id]
    config = AgentConfigRequest(**agent_data["config"])
    
    # Update status
    agent_data["status"] = "processing"
    
    try:
        if request.stream:
            return StreamingResponse(
                stream_agent_response(instance_id, request.input, config),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming response
            response = await generate_agent_response(instance_id, request.input, config)
            agent_data["status"] = "idle"
            return response
    except Exception as e:
        agent_data["status"] = "error"
        logger.error(f"Error invoking agent {instance_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def stream_agent_response(
    instance_id: str,
    user_input: str,
    config: AgentConfigRequest,
):
    """Stream agent response token by token"""
    try:
        ollama = await get_ollama_client()
        # Force reload guardrails to ensure latest patterns (hot-reload support)
        guardrails = await get_advanced_guardrails(force_reload=True)
        
        agent_data = _active_agents.get(instance_id, {})
        conversation = agent_data.get("conversation", [])
        
        # Safety check
        safety_result = await guardrails.check_input(user_input)
        # Only block if action is BLOCK, allow WARN actions to pass through
        if not safety_result.passed and safety_result.action == GuardrailAction.BLOCK:
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
        
        # Add user message to conversation history
        user_message = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.utcnow().isoformat(),
        }
        conversation.append(user_message)
        
        # Save user message to database
        await save_message_to_db(
            instance_id=instance_id,
            agent_id=agent_data.get("agent_id", ""),
            role="user",
            content=user_input,
        )
        
        # Check if tools are enabled and available
        orchestrator = get_orchestrator()
        use_tools = len(config.tools) > 0 and orchestrator.agent_executor is not None
        full_response = ""
        tool_calls_count = 0
        
        if use_tools:
            # Use orchestrator with tool execution
            try:
                # Use orchestrator to process with tools
                result = await orchestrator.process_request(
                    user_input=user_input,
                    use_tools=True,
                    use_rag=False,  # Disable RAG for playground for now
                    max_tokens=config.max_tokens,
                )
                
                response_text = result.get("response", "")
                full_response = response_text
                intermediate_steps = result.get("intermediate_steps", [])
                tool_calls_count = len(intermediate_steps)
                
                # Emit tool call events
                for step in intermediate_steps:
                    if isinstance(step, tuple) and len(step) >= 2:
                        tool_name = str(step[0]) if step[0] else "unknown"
                        yield json.dumps({
                            "type": "tool_call",
                            "tool": tool_name,
                            "parameters": {},
                        }) + "\n"
                
                # Stream response in chunks
                chunk_size = 10
                for i in range(0, len(full_response), chunk_size):
                    chunk = full_response[i:i+chunk_size]
                    yield json.dumps({"type": "token", "content": chunk}) + "\n"
                    await asyncio.sleep(0.01)
                
            except Exception as tool_error:
                logger.warning(f"Tool execution failed, falling back to direct LLM: {tool_error}")
                use_tools = False
        
        if not use_tools:
            # Fallback to direct Ollama streaming (no tools)
            ollama = await get_ollama_client()
            
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
            
            # Generate streaming response
            options = GenerationOptions(
                temperature=config.temperature,
                num_predict=config.max_tokens,
            )
            
            token_count = 0
        
        try:
            logger.info(f"Starting stream for model: {config.model}, input length: {len(user_input)}")
            async for token in ollama.chat_stream(messages, model=config.model, options=options):
                token_count += 1
                full_response += token
                yield json.dumps({"type": "token", "content": token}) + "\n"
                await asyncio.sleep(0.01)  # Small delay for smooth streaming
            
            logger.info(f"Stream completed: {token_count} tokens, {len(full_response)} chars")
            
            # If no tokens were received, this is an error
            if token_count == 0:
                error_msg = f"No response generated from model '{config.model}'. The model may not be available or may have encountered an error."
                logger.warning(error_msg)
                yield json.dumps({
                    "type": "error",
                    "content": error_msg,
                }) + "\n"
                if instance_id in _active_agents:
                    _active_agents[instance_id]["status"] = "error"
                return
            
        except Exception as stream_error:
            error_msg = f"Streaming failed: {str(stream_error)}"
            logger.error(f"Streaming error during token generation: {stream_error}", exc_info=True)
            yield json.dumps({
                "type": "error",
                "content": error_msg,
            }) + "\n"
            if instance_id in _active_agents:
                _active_agents[instance_id]["status"] = "error"
            return
        
        # Add assistant response to history
        assistant_message = {
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.utcnow().isoformat(),
            "tool_calls": tool_calls_count > 0,  # Indicate if tools were used
        }
        conversation.append(assistant_message)
        
        # Save assistant message to database
        await save_message_to_db(
            instance_id=instance_id,
            agent_id=agent_data.get("agent_id", ""),
            role="assistant",
            content=full_response,
            tool_calls=[{"tool_calls_count": tool_calls_count}] if tool_calls_count > 0 else None,
        )
        
        # Update metrics
        agent_data["metrics"]["tokens_used"] += len(full_response) / 4
        agent_data["metrics"]["messages"] += 2
        agent_data["metrics"]["tool_calls"] += tool_calls_count
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
        error_msg = f"Failed to process request: {str(e)}"
        logger.error(f"Streaming error in stream_agent_response: {e}", exc_info=True)
        yield json.dumps({"type": "error", "content": error_msg}) + "\n"
        
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
        # Only block if action is BLOCK, allow WARN actions to pass through
        if not safety_result.passed and safety_result.action == GuardrailAction.BLOCK:
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


@router.get("/{instance_id}/conversation")
async def get_agent_conversation(instance_id: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """Get conversation history for an agent instance"""
    try:
        # Try to load from database first (persistent storage)
        conversation = await load_conversation_history(instance_id)
        
        # If limit is specified, return only the last N messages
        if limit and limit > 0:
            conversation = conversation[-limit:]
        
        return {
            "instance_id": instance_id,
            "messages": conversation,
            "total_messages": len(conversation),
        }
    except Exception as e:
        logger.error(f"Failed to load conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
        
        # If already connected, skip connection check and go straight to listing models
        connection_ok = ollama.is_connected
        if not connection_ok:
            try:
                # Try quick connection with timeout (only if not connected)
                connection_ok = await asyncio.wait_for(ollama.connect(), timeout=3.0)
                logger.info(f"Ollama connection attempt: {connection_ok}")
            except asyncio.TimeoutError:
                logger.debug("Connection attempt timed out, but will try to list models anyway")
                connection_ok = False
            except Exception as connect_err:
                logger.debug(f"Connection attempt failed: {connect_err}, but will try to list models anyway")
                connection_ok = False
        
        # Try to get models - list_models() uses parallel URL checks with 5s timeout per URL
        # Parallel execution means total time is ~5s max (not 5s * number of URLs)
        models = []
        try:
            # list_models() uses parallel checks with 5s timeout, so 8s here gives buffer
            models = await asyncio.wait_for(ollama.list_models(), timeout=8.0)
            # If we got models, we're definitely connected
            if models:
                connection_ok = True
                logger.info(f"Successfully listed {len(models)} models from Ollama")
        except asyncio.TimeoutError:
            logger.warning("Model listing timed out at endpoint level")
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

