"""
Agent Playground API Routes
API endpoints for the Agent Playground UI
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio
import json
import uuid
from typing import Optional

from ..core.prompt_templates import get_prompt_template_manager, PromptCategory
from ..core.queue_system import get_queue_producer, create_queue_consumer, QueuePriority
from ..core.enhanced_state_manager import get_enhanced_state_manager, WorkflowPhase
from ..core.redis_streams_broker import get_redis_streams_broker, StreamEventType
from ..core.advanced_guardrails import get_advanced_guardrails, GuardrailAction
from ..core.orchestrator import get_orchestrator
from ..core.tool_registry import get_tool_registry, ToolCategory
from ..integrations.ollama_container import get_ollama_client, ChatMessage, GenerationOptions
from ..integrations.realtime_streaming import get_stream_publisher
from ..integrations.messaging_service_client import get_messaging_client
from ..agents.base_agent import BaseAgent, AgentResponse
from ..agents.agent_factory import create_agent
from ..agents.tools.enhanced_memory_tools import EnhancedMemoryTools
from ..agents.tools.comprehensive_api_tools import ComprehensiveAPITools
from ..database.agent_tables import initialize_agent_database
from ..database.postgres import get_postgres_manager
from ..logging_config import get_logger
from ..services.document_parser_service import get_document_parser_service

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
    model: str = "mistral:7b"  # Default model changed to Mistral
    temperature: float = 0.7
    max_tokens: int = 2000
    system_prompt: str = """You are a helpful, intelligent AI assistant with access to tools. Your goal is to provide accurate, useful, and contextually appropriate responses.

CRITICAL TOOL USAGE RULES:
- When asked to edit, set, write, or update spreadsheet cells, you MUST use the spreadsheet_set_cell tool
- When asked to read or get spreadsheet cell values, you MUST use the spreadsheet_get_cell tool
- NEVER claim you did something unless you actually called the tool - check tool results before confirming
- If a tool call fails, explain the error to the user
- After calling a tool, report the actual result from the tool, not what you think happened

Guidelines:
- Be clear, concise, and direct in your responses
- If you don't know something, admit it rather than guessing
- Ask clarifying questions when the user's request is ambiguous
- Maintain context from the conversation history
- Be professional but friendly in your tone
- Focus on being helpful and solving the user's problem
- When working with spreadsheets, ALWAYS use the available spreadsheet tools:
  * spreadsheet_set_cell(cell_id, value) - to set a cell value (e.g., "J7", "1.2")
  * spreadsheet_get_cell(cell_id) - to read a cell value
  * spreadsheet_sum_range(start_cell, end_cell, target_cell) - to sum a range
  * spreadsheet_avg_range(start_cell, end_cell, target_cell) - to average a range
- Cell references like "J7" mean column J, row 7

Remember: Quality over quantity. Better to give a short, accurate answer than a long, rambling one. ALWAYS use tools when available - don't just describe what you would do."""
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
# Tool Registration Helpers
# ============================================================================

async def _register_spreadsheet_tools_for_instance(instance_id: str):
    """Register spreadsheet tools for a specific instance with the ToolRegistry"""
    try:
        from langchain_core.tools import Tool, StructuredTool
        from ..agents.tools.spreadsheet_tools import (
            _get_spreadsheet_data,
            _save_spreadsheet_data,
        )
        import asyncio
        
        tool_registry = get_tool_registry()
        
        # Capture the running event loop so sync wrappers can schedule
        # coroutines on it (required for asyncpg connection pool access)
        _main_loop = asyncio.get_running_loop()
        
        # Create instance-aware tool functions
        async def set_cell_tool(cell_id: str, value: str) -> str:
            """Set a cell value in the spreadsheet"""
            try:
                spreadsheet = await _get_spreadsheet_data(instance_id)
                data = spreadsheet["data"]
                columns = spreadsheet.get("columns", [])
                row_count = spreadsheet.get("row_count", 1000)
                
                if cell_id not in data:
                    data[cell_id] = {"id": cell_id, "value": ""}
                
                cell = data[cell_id]
                if value.startswith("="):
                    cell["formula"] = value[1:]
                    cell["value"] = ""
                else:
                    cell["value"] = value
                    cell["formula"] = None
                
                # Save with columns and row_count preserved
                await _save_spreadsheet_data(instance_id, data, columns=columns, row_count=row_count)
                logger.info(f"Tool set_cell: Saved cell {cell_id}={value} for instance {instance_id}")
                return json.dumps({"success": True, "cell_id": cell_id, "value": value})
            except Exception as e:
                logger.error(f"Tool set_cell failed: {e}", exc_info=True)
                return json.dumps({"success": False, "error": str(e)})
        
        def set_cell_sync(cell_id: str, value: str) -> str:
            """Sync wrapper for set_cell - runs on the main event loop where asyncpg lives"""
            future = asyncio.run_coroutine_threadsafe(set_cell_tool(cell_id, value), _main_loop)
            return future.result(timeout=30)
        
        async def get_cell_tool(cell_id: str) -> str:
            """Get a cell value from the spreadsheet"""
            try:
                spreadsheet = await _get_spreadsheet_data(instance_id)
                data = spreadsheet["data"]
                cell = data.get(cell_id, {})
                return json.dumps({
                    "success": True,
                    "cell_id": cell_id,
                    "value": cell.get("value", ""),
                    "formula": cell.get("formula"),
                })
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})
        
        def get_cell_sync(cell_id: str) -> str:
            """Sync wrapper for get_cell - runs on the main event loop where asyncpg lives"""
            future = asyncio.run_coroutine_threadsafe(get_cell_tool(cell_id), _main_loop)
            return future.result(timeout=30)
        
        # Register tools - these are instance-aware via closure
        # Note: If multiple instances register, the last one wins, but that's OK since
        # each instance's tools capture their own instance_id in the closure
        existing_set_cell = tool_registry.get_tool("spreadsheet_set_cell")
        existing_get_cell = tool_registry.get_tool("spreadsheet_get_cell")
        
        if existing_set_cell:
            logger.debug(f"Tool spreadsheet_set_cell already registered, overwriting for instance {instance_id}")
        if existing_get_cell:
            logger.debug(f"Tool spreadsheet_get_cell already registered, overwriting for instance {instance_id}")
        
        # Use StructuredTool so LangGraph gets proper argument schemas
        # (Tool() treats everything as single string input, which breaks native tool calling)
        tool_registry.register_tool(
            StructuredTool.from_function(
                func=set_cell_sync,
                name="spreadsheet_set_cell",
                description="Set a cell value in the spreadsheet. cell_id: cell reference like 'A1', 'B2', 'J7'. value: the value to set (number, text, or formula starting with =).",
            ),
            category=ToolCategory.DATA,
        )
        
        tool_registry.register_tool(
            StructuredTool.from_function(
                func=get_cell_sync,
                name="spreadsheet_get_cell",
                description="Get a cell value from the spreadsheet. cell_id: cell reference like 'A1', 'B2', 'J7'.",
            ),
            category=ToolCategory.DATA,
        )
        
        logger.info(f"Registered spreadsheet tools for instance {instance_id}")
    except Exception as e:
        logger.warning(f"Failed to register spreadsheet tools for instance {instance_id}: {e}", exc_info=True)


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
        
        # Get orchestrator for tool initialization
        orchestrator = get_orchestrator()
        
        # Initialize agent tools from app/agents/tools (once per orchestrator)
        try:
            await orchestrator.initialize_agent_tools()
            logger.info("Agent tools initialized from app/agents/tools")
        except Exception as e:
            logger.warning(f"Failed to initialize agent tools: {e}")
        
        # Load specialized prompt based on agent_type
        from ..core.agent_integration import load_prompt_for_agent_type
        
        # Map agent_type to prompt template if provided
        final_config = config.model_dump()
        if config.agent_type and config.agent_type != "conversational":
            # Try to load specialized prompt from app/agents/prompts first
            if config.system_prompt == "You are a helpful AI assistant.":
                specialized_prompt = load_prompt_for_agent_type(
                    config.agent_type,
                    fallback_prompt=config.system_prompt
                )
                final_config["system_prompt"] = specialized_prompt
                logger.info(f"Loaded specialized prompt for agent_type: {config.agent_type}")
            else:
                # User provided custom prompt, check template manager as fallback
                template_manager = get_prompt_template_manager()
                
                # Map agent_type to template key
                type_to_template = {
                    "task_decomposer": "task_decomposition",
                    "code_generator": "code_generation",
                    "data_analyst": "data_analysis",
                    "vendor_fraud": "vendor_fraud",
                    "document_processing": "document_processing",
                    "automation": "automation",
                    "tool_use": "tool_use",
                    "rag_query": "rag_query",
                }
                
                template_key = type_to_template.get(config.agent_type)
                if template_key:
                    template = template_manager.get_template(template_key)
                    if template:
                        # Use template's system prompt and settings
                        rendered = template.render({})
                        final_config["system_prompt"] = rendered["system"]
                        
                        # Use template's temperature and max_tokens if not overridden
                        if config.temperature == 0.7:  # Default value
                            final_config["temperature"] = template.temperature
                        if config.max_tokens == 2000:  # Default value
                            final_config["max_tokens"] = template.max_tokens
                        
                        # Add tools from template if not already specified
                        if not config.tools and template.tools:
                            final_config["tools"] = [tool.name for tool in template.tools]
                        
                        logger.info(f"Using prompt template '{template_key}' for agent type '{config.agent_type}'")
        
        # Load existing conversation history from database
        conversation_history = await load_conversation_history(instance_id)
        
        # Register spreadsheet tools for this instance with ToolRegistry
        # This ensures the orchestrator's agent executor can use them
        if any(tool.startswith("spreadsheet_") for tool in final_config.get("tools", [])):
            await _register_spreadsheet_tools_for_instance(instance_id)
            logger.info(f"Registered spreadsheet tools for instance {instance_id}")
        
        # Store agent configuration
        _active_agents[instance_id] = {
            "instance_id": instance_id,
            "agent_id": agent_id,
            "config": final_config,
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
                {"agent_id": agent_id, "config": final_config},
            )
        except Exception as e:
            logger.warning(f"Failed to publish event: {e}")
        
        logger.info(f"Initialized agent instance: {instance_id} (type: {config.agent_type})")
        
        return {
            "instance_id": instance_id,
            "agent_id": agent_id,
            "status": "idle",
            "model": final_config["model"],
            "tools": final_config["tools"],
            "agent_type": config.agent_type,
            "name": final_config.get("name", "Test Agent"),
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
            
            # Map agent_type to prompt template if provided
            final_config = agent_config.model_dump()
            if agent_config.agent_type and agent_config.agent_type != "conversational":
                template_manager = get_prompt_template_manager()
                
                # Map agent_type to template key
                type_to_template = {
                    "task_decomposer": "task_decomposition",
                    "code_generator": "code_generation",
                    "data_analyst": "data_analysis",
                    "vendor_fraud": "vendor_fraud",
                    "document_processing": "document_processing",
                    "automation": "automation",
                    "tool_use": "tool_use",
                    "rag_query": "rag_query",
                }
                
                template_key = type_to_template.get(agent_config.agent_type)
                if template_key:
                    template = template_manager.get_template(template_key)
                    if template:
                        # Use template's system prompt and settings
                        rendered = template.render({})
                        final_config["system_prompt"] = rendered["system"]
                        
                        # Use template's temperature and max_tokens if not overridden
                        if agent_config.temperature == 0.7:  # Default value
                            final_config["temperature"] = template.temperature
                        if agent_config.max_tokens == 2000:  # Default value
                            final_config["max_tokens"] = template.max_tokens
                        
                        # Add tools from template if not already specified
                        if not agent_config.tools and template.tools:
                            final_config["tools"] = [tool.name for tool in template.tools]
            
            # Load existing conversation history from database
            conversation_history = await load_conversation_history(instance_id)
            
            # Store agent configuration
            _active_agents[instance_id] = {
                "instance_id": instance_id,
                "agent_id": agent_id,
                "config": final_config,
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
        # Extract chat_room_id from context if provided
        chat_room_id = request.context.get("chat_room_id") if request.context else None
        
        if request.stream:
            return StreamingResponse(
                stream_agent_response(instance_id, request.input, config, chat_room_id),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming response
            response = await generate_agent_response(instance_id, request.input, config)
            agent_data["status"] = "idle"
            
            # If chat_room_id is provided in context, send response to messaging service
            chat_room_id = request.context.get("chat_room_id")
            if chat_room_id and response.get("success"):
                try:
                    messaging_client = get_messaging_client()
                    await messaging_client.send_agent_message(
                        chat_room_id=chat_room_id,
                        content=response.get("response", ""),
                        agent_instance_id=instance_id,
                        message_type="TEXT",
                        metadata={
                            "tokens_used": response.get("tokens_used", 0),
                            "tool_calls": response.get("tool_calls", []),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to send response to messaging service: {e}")
            
            return response
    except Exception as e:
        agent_data["status"] = "error"
        logger.error(f"Error invoking agent {instance_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def stream_agent_response(
    instance_id: str,
    user_input: str,
    config: AgentConfigRequest,
    chat_room_id: Optional[str] = None,
):
    """Stream agent response token by token"""
    try:
        guardrails = await get_advanced_guardrails()
        
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
        tools_available = orchestrator.tool_registry.get_tools()
        logger.info(f"Tools in registry: {[t.name for t in tools_available]}")
        logger.info(f"Config tools requested: {config.tools}")
        langgraph_ok = getattr(orchestrator, '_langgraph_agent_available', False)
        executor_ok = getattr(orchestrator, 'agent_executor', None) is not None
        logger.info(f"LangGraph agent available: {langgraph_ok}, legacy executor: {executor_ok}")
        
        use_tools = len(config.tools) > 0 and (langgraph_ok or executor_ok)
        full_response = ""
        tool_calls_count = 0
        
        if use_tools:
            logger.info(f"Using agent executor with tools for request: {user_input}")
            # Use orchestrator with tool execution
            try:
                # Use orchestrator to process with tools
                # Pass system prompt directly to orchestrator
                result = await orchestrator.process_request(
                    user_input=user_input,
                    use_tools=True,
                    use_rag=False,
                    max_tokens=config.max_tokens,
                    system_prompt=config.system_prompt,
                    model=config.model,
                    temperature=config.temperature,
                )
                
                response_text = result.get("response", "")
                full_response = response_text
                intermediate_steps = result.get("intermediate_steps", [])
                tool_calls_count = len(intermediate_steps)
                
                logger.info(f"Agent executor returned {tool_calls_count} intermediate steps")
                if intermediate_steps:
                    logger.info(f"Intermediate steps: {intermediate_steps}")
                else:
                    logger.warning(f"No tool calls made by agent executor. Response: {response_text[:200]}")
                
                # Emit tool call and result events
                for step in intermediate_steps:
                    tool_name = "unknown"
                    parameters = {}
                    observation = None

                    if isinstance(step, dict):
                        # LangGraph format: {"tool": "...", "output": "..."}
                        tool_name = step.get("tool", "unknown")
                        observation = step.get("output", "")
                    elif isinstance(step, tuple) and len(step) >= 2:
                        # Legacy AgentExecutor format: (AgentAction, observation)
                        action = step[0]
                        observation = step[1]
                        if hasattr(action, 'tool'):
                            tool_name = action.tool
                        elif isinstance(action, dict):
                            tool_name = action.get('tool', str(action))
                        if hasattr(action, 'tool_input'):
                            parameters = action.tool_input if isinstance(action.tool_input, dict) else {}
                    else:
                        continue
                        
                        # Emit tool_call event
                        yield json.dumps({
                            "type": "tool_call",
                            "tool": tool_name,
                            "parameters": parameters,
                        }) + "\n"
                        
                    # Parse observation if JSON string
                    parsed_result = observation
                    if isinstance(observation, str):
                        try:
                            parsed_result = json.loads(observation)
                        except (json.JSONDecodeError, TypeError):
                            parsed_result = observation

                        # Emit tool_result event
                        yield json.dumps({
                            "type": "tool_result",
                            "tool": tool_name,
                            "parameters": parameters,
                        "result": parsed_result if parsed_result is not None else {},
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
            tool_calls=[
                {"tool": s.get("tool", "unknown"), "output": s.get("output", "")}
                if isinstance(s, dict) else {"tool": "unknown"}
                for s in intermediate_steps
            ] if tool_calls_count > 0 else None,
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
        
        # If chat_room_id is provided, send final response to messaging service
        if chat_room_id:
            try:
                messaging_client = get_messaging_client()
                await messaging_client.send_agent_message(
                    chat_room_id=chat_room_id,
                    content=full_response,
                    agent_instance_id=instance_id,
                    message_type="TEXT",
                    metadata={
                        "tool_calls": tool_calls_count > 0,
                        "intermediate_steps_count": tool_calls_count,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to send streaming response to messaging service: {e}")
    
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
    """Generate non-streaming agent response (uses orchestrator with LangGraph tool calling)"""
    try:
        agent_data = _active_agents.get(instance_id, {})
        conversation = agent_data.get("conversation", [])
        
        # Add to conversation
        conversation.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Use the orchestrator (same path as streaming) so tools actually get called
        orchestrator = get_orchestrator()
        has_tools = bool(config.tools)
        result = await orchestrator.process_request(
            user_input=user_input,
            use_tools=has_tools,
            use_rag=False,
            max_tokens=config.max_tokens,
            system_prompt=config.system_prompt,
            model=config.model,
            temperature=config.temperature,
        )
        
        response_text = result.get("response", "")
        intermediate_steps = result.get("intermediate_steps", [])
        
        # Add to conversation
        conversation.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Update metrics
        agent_data["metrics"]["tokens_used"] += len(response_text.split())
        agent_data["metrics"]["messages"] += 2
        
        return {
            "success": result.get("success", True),
            "response": response_text,
            "tokens_used": len(response_text.split()),
            "duration_ms": result.get("duration_ms", 0),
            "tool_calls": [
                {"tool": s.get("tool", "unknown"), "output": s.get("output", "")}
                if isinstance(s, dict) else {"tool": "unknown"}
                for s in intermediate_steps
            ],
            "intermediate_steps": intermediate_steps,
        }
    
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


@router.post("/{instance_id}/stop")
async def stop_agent(instance_id: str) -> Dict[str, Any]:
    """Stop an agent instance (idempotent - safe to call multiple times)"""
    # Check if agent exists
    if instance_id not in _active_agents:
        # Already stopped or never existed - return success (idempotent)
        logger.debug(f"Agent instance {instance_id} not found (may already be stopped)")
        return {"success": True, "message": "Agent already stopped or not found"}
    
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
        # Continue anyway - don't fail the stop request if event publishing fails
    
    # Remove from active agents
    try:
        del _active_agents[instance_id]
        logger.debug(f"Stopped agent instance: {instance_id}")
    except KeyError:
        # Already deleted (race condition) - that's fine
        logger.info(f"Agent instance {instance_id} already removed")
    
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
            "name": data["config"].get("name", "Agent"),
            "status": data["status"],
            "model": data["config"]["model"],
            "agent_type": data["config"].get("agent_type", "conversational"),
            "tools": data["config"].get("tools", []),
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


@router.get("/agent-types")
async def list_agent_types() -> Dict[str, Any]:
    """List available agent types with their prompt templates"""
    manager = get_prompt_template_manager()
    
    # Map agent types to template keys
    agent_types = [
        {
            "id": "conversational",
            "name": "Conversational Agent",
            "description": "General conversation and assistance",
            "template_key": "conversation",
        },
        {
            "id": "task_decomposer",
            "name": "Task Decomposer",
            "description": "Break down complex tasks into manageable subtasks",
            "template_key": "task_decomposition",
        },
        {
            "id": "code_generator",
            "name": "Code Generator",
            "description": "Generate code based on requirements",
            "template_key": "code_generation",
        },
        {
            "id": "data_analyst",
            "name": "Data Analyst",
            "description": "Analyze data and provide insights",
            "template_key": "data_analysis",
        },
        {
            "id": "vendor_fraud",
            "name": "Vendor Fraud Detector",
            "description": "Analyze invoices and detect potential fraud",
            "template_key": "vendor_fraud",
        },
        {
            "id": "document_processing",
            "name": "Document Processor",
            "description": "Extract and structure information from documents",
            "template_key": "document_processing",
        },
        {
            "id": "automation",
            "name": "Automation Agent",
            "description": "Plan and execute automation workflows",
            "template_key": "automation",
        },
        {
            "id": "tool_use",
            "name": "Tool Use Agent",
            "description": "Agent that can use tools to accomplish tasks",
            "template_key": "tool_use",
        },
        {
            "id": "rag_query",
            "name": "RAG Query Agent",
            "description": "Answer questions using retrieved context",
            "template_key": "rag_query",
        },
    ]
    
    # Enrich with template information
    for agent_type in agent_types:
        template = manager.get_template(agent_type["template_key"])
        if template:
            agent_type["temperature"] = template.temperature
            agent_type["max_tokens"] = template.max_tokens
            agent_type["tools"] = [tool.name for tool in template.tools] if template.tools else []
            agent_type["template_id"] = template.template_id
    
    return {
        "agent_types": agent_types,
    }


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


@router.post("/spreadsheet/{instance_id}/set-cell")
async def spreadsheet_set_cell(instance_id: str, cell_id: str, value: str):
    """Set a cell value in the spreadsheet"""
    try:
        from ..agents.tools.spreadsheet_tools import _spreadsheet_states
        if instance_id not in _spreadsheet_states:
            _spreadsheet_states[instance_id] = {}
        
        state = _spreadsheet_states[instance_id]
        state[cell_id] = {
            "value": value,
            "formula": value[1:] if value.startswith("=") else None,
        }
        
        return {
            "success": True,
            "cell_id": cell_id,
            "value": value,
        }
    except Exception as e:
        logger.error(f"Failed to set cell: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spreadsheet/{instance_id}/get-cell")
async def spreadsheet_get_cell(instance_id: str, cell_id: str):
    """Get a cell value from the spreadsheet"""
    try:
        from ..agents.tools.spreadsheet_tools import _spreadsheet_states
        state = _spreadsheet_states.get(instance_id, {})
        cell = state.get(cell_id, {})
        
        return {
            "success": True,
            "cell_id": cell_id,
            "value": cell.get("value", ""),
            "formula": cell.get("formula"),
        }
    except Exception as e:
        logger.error(f"Failed to get cell: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spreadsheet/{instance_id}/state")
async def spreadsheet_get_state(instance_id: str):
    """Get entire spreadsheet state"""
    try:
        from ..agents.tools.spreadsheet_tools import get_spreadsheet_state
        state = get_spreadsheet_state(instance_id)
        return {
            "success": True,
            "state": state,
        }
    except Exception as e:
        logger.error(f"Failed to get spreadsheet state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Spreadsheet Data Storage (PostgreSQL with user_id)
# ============================================================================

class SpreadsheetSaveRequest(BaseModel):
    """Request to save spreadsheet data"""
    user_id: str = Field(default="admin", description="User ID (defaults to 'admin' for testing)")
    instance_id: Optional[str] = None
    agent_name: Optional[str] = None
    columns: List[str]
    row_count: int
    data: Dict[str, Any]


class SpreadsheetLoadRequest(BaseModel):
    """Request to load spreadsheet data"""
    user_id: str = Field(default="admin", description="User ID (defaults to 'admin' for testing)")
    instance_id: Optional[str] = None


@router.post("/spreadsheet/save")
async def save_spreadsheet_data(request: SpreadsheetSaveRequest):
    """Save spreadsheet data to PostgreSQL with user_id"""
    try:
        await ensure_database_initialized()
        postgres = await get_postgres_manager()
        
        # Generate spreadsheet_id from user_id and instance_id
        spreadsheet_id = f"{request.user_id}_{request.instance_id or 'default'}"
        
        # Save or update spreadsheet data
        await postgres.execute("""
            INSERT INTO cyrex.spreadsheet_data 
            (spreadsheet_id, user_id, instance_id, agent_name, columns, row_count, data, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            ON CONFLICT (spreadsheet_id) DO UPDATE SET
                columns = EXCLUDED.columns,
                row_count = EXCLUDED.row_count,
                data = EXCLUDED.data,
                agent_name = EXCLUDED.agent_name,
                updated_at = NOW()
        """,
            spreadsheet_id,
            request.user_id,
            request.instance_id,
            request.agent_name,
            json.dumps(request.columns),
            request.row_count,
            json.dumps(request.data),
        )
        
        logger.info(f"Saved spreadsheet data for user {request.user_id}, instance {request.instance_id}")
        return {
            "success": True,
            "spreadsheet_id": spreadsheet_id,
            "message": "Spreadsheet data saved successfully"
        }
    except Exception as e:
        logger.error(f"Failed to save spreadsheet data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/spreadsheet/load")
async def load_spreadsheet_data(request: SpreadsheetLoadRequest):
    """Load spreadsheet data from PostgreSQL by user_id"""
    try:
        await ensure_database_initialized()
        postgres = await get_postgres_manager()
        
        # Generate spreadsheet_id from user_id and instance_id
        spreadsheet_id = f"{request.user_id}_{request.instance_id or 'default'}"
        
        # Load spreadsheet data
        row = await postgres.fetchrow("""
            SELECT columns, row_count, data, agent_name, instance_id
            FROM cyrex.spreadsheet_data
            WHERE spreadsheet_id = $1
        """, spreadsheet_id)
        
        if not row:
            # Return empty spreadsheet if not found
            # Return A-Z columns (26 columns) and 1000 rows by default
            default_columns = [chr(65 + i) for i in range(26)]  # A-Z
            return {
                "success": True,
                "found": False,
                "columns": default_columns,
                "row_count": 1000,
                "data": {},
                "agent_name": request.agent_name,
                "instance_id": request.instance_id
            }
        
        return {
            "success": True,
            "found": True,
            "columns": json.loads(row['columns']) if isinstance(row['columns'], str) else row['columns'],
            "row_count": row['row_count'],
            "data": json.loads(row['data']) if isinstance(row['data'], str) else row['data'],
            "agent_name": row['agent_name'],
            "instance_id": row['instance_id']
        }
    except Exception as e:
        logger.error(f"Failed to load spreadsheet data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spreadsheet/list/{user_id}")
async def list_user_spreadsheets(user_id: str):
    """List all spreadsheets for a user"""
    try:
        await ensure_database_initialized()
        postgres = await get_postgres_manager()
        
        rows = await postgres.fetch("""
            SELECT spreadsheet_id, instance_id, agent_name, row_count, updated_at
            FROM cyrex.spreadsheet_data
            WHERE user_id = $1
            ORDER BY updated_at DESC
        """, user_id)
        
        return {
            "success": True,
            "spreadsheets": [
                {
                    "spreadsheet_id": row['spreadsheet_id'],
                    "instance_id": row['instance_id'],
                    "agent_name": row['agent_name'],
                    "row_count": row['row_count'],
                    "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None
                }
                for row in rows
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list spreadsheets: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/spreadsheet/parse-document")
async def parse_document_for_spreadsheet(
    file: UploadFile = File(...),
    instance_id: Optional[str] = Form(None),
    user_id: str = Form("admin"),
    use_ocr: bool = Form(True),
    extract_tables: bool = Form(True),
    detect_layout: bool = Form(True),  # Enable advanced analysis by default
    ocr_language: Optional[str] = Form(None),  # e.g., 'eng', 'spa', 'eng+spa'
    use_doclaynet: bool = Form(False),  # Use DocLayNet for advanced layout
    start_cell: str = Form("A1"),
):
    """
    Parse a document and extract data for spreadsheet import
    
    Supports: PDF, DOCX, XLSX, CSV, TXT, MD, PNG, JPG, TIFF, HTML
    
    Args:
        file: Uploaded document file
        instance_id: Optional spreadsheet instance ID
        use_ocr: Whether to use OCR for images/scanned PDFs
        extract_tables: Whether to extract tables from documents
        detect_layout: Whether to perform layout analysis and ML-based classification (enabled by default)
        start_cell: Starting cell for data insertion (e.g., "A1")
    
    Returns:
        Parsed document data ready for spreadsheet import
    """
    try:
        # Read file content
        file_content = await file.read()
        filename = file.filename or "unknown"
        
        # Validate file size (max 50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is 50MB, got {len(file_content) / 1024 / 1024:.2f}MB"
            )
        
        # Get learned template if available
        from ..services.template_learning_service import get_template_learning_service
        learning_service = get_template_learning_service()
        template = await learning_service.get_template(user_id, "general")  # Will be updated after classification
        
        # Parse document
        parser_service = get_document_parser_service()
        parsed = await parser_service.parse_document(
            file_content=file_content,
            filename=filename,
            use_ocr=use_ocr,
            extract_tables=extract_tables,
            detect_layout=detect_layout,
            ocr_language=ocr_language,
            use_doclaynet=use_doclaynet,
        )
        
        # Apply learned template if available and category matches
        if parsed.document_category and template:
            category_template = await learning_service.get_template(user_id, parsed.document_category)
            if category_template:
                # Apply learned field mappings
                if category_template['field_mappings']:
                    for field, mapping in category_template['field_mappings'].items():
                        if field in parsed.key_value_pairs:
                            # Apply learned correction patterns
                            pass  # Could apply corrections here
                
                # Apply learned column mappings
                if category_template['column_mappings']:
                    if parsed.column_mapping:
                        parsed.column_mapping.update(category_template['column_mappings'])
                    else:
                        parsed.column_mapping = category_template['column_mappings'].copy()
        
        # Mark template success if extraction looks good
        if parsed.document_category and parsed.confidence_scores.get('overall', 0) > 0.7:
            category_template = await learning_service.get_template(user_id, parsed.document_category)
            if category_template:
                await learning_service.mark_success(category_template['template_id'])
        
        # Convert to spreadsheet format
        spreadsheet_data = parser_service.to_spreadsheet_format(parsed, start_cell=start_cell)
        
        # Prepare response
        response = {
            "success": True,
            "document_type": parsed.document_type.value if parsed.document_type else "unknown",
            "document_category": parsed.document_category,  # ML-classified category
            "filename": filename,
            "parsed_data": {
                "tables": parsed.tables,
                "text_sections": parsed.text_sections,
                "key_value_pairs": parsed.key_value_pairs,
                "raw_text_preview": parsed.raw_text[:500] if parsed.raw_text else "",  # Preview only
                "layout_elements": parsed.layout_elements,  # Layout structure
            },
            "spreadsheet_mapping": spreadsheet_data,
            "column_mapping": parsed.column_mapping,  # Smart column suggestions
            "confidence_scores": parsed.confidence_scores,
            "metadata": parsed.metadata,
            "warnings": [],
        }
        
        # Add warnings if needed
        if parsed.confidence_scores.get('overall', 0) < 0.5:
            response["warnings"].append("Low confidence in extraction. Please review the data.")
        if parsed.metadata.get('ocr_used'):
            response["warnings"].append("OCR was used. Accuracy may be lower than text-based extraction.")
        
        logger.info(f"Successfully parsed document: {filename}, type: {parsed.document_type.value}")
        return response
        
    except ValueError as e:
        logger.error(f"Invalid document format: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        logger.error(f"Missing dependency: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Missing required dependency. Please install: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error parsing document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to parse document: {str(e)}")


@router.post("/spreadsheet/parse-document-batch")
async def parse_documents_batch(
    files: List[UploadFile] = File(...),
    instance_id: Optional[str] = Form(None),
    user_id: str = Form("admin"),
    use_ocr: bool = Form(True),
    extract_tables: bool = Form(True),
    detect_layout: bool = Form(True),
    ocr_language: Optional[str] = Form(None),
    use_doclaynet: bool = Form(False),
    start_cell: str = Form("A1"),
):
    """
    Parse multiple documents in batch
    
    Args:
        files: List of uploaded document files
        instance_id: Optional spreadsheet instance ID
        user_id: User ID for template learning
        use_ocr: Whether to use OCR
        extract_tables: Whether to extract tables
        detect_layout: Whether to perform layout analysis
        ocr_language: OCR language code
        use_doclaynet: Use DocLayNet for layout
        start_cell: Starting cell for data insertion
    
    Returns:
        List of parsed documents with results
    """
    try:
        if len(files) > 10:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
        
        results = []
        parser_service = get_document_parser_service()
        
        for idx, file in enumerate(files):
            try:
                file_content = await file.read()
                filename = file.filename or f"document_{idx + 1}"
                
                # Validate file size
                max_size = 50 * 1024 * 1024  # 50MB
                if len(file_content) > max_size:
                    results.append({
                        "filename": filename,
                        "success": False,
                        "error": f"File too large: {len(file_content) / 1024 / 1024:.2f}MB",
                    })
                    continue
                
                # Parse document
                parsed = await parser_service.parse_document(
                    file_content=file_content,
                    filename=filename,
                    use_ocr=use_ocr,
                    extract_tables=extract_tables,
                    detect_layout=detect_layout,
                    ocr_language=ocr_language,
                    use_doclaynet=use_doclaynet,
                )
                
                # Convert to spreadsheet format
                spreadsheet_data = parser_service.to_spreadsheet_format(parsed, start_cell=start_cell)
                
                results.append({
                    "filename": filename,
                    "success": True,
                    "document_type": parsed.document_type.value if parsed.document_type else "unknown",
                    "document_category": parsed.document_category,
                    "spreadsheet_mapping": spreadsheet_data,
                    "confidence_scores": parsed.confidence_scores,
                    "metadata": parsed.metadata,
                })
                
            except Exception as e:
                logger.error(f"Error parsing file {filename}: {e}", exc_info=True)
                results.append({
                    "filename": filename,
                    "success": False,
                    "error": str(e),
                })
        
        return {
            "success": True,
            "total_files": len(files),
            "successful": sum(1 for r in results if r.get("success")),
            "failed": sum(1 for r in results if not r.get("success")),
            "results": results,
        }
        
    except Exception as e:
        logger.error(f"Error in batch parsing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to parse documents: {str(e)}")


@router.post("/spreadsheet/save-correction")
async def save_document_correction(
    user_id: str,
    document_category: str,
    original_extraction: Dict[str, Any],
    corrected_data: Dict[str, Any],
    correction_type: str = "field_mapping",
    correction_details: Optional[Dict[str, Any]] = None,
):
    """
    Save a user correction for template learning
    
    Args:
        user_id: User ID
        document_category: Document category (invoice, receipt, etc.)
        original_extraction: Original extracted data
        corrected_data: User-corrected data
        correction_type: Type of correction
        correction_details: Additional correction details
    
    Returns:
        Correction ID and learning status
    """
    try:
        from ..services.template_learning_service import get_template_learning_service
        
        learning_service = get_template_learning_service()
        correction_id = await learning_service.save_correction(
            user_id=user_id,
            document_category=document_category,
            original_extraction=original_extraction,
            corrected_data=corrected_data,
            correction_type=correction_type,
            correction_details=correction_details,
        )
        
        # Learn from corrections
        learned_patterns = await learning_service.learn_from_corrections(
            user_id=user_id,
            document_category=document_category,
        )
        
        return {
            "success": True,
            "correction_id": correction_id,
            "learned_patterns": learned_patterns,
            "message": "Correction saved and patterns learned",
        }
        
    except Exception as e:
        logger.error(f"Error saving correction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save correction: {str(e)}")


# ============================================================================
# Group Chat Endpoints
# ============================================================================

class GroupChatCreateRequest(BaseModel):
    """Request to create a group chat"""
    name: str
    agent_instance_ids: List[str] = Field(..., description="List of agent instance IDs to include in the group chat")


class GroupChatMessageRequest(BaseModel):
    """Request to send a message to a group chat"""
    message: str
    stream: bool = True
    max_rounds: int = Field(default=0, description="Maximum conversation rounds (0 = single response, >0 = agents continue talking)")


@router.post("/group-chat/create")
async def create_group_chat(request: GroupChatCreateRequest) -> Dict[str, Any]:
    """Create a new group chat with multiple agents"""
    try:
        await ensure_database_initialized()
        
        # Validate that all agent instances exist
        missing_agents = []
        agent_details = []
        for instance_id in request.agent_instance_ids:
            if instance_id not in _active_agents:
                missing_agents.append(instance_id)
            else:
                agent_data = _active_agents[instance_id]
                agent_details.append({
                    "instance_id": instance_id,
                    "agent_id": agent_data["agent_id"],
                    "name": agent_data["config"].get("name", "Agent"),
                    "agent_type": agent_data["config"].get("agent_type", "conversational"),
                })
        
        if missing_agents:
            raise HTTPException(
                status_code=404,
                detail=f"Agent instances not found: {missing_agents}"
            )
        
        # Create group chat
        group_chat_id = str(uuid.uuid4())
        _group_chats[group_chat_id] = {
            "group_chat_id": group_chat_id,
            "name": request.name,
            "agent_instances": agent_details,
            "created_at": datetime.utcnow().isoformat(),
            "messages": [],
        }
        
        logger.info(f"Created group chat: {group_chat_id} with {len(agent_details)} agents")
        
        return {
            "group_chat_id": group_chat_id,
            "name": request.name,
            "agent_count": len(agent_details),
            "agents": agent_details,
            "created_at": _group_chats[group_chat_id]["created_at"],
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create group chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/group-chat/list")
async def list_group_chats() -> List[Dict[str, Any]]:
    """List all group chats"""
    return [
        {
            "groupChatId": chat["group_chat_id"],
            "name": chat["name"],
            "agentCount": len(chat["agent_instances"]),
            "created_at": chat["created_at"],
        }
        for chat in _group_chats.values()
    ]


@router.get("/group-chat/{group_chat_id}")
async def get_group_chat(group_chat_id: str) -> Dict[str, Any]:
    """Get group chat details"""
    if group_chat_id not in _group_chats:
        raise HTTPException(status_code=404, detail="Group chat not found")
    
    chat = _group_chats[group_chat_id]
    return {
        "groupChatId": chat["group_chat_id"],
        "name": chat["name"],
        "agentCount": len(chat["agent_instances"]),
        "agents": chat["agent_instances"],
        "created_at": chat["created_at"],
    }


@router.post("/group-chat/{group_chat_id}/message")
async def send_group_chat_message(
    group_chat_id: str,
    request: GroupChatMessageRequest,
):
    """Send a message to a group chat - all agents respond"""
    if group_chat_id not in _group_chats:
        raise HTTPException(status_code=404, detail="Group chat not found")
    
    group_chat = _group_chats[group_chat_id]
    agent_instances = group_chat["agent_instances"]
    
    if not agent_instances:
        raise HTTPException(status_code=400, detail="Group chat has no agents")
    
    if request.stream:
        # If max_rounds > 0, use conversation loop for multi-round agent-to-agent communication
        if request.max_rounds > 0:
            return StreamingResponse(
                stream_group_chat_conversation_loop(
                    group_chat_id,
                    request.message,
                    max_rounds=request.max_rounds
                ),
                media_type="text/event-stream",
            )
        else:
            # Single round: all agents respond once
            return StreamingResponse(
                stream_group_chat_response(group_chat_id, request.message),
                media_type="text/event-stream",
            )
    else:
        # Non-streaming: collect all responses with agent-to-agent communication
        group_chat = _group_chats[group_chat_id]
        
        # Create agent name mapping
        agent_name_map = {info["instance_id"]: info["name"] for info in agent_instances}
        
        # Save user message
        group_chat["messages"].append({
            "role": "user",
            "content": request.message,
            "sender": "user",
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Build shared context from group chat messages
        shared_context = []
        for msg in group_chat["messages"][-20:]:
            if msg.get("role") == "user":
                shared_context.append({
                    "role": "user",
                    "content": f"[User]: {msg['content']}",
                })
            elif msg.get("role") == "assistant" and "agent_name" in msg:
                agent_name = msg["agent_name"]
                content = msg["content"]
                shared_context.append({
                    "role": "assistant",
                    "content": f"[{agent_name}]: {content}",
                })
        
        responses = []
        for agent_info in agent_instances:
            instance_id = agent_info["instance_id"]
            if instance_id in _active_agents:
                agent_data = _active_agents[instance_id]
                config = AgentConfigRequest(**agent_data["config"])
                
                # Build enhanced system prompt
                enhanced_system_prompt = f"""{config.system_prompt}

You are participating in a group chat with other AI agents. You can see messages from:
{', '.join([name for inst_id, name in agent_name_map.items() if inst_id != instance_id])}

When responding, you can:
- Respond to the user's message
- Respond to or reference what other agents have said
- Engage in conversation with other agents
- Build upon previous messages in the conversation

The conversation history below shows messages from all participants in the group chat."""
                
                try:
                    # Build context string from shared messages
                    context_string = ""
                    if shared_context and len(shared_context) > 0:
                        context_string = "\n\nPrevious messages in this group chat:\n"
                        for ctx_msg in shared_context[-10:]:
                            context_string += f"{ctx_msg['content']}\n"
                        context_string += "\n"
                    
                    # Combine user message with context
                    if context_string:
                        user_input_with_context = f"{context_string}[User]: {request.message}"
                    else:
                        user_input_with_context = request.message
                    
                    # Temporarily update system prompt for this request
                    original_prompt = config.system_prompt
                    config.system_prompt = enhanced_system_prompt
                    
                    response = await generate_agent_response(
                        instance_id,
                        user_input_with_context,
                        config,
                    )
                    
                    # Restore original prompt
                    config.system_prompt = original_prompt
                    
                    response_text = response.get("response", "")
                    
                    # Save to group chat history
                    group_chat["messages"].append({
                        "role": "assistant",
                        "content": response_text,
                        "agent_id": agent_info["agent_id"],
                        "agent_name": agent_info["name"],
                        "instance_id": instance_id,
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                    
                    responses.append({
                        "agent_id": agent_info["agent_id"],
                        "agent_name": agent_info["name"],
                        "response": response_text,
                        "success": response.get("success", False),
                    })
                except Exception as e:
                    logger.error(f"Agent {agent_info['name']} failed: {e}", exc_info=True)
                    responses.append({
                        "agent_id": agent_info["agent_id"],
                        "agent_name": agent_info["name"],
                        "response": f"Error: {str(e)}",
                        "success": False,
                    })
        
        return {
            "group_chat_id": group_chat_id,
            "message": request.message,
            "responses": responses,
        }


async def stream_group_chat_response(group_chat_id: str, user_message: str):
    """Stream responses from all agents in a group chat with agent-to-agent communication"""
    try:
        group_chat = _group_chats[group_chat_id]
        agent_instances = group_chat["agent_instances"]
        
        # Create a mapping of instance_id to agent name for context
        agent_name_map = {info["instance_id"]: info["name"] for info in agent_instances}
        
        # Save user message to group chat history
        group_chat["messages"].append({
            "role": "user",
            "content": user_message,
            "sender": "user",
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Build shared conversation context from group chat messages
        # This allows agents to see what other agents have said
        shared_context = []
        for msg in group_chat["messages"][-20:]:  # Last 20 messages for context
            if msg.get("role") == "user":
                shared_context.append({
                    "role": "user",
                    "content": f"[User]: {msg['content']}",
                })
            elif msg.get("role") == "assistant" and "agent_name" in msg:
                # Format agent messages so other agents know who said what
                agent_name = msg["agent_name"]
                content = msg["content"]
                shared_context.append({
                    "role": "assistant",
                    "content": f"[{agent_name}]: {content}",
                })
        
        # Process each agent sequentially and stream responses
        for agent_info in agent_instances:
            instance_id = agent_info["instance_id"]
            if instance_id not in _active_agents:
                continue
                
            agent_data = _active_agents[instance_id]
            config = AgentConfigRequest(**agent_data["config"])
            
            # Emit agent start event
            yield json.dumps({
                "type": "agent_start",
                "agent_id": agent_info["agent_id"],
                "agent_name": agent_info["name"],
            }) + "\n"
            
            try:
                # Build enhanced system prompt that includes group chat awareness
                enhanced_system_prompt = f"""{config.system_prompt}

You are participating in a group chat with other AI agents. You can see messages from:
{', '.join([name for inst_id, name in agent_name_map.items() if inst_id != instance_id])}

When responding, you can:
- Respond to the user's message
- Respond to or reference what other agents have said
- Engage in conversation with other agents
- Build upon previous messages in the conversation

The conversation history below shows messages from all participants in the group chat."""
                
                # Build messages with shared context
                messages = [ChatMessage(role="system", content=enhanced_system_prompt)]
                
                # Add shared conversation context (messages from all agents)
                for ctx_msg in shared_context:
                    messages.append(ChatMessage(role=ctx_msg["role"], content=ctx_msg["content"]))
                
                # Add the current user message
                messages.append(ChatMessage(role="user", content=f"[User]: {user_message}"))
                
                # Use orchestrator for tool support and better response generation
                orchestrator = get_orchestrator()
                has_tools = bool(config.tools)
                
                if has_tools and (getattr(orchestrator, '_langgraph_agent_available', False) or getattr(orchestrator, 'agent_executor', None) is not None):
                    # Use orchestrator with tools
                    # Build context string from shared messages for orchestrator
                    context_string = ""
                    if shared_context and len(shared_context) > 0:
                        context_string = "\n\nPrevious messages in this group chat:\n"
                        for ctx_msg in shared_context[-10:]:  # Last 10 for context
                            context_string += f"{ctx_msg['content']}\n"
                        context_string += "\n"
                    
                    # Combine user message with context
                    if context_string:
                        user_input_with_context = f"{context_string}[User]: {user_message}"
                    else:
                        user_input_with_context = user_message
                    
                    full_response = ""
                    tool_calls_count = 0
                    
                    result = await orchestrator.process_request(
                        user_input=user_input_with_context,
                        use_tools=has_tools,
                        use_rag=False,
                        max_tokens=config.max_tokens,
                        system_prompt=enhanced_system_prompt,
                        model=config.model,
                        temperature=config.temperature,
                    )
                    
                    response_text = result.get("response", "")
                    intermediate_steps = result.get("intermediate_steps", [])
                    tool_calls_count = len(intermediate_steps)
                    
                    # Stream the response token by token (simulate streaming)
                    words = response_text.split()
                    for i, word in enumerate(words):
                        token = word + (" " if i < len(words) - 1 else "")
                        full_response += token
                        yield json.dumps({
                            "type": "token",
                            "agent_id": agent_info["agent_id"],
                            "agent_name": agent_info["name"],
                            "content": token,
                        }) + "\n"
                    
                    # Emit tool call events if any
                    if tool_calls_count > 0:
                        for step in intermediate_steps:
                            tool_name = "unknown"
                            parameters = {}
                            observation = None
                            
                            if isinstance(step, dict):
                                tool_name = step.get("tool", "unknown")
                                observation = step.get("output", "")
                            elif isinstance(step, tuple) and len(step) >= 2:
                                action = step[0]
                                observation = step[1]
                                if hasattr(action, 'tool'):
                                    tool_name = action.tool
                                if hasattr(action, 'tool_input'):
                                    parameters = action.tool_input if isinstance(action.tool_input, dict) else {}
                            
                            yield json.dumps({
                                "type": "tool_call",
                                "agent_id": agent_info["agent_id"],
                                "agent_name": agent_info["name"],
                                "tool": tool_name,
                                "parameters": parameters,
                            }) + "\n"
                            
                            parsed_result = observation
                            if isinstance(observation, str):
                                try:
                                    parsed_result = json.loads(observation)
                                except (json.JSONDecodeError, TypeError):
                                    parsed_result = observation
                            
                            yield json.dumps({
                                "type": "tool_result",
                                "agent_id": agent_info["agent_id"],
                                "agent_name": agent_info["name"],
                                "tool": tool_name,
                                "parameters": parameters,
                                "result": parsed_result if parsed_result is not None else {},
                            }) + "\n"
                else:
                    # Fallback to direct Ollama call if no tools
                    ollama = await get_ollama_client()
                    options = GenerationOptions(
                        temperature=config.temperature,
                        num_predict=config.max_tokens,
                    )
                    
                    full_response = ""
                    async for token in ollama.chat_stream(messages, model=config.model, options=options):
                        full_response += token
                        yield json.dumps({
                            "type": "token",
                            "agent_id": agent_info["agent_id"],
                            "agent_name": agent_info["name"],
                            "content": token,
                        }) + "\n"
                
                # Save agent's response to group chat history (so other agents can see it)
                group_chat["messages"].append({
                    "role": "assistant",
                    "content": full_response,
                    "agent_id": agent_info["agent_id"],
                    "agent_name": agent_info["name"],
                    "instance_id": instance_id,
                    "timestamp": datetime.utcnow().isoformat(),
                })
                
                # Also save to agent's own conversation history
                conversation = agent_data.get("conversation", [])
                conversation.append({
                    "role": "user",
                    "content": user_message,
                    "timestamp": datetime.utcnow().isoformat(),
                })
                conversation.append({
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": datetime.utcnow().isoformat(),
                })
                
                # Emit agent done event
                yield json.dumps({
                    "type": "agent_done",
                    "agent_id": agent_info["agent_id"],
                    "agent_name": agent_info["name"],
                }) + "\n"
                
            except Exception as e:
                logger.error(f"Error from agent {agent_info['name']}: {e}", exc_info=True)
                yield json.dumps({
                    "type": "agent_error",
                    "agent_id": agent_info["agent_id"],
                    "agent_name": agent_info["name"],
                    "error": str(e),
                }) + "\n"
        
        # All agents done
        yield json.dumps({"type": "done"}) + "\n"
        
    except Exception as e:
        logger.error(f"Group chat streaming error: {e}", exc_info=True)
        yield json.dumps({
            "type": "error",
            "content": f"Failed to process group chat: {str(e)}",
        }) + "\n"


async def stream_group_chat_conversation_loop(
    group_chat_id: str,
    initial_message: str,
    max_rounds: int = 3,
):
    """Allow agents to continue conversing with each other for multiple rounds"""
    try:
        group_chat = _group_chats[group_chat_id]
        agent_instances = group_chat["agent_instances"]
        agent_name_map = {info["instance_id"]: info["name"] for info in agent_instances}
        
        # Save initial user message
        group_chat["messages"].append({
            "role": "user",
            "content": initial_message,
            "sender": "user",
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # First round: all agents respond to user
        yield json.dumps({"type": "round_start", "round": 1, "trigger": "user"}) + "\n"
        
        async for event in stream_group_chat_response(group_chat_id, initial_message):
            yield event
        
        # Subsequent rounds: agents respond to each other
        for round_num in range(2, max_rounds + 1):
            # Get the last messages from agents
            recent_agent_messages = [
                msg for msg in group_chat["messages"][-len(agent_instances):]
                if msg.get("role") == "assistant" and "agent_name" in msg
            ]
            
            if not recent_agent_messages:
                break
            
            # Each agent responds to the most recent messages from other agents
            yield json.dumps({
                "type": "round_start",
                "round": round_num,
                "trigger": "agent_conversation"
            }) + "\n"
            
            # Create a summary of recent agent messages for context
            other_agents_summary = "\n".join([
                f"{msg['agent_name']}: {msg['content'][:200]}"
                for msg in recent_agent_messages[-3:]  # Last 3 agent messages
            ])
            
            # Each agent responds to the conversation
            for agent_info in agent_instances:
                instance_id = agent_info["instance_id"]
                if instance_id not in _active_agents:
                    continue
                
                agent_data = _active_agents[instance_id]
                config = AgentConfigRequest(**agent_data["config"])
                
                # Build context-aware prompt
                conversation_prompt = f"""The other agents in the group chat have been discussing:

{other_agents_summary}

Please respond to the conversation. You can:
- Add your thoughts or opinions
- Ask questions
- Build upon what others have said
- Provide additional information
- Continue the discussion naturally"""
                
                yield json.dumps({
                    "type": "agent_start",
                    "agent_id": agent_info["agent_id"],
                    "agent_name": agent_info["name"],
                    "round": round_num,
                }) + "\n"
                
                try:
                    # Use orchestrator for response
                    orchestrator = get_orchestrator()
                    result = await orchestrator.process_request(
                        user_input=conversation_prompt,
                        use_tools=bool(config.tools),
                        use_rag=False,
                        max_tokens=config.max_tokens,
                        system_prompt=config.system_prompt,
                        model=config.model,
                        temperature=config.temperature,
                    )
                    
                    response_text = result.get("response", "")
                    
                    # Stream response
                    words = response_text.split()
                    for i, word in enumerate(words):
                        token = word + (" " if i < len(words) - 1 else "")
                        yield json.dumps({
                            "type": "token",
                            "agent_id": agent_info["agent_id"],
                            "agent_name": agent_info["name"],
                            "content": token,
                            "round": round_num,
                        }) + "\n"
                    
                    # Save to group chat
                    group_chat["messages"].append({
                        "role": "assistant",
                        "content": response_text,
                        "agent_id": agent_info["agent_id"],
                        "agent_name": agent_info["name"],
                        "instance_id": instance_id,
                        "round": round_num,
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                    
                    yield json.dumps({
                        "type": "agent_done",
                        "agent_id": agent_info["agent_id"],
                        "agent_name": agent_info["name"],
                        "round": round_num,
                    }) + "\n"
                    
                except Exception as e:
                    logger.error(f"Error from agent {agent_info['name']} in round {round_num}: {e}")
                    yield json.dumps({
                        "type": "agent_error",
                        "agent_id": agent_info["agent_id"],
                        "agent_name": agent_info["name"],
                        "round": round_num,
                        "error": str(e),
                    }) + "\n"
            
            yield json.dumps({"type": "round_done", "round": round_num}) + "\n"
        
        yield json.dumps({"type": "conversation_complete", "total_rounds": max_rounds}) + "\n"
        
    except Exception as e:
        logger.error(f"Conversation loop error: {e}", exc_info=True)
        yield json.dumps({
            "type": "error",
            "content": f"Failed to process conversation loop: {str(e)}",
        }) + "\n"


# ============================================================================
# Cache Management Endpoints (for performance optimization)
# ============================================================================

@router.get("/cache/stats")
async def get_cache_stats():
    """
    Get cache statistics for LLM instances and agent executors
    Useful for monitoring cache performance and hit rates
    """
    try:
        orchestrator = get_orchestrator()
        stats = await orchestrator.get_cache_stats()
        return {
            "success": True,
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_caches():
    """
    Clear all caches (LLM instances and agent executors)
    Useful for debugging or freeing memory
    """
    try:
        orchestrator = get_orchestrator()
        result = await orchestrator.clear_caches()
        return {
            "success": True,
            "message": result["message"],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to clear caches: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class ModelWarmRequest(BaseModel):
    """Request to warm (pre-load) models"""
    model_names: List[str] = Field(..., description="List of model names to pre-load")


@router.post("/cache/warm")
async def warm_models(request: ModelWarmRequest):
    """
    Pre-warm (pre-load) models into cache
    This loads models into memory before they're needed, reducing first-request latency
    Call this on startup or during idle time for best results
    """
    try:
        orchestrator = get_orchestrator()
        result = await orchestrator.warm_models(request.model_names)
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to warm models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

