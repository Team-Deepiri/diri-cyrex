"""
LangGraph Tool-Calling Agent with PDGE Integration

Uses native tool calling via ChatOllama/ChatOpenAI instead of ReAct text parsing.
Tool execution is handled by the Parallel Dependency Graph Execution (PDGE)
engine, which:
  - Executes independent tool calls simultaneously
  - Serializes writes to the same resource
  - Caches read-only tool results semantically
  - Detects GPU availability for compute-heavy tools

Why this works:
- ChatOllama uses /api/chat (not /api/generate) which supports tool definitions
- The model outputs structured tool_calls, not text to parse
- PDGE replaces LangGraph's default sequential ToolNode
- Zero prompt template overhead, zero output parsing, zero agent_scratchpad
"""
from typing import Optional, Dict, Any, List
import os
import time
import asyncio
from ..logging_config import get_logger

logger = get_logger("cyrex.langgraph_agent")

# ---------------------------------------------------------------------------
# LangGraph imports
# ---------------------------------------------------------------------------
HAS_LANGGRAPH = False
StateGraph = None
MessagesState = None
try:
    from langgraph.graph import StateGraph, MessagesState
    HAS_LANGGRAPH = True
    logger.info("LangGraph StateGraph available")
except ImportError as e:
    logger.warning(f"LangGraph not available: {e}")

# Prebuilt fallback (used only if custom graph fails)
HAS_LANGGRAPH_PREBUILT = False
create_react_agent = None
try:
    from langgraph.prebuilt import create_react_agent
    HAS_LANGGRAPH_PREBUILT = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# LLM imports
# ---------------------------------------------------------------------------
HAS_CHAT_OLLAMA = False
ChatOllama = None
try:
    from langchain_ollama import ChatOllama
    HAS_CHAT_OLLAMA = True
    logger.info("ChatOllama available for native tool calling")
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama
        HAS_CHAT_OLLAMA = True
        logger.info("ChatOllama available via langchain-community")
    except ImportError as e:
        logger.warning(f"ChatOllama not available: {e}")

HAS_CHAT_OPENAI = False
ChatOpenAI = None
try:
    from langchain_openai import ChatOpenAI
    HAS_CHAT_OPENAI = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# LangChain message types
# ---------------------------------------------------------------------------
try:
    from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
except ImportError:
    AIMessage = ToolMessage = SystemMessage = HumanMessage = None

# ---------------------------------------------------------------------------
# PDGE + Streaming Bridge imports
# ---------------------------------------------------------------------------
from .parallel_tool_executor import PDGEngine
from .streaming_tool_bridge import get_bridge


def is_available() -> bool:
    """Check if LangGraph agent can be created"""
    return (HAS_LANGGRAPH or HAS_LANGGRAPH_PREBUILT) and (HAS_CHAT_OLLAMA or HAS_CHAT_OPENAI)


def create_chat_model(
    model_name: str,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: float = 120.0,
    num_ctx: Optional[int] = None,
    keep_alive: str = "30m",
):
    """
    Create a chat model that supports native tool calling.
    Returns ChatOllama or ChatOpenAI depending on what's available.
    """
    if base_url is None:
        base_url = os.getenv("OLLAMA_BASE_URL")
        if base_url is None:
            is_docker = os.path.exists("/.dockerenv") or os.path.exists("/proc/1/cgroup")
            base_url = "http://ollama:11434" if is_docker else "http://localhost:11434"

    if HAS_CHAT_OLLAMA:
        logger.info(
            f"Creating ChatOllama: model={model_name}, base_url={base_url}, "
            f"num_ctx={num_ctx}, keep_alive={keep_alive}, max_tokens={max_tokens}"
        )
        params = {
            "model": model_name,
            "base_url": base_url,
            "temperature": temperature,
            "num_predict": max_tokens,  # Ollama parameter name for max_tokens
            "timeout": timeout,
            "keep_alive": keep_alive,  # Keep model loaded in memory
        }
        if num_ctx is not None:
            params["num_ctx"] = num_ctx
        # Ensure keep_alive is always set to prevent model unloading
        if not params.get("keep_alive"):
            params["keep_alive"] = "30m"
        return ChatOllama(**params)

    elif HAS_CHAT_OPENAI:
        logger.info(f"Creating ChatOpenAI: model={model_name}")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise RuntimeError("No chat model backend available. Install langchain-ollama or langchain-openai.")


# ---------------------------------------------------------------------------
# Build agent graph with PDGE tool node
# ---------------------------------------------------------------------------

def build_agent(
    model_name: str,
    tools: List[Any],
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: float = 120.0,
    num_ctx: Optional[int] = None,
    keep_alive: str = "30m",
):
    """
    Build a LangGraph agent with PDGE-powered parallel tool execution.

    Instead of using LangGraph's default ToolNode (sequential execution),
    this builds a custom graph:

        START -> agent_node -> should_continue? -> pdge_tool_node -> agent_node -> ... -> END

    The pdge_tool_node uses PDGEngine to:
    - Execute independent tools in parallel
    - Serialize writes to shared resources
    - Cache read-only results
    - Detect GPU for compute tools

    Args:
        model_name: Ollama model name (e.g. "mistral-nemo:12b")
        tools: List of LangChain tools from the tool registry
        base_url: Ollama base URL (auto-detected if not provided)
        temperature: LLM temperature
        max_tokens: Max tokens to generate
        timeout: Request timeout in seconds
        num_ctx: Context window size (None = use model default)
        keep_alive: How long to keep model in memory (e.g. "30m")

    Returns:
        Tuple of (compiled_graph, pdge_engine)
    """
    if not (HAS_LANGGRAPH or HAS_LANGGRAPH_PREBUILT):
        raise RuntimeError("LangGraph not available. Install langgraph>=0.2.0")

    if not tools:
        raise ValueError("At least one tool is required to build a tool-calling agent")

    # Optimize tool descriptions (trim verbose ones)
    from .tool_schema_optimizer import optimize_tool_list
    optimized_tools = optimize_tool_list(tools)

    llm = create_chat_model(
        model_name=model_name,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        num_ctx=num_ctx,
        keep_alive=keep_alive,
    )

    # Bind tools to LLM (this tells Ollama the tool schemas)
    llm_with_tools = llm.bind_tools(optimized_tools)

    # Create PDGE engine for parallel tool execution
    pdge = PDGEngine(optimized_tools)

    # Build custom graph with PDGE tool node
    if HAS_LANGGRAPH and StateGraph and MessagesState:
        graph = _build_pdge_graph(llm_with_tools, optimized_tools, pdge)
    elif HAS_LANGGRAPH_PREBUILT:
        # Fallback: use prebuilt (sequential tools) if custom graph fails
        logger.warning("Using prebuilt create_react_agent (no PDGE parallel execution)")
        graph = create_react_agent(llm, optimized_tools)
        # Wrap in tuple for consistent interface
        tool_names = [t.name for t in tools]
        logger.info(f"Built LangGraph agent (prebuilt): model={model_name}, tools={tool_names}, num_ctx={num_ctx}")
        return graph, pdge
    else:
        raise RuntimeError("LangGraph not available")

    tool_names = [t.name for t in tools]
    logger.info(f"Built PDGE LangGraph agent: model={model_name}, tools={tool_names}, num_ctx={num_ctx}")

    return graph, pdge


def _build_pdge_graph(llm_with_tools: Any, tools: List[Any], pdge: PDGEngine):
    """
    Build a custom StateGraph that replaces the default ToolNode
    with PDGE parallel execution.
    """
    # Tool lookup by name
    tool_map = {getattr(t, "name", str(t)): t for t in tools}

    # --- Agent node: call LLM with tool bindings ---
    async def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        node_start = time.time()
        logger.debug(f"Agent node: invoking LLM with {len(messages)} messages")
        try:
        response = await llm_with_tools.ainvoke(messages)
            node_duration = (time.time() - node_start) * 1000
            logger.info(f"Agent node LLM call completed in {node_duration:.0f}ms")
        return {"messages": [response]}
        except Exception as e:
            node_duration = (time.time() - node_start) * 1000
            logger.error(f"Agent node LLM call failed after {node_duration:.0f}ms: {e}")
            raise

    # --- PDGE tool node: parallel execution ---
    async def pdge_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}

        last_msg = messages[-1]

        # Extract tool_calls from last AIMessage
        if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
            return {"messages": []}

        raw_calls = last_msg.tool_calls
        t0 = time.time()

        # Execute through PDGE
        results = await pdge.execute(raw_calls)

        # Convert results to ToolMessages
        tool_messages = []
        for tc, result in zip(raw_calls, results):
            tool_messages.append(ToolMessage(
                content=result.output,
                tool_call_id=tc.get("id", ""),
                name=tc.get("name", ""),
            ))

        total_ms = (time.time() - t0) * 1000
        logger.info(
            f"PDGE tool node: {len(raw_calls)} calls in {total_ms:.0f}ms "
            f"(cached: {sum(1 for r in results if r.from_cache)})"
        )

        return {"messages": tool_messages}

    # --- Routing function ---
    def should_continue(state: Dict[str, Any]) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "__end__"
        last_msg = messages[-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "__end__"

    # --- Build graph ---
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", pdge_tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": "__end__"})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ---------------------------------------------------------------------------
# Invoke
# ---------------------------------------------------------------------------

async def invoke(
    agent_and_pdge,
    user_input: str,
    system_prompt: Optional[str] = None,
    timeout: float = 120.0,
    stream: bool = False,
):
    """
    Invoke the LangGraph agent and return a structured result.

    Args:
        agent_and_pdge: Tuple of (compiled_graph, pdge_engine) from build_agent,
                        OR just a compiled graph (prebuilt fallback)
        user_input: The user's message
        system_prompt: Optional system prompt (prepended as system message)
        timeout: Max execution time
        stream: If True, return async iterator of StreamChunks (revolutionary <200ms first-token)

    Returns:
        If stream=False: Dict with 'response', 'tool_calls', 'intermediate_steps', etc.
        If stream=True: AsyncIterator[StreamChunk] (use as: async for chunk in invoke(...))
    """
    # Route to streaming or non-streaming implementation
    if stream:
        # Return the generator directly (caller uses: async for chunk in invoke(...))
        return _invoke_streaming(agent_and_pdge, user_input, system_prompt, timeout)
    
    # Non-streaming path (returns dict)
    return await _invoke_non_streaming(agent_and_pdge, user_input, system_prompt, timeout)


async def _invoke_streaming(
    agent_and_pdge,
    user_input: str,
    system_prompt: Optional[str],
    timeout: float,
):
    """Streaming implementation (async generator)."""
    start = time.time()
    
    # Unpack agent
    if isinstance(agent_and_pdge, tuple):
        agent, pdge = agent_and_pdge
    else:
        agent = agent_and_pdge
        pdge = None
    
    if not pdge:
        # Fallback: can't stream without PDGE
        from .streaming_coordinator import StreamChunk
        yield StreamChunk(
            type="error",
            content="Streaming requires PDGE engine",
            timestamp_ms=0.0,
            metadata={"error": "pdge_missing"},
        )
        return
    
    # Pre-warm resources
    bridge = get_bridge()
    pre_warm_task = asyncio.create_task(bridge.pre_warm_for_input(user_input))
    
    # Build message list
    messages = []
    if system_prompt:
        messages.append(("system", system_prompt))
    messages.append(("user", user_input))
    
    # Revolutionary streaming path
    from .streaming_coordinator import StreamingPDGECoordinator
    coordinator = StreamingPDGECoordinator(pdge)
    
    try:
        # Stream from LangGraph
        llm_stream = agent.astream({"messages": messages})
        
        # Coordinate streaming + parallel tool execution
        async for chunk in coordinator.coordinate_stream(llm_stream):
            yield chunk
        
        # Log final metrics
        metrics = coordinator.metrics
        logger.info(
            f"Streaming completed: first_token={metrics['first_token_ms']:.0f}ms, "
            f"total={metrics['total_time_ms']:.0f}ms, tools={metrics['tools_detected']}"
        )
        
    except asyncio.TimeoutError:
        latency_ms = (time.time() - start) * 1000
        logger.error(f"Streaming timed out after {timeout}s")
        from .streaming_coordinator import StreamChunk
        yield StreamChunk(
            type="error",
            content="Agent timed out",
            timestamp_ms=latency_ms,
            metadata={"timed_out": True},
        )
    finally:
        await bridge.cleanup()


async def _invoke_non_streaming(
    agent_and_pdge,
    user_input: str,
    system_prompt: Optional[str],
    timeout: float,
) -> Dict[str, Any]:
    """Non-streaming implementation (returns dict)."""
    start = time.time()

    # Unpack agent
    if isinstance(agent_and_pdge, tuple):
        agent, pdge = agent_and_pdge
    else:
        agent = agent_and_pdge
        pdge = None

    # Pre-warm resources in parallel with LLM inference (zero added latency)
    bridge = get_bridge()
    pre_warm_task = asyncio.create_task(bridge.pre_warm_for_input(user_input))

    # Build message list
    messages = []
    if system_prompt:
        messages.append(("system", system_prompt))
    messages.append(("user", user_input))

    # Standard non-streaming path with detailed timing
    llm_start = time.time()
    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": messages}),
            timeout=timeout,
        )
        llm_duration = (time.time() - llm_start) * 1000
        logger.info(f"LLM inference took {llm_duration:.0f}ms")
    except asyncio.TimeoutError:
        latency_ms = (time.time() - start) * 1000
        logger.error(f"LangGraph agent timed out after {timeout}s")
        await bridge.cleanup()
        return {
            "response": None,
            "tool_calls": [],
            "intermediate_steps": [],
            "latency_ms": latency_ms,
            "timed_out": True,
            "pdge_stats": pdge.stats if pdge else {},
        }

    # Cleanup pre-warm tasks
    await bridge.cleanup()

    latency_ms = (time.time() - start) * 1000
    out_messages = result.get("messages", [])

    # Extract response, tool calls, intermediate steps
    response_text = ""
    tool_calls = []
    intermediate_steps = []

    for msg in out_messages:
        msg_type = type(msg).__name__

        if msg_type == "AIMessage":
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "tool": tc.get("name", "unknown"),
                        "input": tc.get("args", {}),
                    })
            else:
                if hasattr(msg, "content") and msg.content:
                    response_text = msg.content

        elif msg_type == "ToolMessage":
            if hasattr(msg, "content"):
                intermediate_steps.append({
                    "tool": getattr(msg, "name", "unknown"),
                    "output": msg.content,
                })

    # Get PDGE stats safely
    pdge_executions = 0
    if pdge and hasattr(pdge, 'stats'):
        try:
            pdge_stats = pdge.stats
            if isinstance(pdge_stats, dict):
                pdge_executions = pdge_stats.get('total_executions', 0)
        except Exception:
            pass

    logger.info(
        f"LangGraph agent completed: {latency_ms:.0f}ms, "
        f"tool_calls={len(tool_calls)}, "
        f"response_len={len(response_text)}, "
        f"pdge_executions={pdge_executions}"
    )

    return {
        "response": response_text,
        "tool_calls": tool_calls,
        "intermediate_steps": intermediate_steps,
        "latency_ms": latency_ms,
        "timed_out": False,
        "pdge_stats": pdge.stats if pdge else {},
    }
