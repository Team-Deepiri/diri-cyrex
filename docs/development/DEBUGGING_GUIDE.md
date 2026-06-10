# AI Engineer Debugging Guide

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Logging System](#logging-system)
3. [Where to Start When You Don't Know](#where-to-start-when-you-dont-know)
4. [Systematic Debugging Methodology](#systematic-debugging-methodology)
5. [Common Issues and Patterns](#common-issues-and-patterns)
6. [Reasoning Through Problems](#reasoning-through-problems)
7. [Quick Reference](#quick-reference)

---

## Architecture Overview

Understanding the execution flow is critical for effective debugging. Here's how requests move through the system:

### Request Flow

```
User Request (Frontend)
    ↓
Agent Playground API (`app/routes/agent_playground_api.py`)
    ├─> stream_agent_response() or generate_agent_response()
    │
    ↓
Workflow Orchestrator (`app/core/orchestrator.py`)
    ├─> process_request()
    │   ├─> Tool Registry (get available tools)
    │   ├─> LangGraph Agent (`app/core/langgraph_agent.py`)
    │   │   ├─> build_agent() - Creates graph with ChatOllama + tools
    │   │   └─> invoke() - Runs the agent graph
    │   │       ├─> LLM generates response (with tool calls if needed)
    │   │       ├─> Tool execution (if tool calls present)
    │   │       └─> LLM processes tool results
    │   └─> Return structured response
    │
    ↓
Response to Frontend
    ├─> Tool calls logged
    ├─> Intermediate steps captured
    └─> Final response text
```

### Key Components

**Orchestrator** (`app/core/orchestrator.py`)
- Entry point for all agent requests
- Manages tool availability
- Routes to LangGraph agent
- Handles caching and model selection

**LangGraph Agent** (`app/core/langgraph_agent.py`)
- Builds the agent graph using `create_react_agent`
- Uses `ChatOllama` for native tool calling
- Executes tools via LangGraph's tool node
- Returns structured results

**Tool Registry** (`app/core/tool_registry.py`)
- Central registry for all tools
- Tools registered per instance
- Categories: DATA, API, MEMORY, etc.

**Agent Playground API** (`app/routes/agent_playground_api.py`)
- FastAPI endpoints for agent interaction
- Registers instance-specific tools (spreadsheet tools)
- Streams responses back to frontend

---

## Logging System

### Logger Names

All loggers follow the pattern: `cyrex.<module>.<submodule>`

Common loggers:
- `cyrex.orchestrator` - Main orchestration logic
- `cyrex.langgraph_agent` - LangGraph agent execution
- `cyrex.routes.agent_playground` - API endpoints
- `cyrex.tool_registry` - Tool registration
- `cyrex.database.*` - Database operations
- `cyrex.integrations.*` - External service integrations

### How to Use Logging

```python
from ..logging_config import get_logger

logger = get_logger("cyrex.your_module")

# Structured logging (recommended)
logger.info("Event description", key1="value1", key2="value2")

# Error logging with stack trace
logger.error("Operation failed", exc_info=True, error_code="E001")

# Debug logging (only in development)
logger.debug("Detailed state", state=state_dict)
```

### Viewing Logs

**Docker Container Logs:**
```bash
# All logs
docker logs deepiri-cyrex-dev

# Follow logs in real-time
docker logs -f deepiri-cyrex-dev

# Filter for errors
docker logs deepiri-cyrex-dev 2>&1 | grep -iE 'ERROR|exception|Traceback'

# Filter for specific logger
docker logs deepiri-cyrex-dev 2>&1 | grep 'cyrex.orchestrator'

# Last 50 lines
docker logs --tail 50 deepiri-cyrex-dev
```

**Structured JSON Logs:**
Logs are in JSON format, making them easy to parse:
```bash
# Extract specific fields
docker logs deepiri-cyrex-dev 2>&1 | jq 'select(.logger == "cyrex.orchestrator")'

# Find errors with context
docker logs deepiri-cyrex-dev 2>&1 | jq 'select(.level == "error")'
```

### Log Levels

- **DEBUG**: Detailed diagnostic information (development only)
- **INFO**: General informational messages (normal operation)
- **WARNING**: Warning messages (non-critical issues)
- **ERROR**: Error messages (operation failures)
- **CRITICAL**: Critical errors (system failures)

---

## Where to Start When You Don't Know

### Step 1: Check the Error Message

If there's an explicit error, start there:

```bash
docker logs deepiri-cyrex-dev 2>&1 | grep -iE 'ERROR|exception|Traceback' | tail -20
```

Look for:
- Python exceptions (TypeError, ValueError, RuntimeError, etc.)
- Import errors
- Connection errors
- Timeout errors

### Step 2: Trace the Request Flow

Follow the request from entry point to failure:

1. **Check API endpoint logs:**
   ```bash
   docker logs deepiri-cyrex-dev 2>&1 | grep 'cyrex.routes.agent_playground'
   ```

2. **Check orchestrator logs:**
   ```bash
   docker logs deepiri-cyrex-dev 2>&1 | grep 'cyrex.orchestrator'
   ```

3. **Check agent execution logs:**
   ```bash
   docker logs deepiri-cyrex-dev 2>&1 | grep 'cyrex.langgraph_agent'
   ```

4. **Check tool execution logs:**
   ```bash
   docker logs deepiri-cyrex-dev 2>&1 | grep -iE 'tool|spreadsheet'
   ```

### Step 3: Identify the Failure Point

Based on logs, determine where it fails:

- **Before orchestrator**: API routing issue, authentication, validation
- **In orchestrator**: Tool registration, model initialization, graph building
- **In agent execution**: LLM call, tool calling, response parsing
- **In tool execution**: Tool function error, database error, external API error

### Step 4: Check Component Health

Verify each component is working:

```bash
# Check if container is running
docker ps | grep cyrex

# Check container health
docker inspect deepiri-cyrex-dev | grep -A 5 Health

# Check database connectivity (if applicable)
docker exec deepiri-cyrex-dev python -c "from app.database.postgres import get_postgres_manager; import asyncio; asyncio.run(get_postgres_manager().test_connection())"

# Check Ollama connectivity
curl http://localhost:11434/api/tags
```

---

## Systematic Debugging Methodology

### 1. Reproduce the Issue

Create a minimal test case that reproduces the problem:

```python
# test_debug.py
import requests
import json

# Initialize agent
response = requests.post("http://localhost:8000/api/agent/initialize", json={
    "agentId": "test",
    "model": "mistral-nemo:12b",
    "tools": ["spreadsheet_set_cell"]
})
instance_id = response.json()["instance"]["instanceId"]

# Reproduce issue
response = requests.post("http://localhost:8000/api/agent/invoke", json={
    "instanceId": instance_id,
    "message": "Set cell A1 to 42"
})
print(json.dumps(response.json(), indent=2))
```

### 2. Add Strategic Logging

Add logging at key decision points:

```python
# In orchestrator.py
logger.info("Processing request", 
    user_input=user_input[:100], 
    model=model, 
    tool_count=len(tools),
    has_langgraph=_langgraph_agent_available
)

# In langgraph_agent.py
logger.info("Invoking agent",
    model=model_name,
    tool_names=[t.name for t in tools],
    user_input_len=len(user_input)
)

# In tool execution
logger.info("Tool called",
    tool_name=tool_name,
    args=args,
    instance_id=instance_id
)
```

### 3. Isolate the Problem

Narrow down the issue by testing components individually:

**Test tool registration:**
```python
from app.core.tool_registry import get_tool_registry
registry = get_tool_registry()
tools = registry.get_tools()
print([t.name for t in tools])
```

**Test LLM directly:**
```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="mistral-nemo:12b")
response = llm.invoke("Hello")
print(response.content)
```

**Test tool execution:**
```python
# Directly call the tool function
from app.routes.agent_playground_api import _register_spreadsheet_tools_for_instance
# ... test tool in isolation
```

### 4. Check Dependencies

Verify all dependencies are available:

```python
# Check imports
try:
    from langgraph.prebuilt import create_react_agent
    print("LangGraph: OK")
except ImportError as e:
    print(f"LangGraph: FAILED - {e}")

try:
    from langchain_ollama import ChatOllama
    print("ChatOllama: OK")
except ImportError as e:
    print(f"ChatOllama: FAILED - {e}")
```

### 5. Verify Configuration

Check environment variables and settings:

```bash
# Inside container
docker exec deepiri-cyrex-dev env | grep -E 'OLLAMA|DATABASE|REDIS'

# Check settings
docker exec deepiri-cyrex-dev python -c "from app.settings import settings; print(settings.dict())"
```

### 6. Test with Minimal Setup

Strip down to the bare minimum:

1. Remove all tools except one
2. Use simplest system prompt
3. Disable caching
4. Disable guardrails
5. Use default model settings

If it works with minimal setup, add components back one by one.

---

## Common Issues and Patterns

### Issue: Tools Not Being Called

**Symptoms:**
- Agent responds but doesn't execute tools
- `toolCalls: 0` in metrics
- No tool execution logs

**Debugging Steps:**

1. **Check if tools are registered:**
   ```bash
   docker logs deepiri-cyrex-dev 2>&1 | grep 'Registered tool'
   ```

2. **Check if LangGraph agent is available:**
   ```bash
   docker logs deepiri-cyrex-dev 2>&1 | grep 'LangGraph.*available'
   ```

3. **Check tool definitions:**
   - Tools must use `StructuredTool.from_function()` for native tool calling
   - Tool functions must have proper type hints
   - Tool descriptions must be clear

4. **Check model capabilities:**
   - Model must support tool calling (ChatOllama, not OllamaLLM)
   - Model must be loaded in Ollama: `ollama list`

5. **Check agent graph:**
   ```python
   # In orchestrator, log the graph structure
   logger.info("Agent graph built", 
       tools=[t.name for t in tools],
       model=model_name
   )
   ```

**Common Causes:**
- Using `Tool()` instead of `StructuredTool.from_function()`
- Model doesn't support tool calling
- Tools not passed to agent builder
- LangGraph not available or misconfigured

### Issue: Tool Execution Errors

**Symptoms:**
- Tool is called but fails
- `TypeError`, `RuntimeError`, or other exceptions in tool execution
- Tool returns error response

**Debugging Steps:**

1. **Check tool function signature:**
   ```python
   # Tool must accept structured arguments
   def set_cell_sync(cell_id: str, value: str) -> str:
       # Not: def set_cell_sync(input: str) -> str
   ```

2. **Check async/sync handling:**
   - If tool is sync but calls async code, use `asyncio.new_event_loop()`
   - Don't use `asyncio.get_event_loop()` in ThreadPoolExecutor threads

3. **Check tool return format:**
   - Tools should return strings (JSON strings for structured data)
   - Errors should be caught and returned as error responses

4. **Check database/external service connectivity:**
   ```bash
   # Test database
   docker exec deepiri-cyrex-dev python -c "from app.database.postgres import get_postgres_manager; import asyncio; asyncio.run(get_postgres_manager().test_connection())"
   ```

**Common Causes:**
- Event loop issues in sync wrappers
- Missing required arguments
- Database connection failures
- External API timeouts

### Issue: Slow Response Times

**Symptoms:**
- Requests take >30 seconds
- Timeout errors
- High latency in metrics

**Debugging Steps:**

1. **Check where time is spent:**
   ```python
   # Add timing logs
   start = time.time()
   # ... operation ...
   logger.info("Operation completed", duration_ms=(time.time() - start) * 1000)
   ```

2. **Check model loading:**
   ```bash
   # Check if model is pre-loaded
   curl http://localhost:11434/api/ps
   ```

3. **Check database queries:**
   ```bash
   docker logs deepiri-cyrex-dev 2>&1 | grep -iE 'database|query|postgres' | tail -20
   ```

4. **Check Redis/Milvus connectivity:**
   ```bash
   docker logs deepiri-cyrex-dev 2>&1 | grep -iE 'redis|milvus|timeout'
   ```

5. **Check agent iteration count:**
   - High iteration count = agent stuck in loop
   - Check `max_iterations` setting

**Common Causes:**
- Model not pre-loaded (cold start)
- Database query performance
- Agent hitting iteration limit
- Network latency to external services
- Large context windows

### Issue: Import Errors

**Symptoms:**
- `ModuleNotFoundError` or `ImportError`
- Component not available warnings

**Debugging Steps:**

1. **Check if package is installed:**
   ```bash
   docker exec deepiri-cyrex-dev pip list | grep langgraph
   ```

2. **Check requirements.txt:**
   - Ensure package is in appropriate requirements file
   - Check version compatibility

3. **Check import fallbacks:**
   ```python
   # Code should have graceful fallbacks
   try:
       from langgraph.prebuilt import create_react_agent
       HAS_LANGGRAPH = True
   except ImportError:
       HAS_LANGGRAPH = False
       logger.warning("LangGraph not available")
   ```

4. **Rebuild container:**
   ```bash
   docker compose -f docker-compose.dev.yml build cyrex
   docker compose -f docker-compose.dev.yml up -d cyrex
   ```

**Common Causes:**
- Package not in requirements.txt
- Version mismatch
- Missing dependency
- Container not rebuilt after adding dependency

### Issue: Agent Stuck in Loop

**Symptoms:**
- Agent hits iteration limit
- Same tool called repeatedly
- No final response

**Debugging Steps:**

1. **Check intermediate steps:**
   ```bash
   docker logs deepiri-cyrex-dev 2>&1 | grep 'intermediate_steps'
   ```

2. **Check tool responses:**
   - Tool may be returning error that agent doesn't understand
   - Tool response format may be confusing the agent

3. **Check system prompt:**
   - Prompt may be too vague
   - Missing instructions on when to stop

4. **Check max_iterations:**
   ```python
   # In orchestrator or agent config
   max_iterations = 15  # Default, may need adjustment
   ```

**Common Causes:**
- Tool returning unclear error messages
- System prompt too vague
- Agent can't determine task completion
- Tool responses not in expected format

---

## Reasoning Through Problems

### The Scientific Method

1. **Observe**: What exactly is happening? What are the symptoms?
2. **Hypothesize**: What could cause this? List all possibilities.
3. **Test**: Create a test to verify each hypothesis.
4. **Analyze**: Which hypothesis is correct?
5. **Fix**: Implement the solution.
6. **Verify**: Confirm the fix works and doesn't break anything else.

### Example: Tool Not Being Called

**Observe:**
- Agent responds with text but doesn't call tools
- Metrics show `toolCalls: 0`
- No tool execution logs

**Hypothesize:**
1. Tools not registered
2. Tools not passed to agent
3. Model doesn't support tool calling
4. LangGraph not configured correctly
5. Tool definitions incorrect

**Test:**
```python
# Test 1: Are tools registered?
registry = get_tool_registry()
tools = registry.get_tools()
assert len(tools) > 0

# Test 2: Are tools passed to agent?
# Check orchestrator logs for tool count

# Test 3: Does model support tool calling?
# Check if ChatOllama is used (not OllamaLLM)

# Test 4: Is LangGraph available?
# Check logs for "LangGraph prebuilt agent available"

# Test 5: Are tool definitions correct?
# Check if StructuredTool.from_function() is used
```

**Analyze:**
- If Test 1 fails: Tool registration issue
- If Test 2 fails: Orchestrator not passing tools
- If Test 3 fails: Wrong LLM class
- If Test 4 fails: LangGraph not installed
- If Test 5 fails: Tool definition issue

**Fix:**
Based on which test fails, fix the root cause.

**Verify:**
Run the test case again and confirm tools are called.

### The Divide and Conquer Approach

Break the problem into smaller pieces:

1. **Isolate the component**: Does it fail in isolation?
2. **Test inputs**: What inputs cause the failure?
3. **Test outputs**: What outputs are expected vs. actual?
4. **Test boundaries**: What are the edge cases?

### Example: Tool Execution Error

**Isolate:**
```python
# Test tool function directly
def test_set_cell():
    result = set_cell_sync("A1", "42")
    assert "success" in result
```

**Test inputs:**
- Valid cell ID: "A1" ✓
- Invalid cell ID: "ZZ999" ?
- Empty value: "" ?
- Formula: "=SUM(A1:A10)" ?

**Test outputs:**
- Expected: `{"success": true, "cell_id": "A1", "value": "42"}`
- Actual: `RuntimeError: There is no current event loop`

**Test boundaries:**
- What if called from async context?
- What if called from sync context?
- What if called from thread pool?

**Root cause identified:**
Tool wrapper uses `asyncio.get_event_loop()` which fails in ThreadPoolExecutor threads.

### The Bottom-Up Approach

Start from the lowest level and work up:

1. **Hardware/Infrastructure**: Is the container running? Is the database up?
2. **Dependencies**: Are all packages installed? Are versions compatible?
3. **Configuration**: Are environment variables set? Are settings correct?
4. **Code Logic**: Does the code execute? Are there syntax errors?
5. **Business Logic**: Does the logic produce correct results?

### Example: Slow Response Times

**Level 1 - Infrastructure:**
```bash
docker ps | grep cyrex  # Container running?
curl http://localhost:11434/api/tags  # Ollama accessible?
```

**Level 2 - Dependencies:**
```bash
docker exec deepiri-cyrex-dev pip list | grep langgraph  # Installed?
```

**Level 3 - Configuration:**
```bash
docker exec deepiri-cyrex-dev env | grep OLLAMA  # Config correct?
```

**Level 4 - Code Logic:**
```python
# Add timing logs to identify slow operations
start = time.time()
result = await agent.invoke(...)
logger.info("Agent invocation", duration_ms=(time.time() - start) * 1000)
```

**Level 5 - Business Logic:**
- Is the agent doing unnecessary work?
- Can we cache results?
- Can we parallelize operations?

---

## Quick Reference

### Essential Commands

```bash
# View logs
docker logs -f deepiri-cyrex-dev

# Filter logs
docker logs deepiri-cyrex-dev 2>&1 | grep 'cyrex.orchestrator'

# Restart container
docker compose -f docker-compose.dev.yml restart cyrex

# Rebuild container
docker compose -f docker-compose.dev.yml build cyrex
docker compose -f docker-compose.dev.yml up -d cyrex

# Check container status
docker ps | grep cyrex

# Execute Python in container
docker exec -it deepiri-cyrex-dev python

# Check Ollama models
curl http://localhost:11434/api/tags
```

### Key Files to Check

- `app/core/orchestrator.py` - Main orchestration logic
- `app/core/langgraph_agent.py` - LangGraph agent implementation
- `app/routes/agent_playground_api.py` - API endpoints and tool registration
- `app/core/tool_registry.py` - Tool registry
- `app/logging_config.py` - Logging configuration
- `requirements.txt` - Python dependencies

### Common Log Patterns

```bash
# Find errors
docker logs deepiri-cyrex-dev 2>&1 | grep -iE 'ERROR|exception|Traceback'

# Find tool calls
docker logs deepiri-cyrex-dev 2>&1 | grep -iE 'tool.*call|Registered tool'

# Find agent execution
docker logs deepiri-cyrex-dev 2>&1 | grep -iE 'agent.*invoke|LangGraph'

# Find database operations
docker logs deepiri-cyrex-dev 2>&1 | grep -iE 'database|postgres|query'

# Find timing issues
docker logs deepiri-cyrex-dev 2>&1 | grep -iE 'duration|timeout|slow'
```

### Debug Checklist

When debugging a new issue:

- [ ] Check error logs for exceptions
- [ ] Trace request flow through components
- [ ] Verify all dependencies are installed
- [ ] Check configuration and environment variables
- [ ] Test components in isolation
- [ ] Add strategic logging at key points
- [ ] Reproduce with minimal setup
- [ ] Verify fix doesn't break other functionality

---

## Conclusion

Effective debugging requires:

1. **Understanding the architecture** - Know how components interact
2. **Strategic logging** - Log at key decision points
3. **Systematic approach** - Follow a methodical process
4. **Isolation** - Test components individually
5. **Reasoning** - Use scientific method and divide-and-conquer

Remember: Most bugs are not mysterious. They follow patterns. Once you understand the system and have good logging, you can solve almost any issue.

