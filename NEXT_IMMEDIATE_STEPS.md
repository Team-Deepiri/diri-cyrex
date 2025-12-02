# Next Immediate Steps - After Critical Fixes

## ✅ What's Done
- Agent executor implementation complete
- Tools integration in process_request() complete
- RAG chain formatting fixed
- All code changes committed

---

## 🚀 Immediate Next Steps (In Order)

### Step 1: Verify Dependencies (5 minutes)

**Check if LangChain agents are installed:**

```bash
cd deepiri/diri-cyrex
pip list | grep langchain
```

**Required packages:**
- `langchain>=0.1.0`
- `langchain-core>=0.1.23`
- `langchain-community>=0.0.20`
- `langchain-openai>=0.0.5` (for OpenAI agents)

**If missing, install:**
```bash
pip install langchain>=0.1.0 langchain-core>=0.1.23 langchain-community>=0.0.20 langchain-openai>=0.0.5
```

**Note:** `langchain-hub` is optional - we have a fallback prompt.

---

### Step 2: Test Orchestrator Initialization (5 minutes)

**Create a quick test script:**

```python
# test_orchestrator_init.py
import asyncio
from app.core.orchestrator import get_orchestrator
from app.core.tool_registry import get_tool_registry

async def test_init():
    print("Testing orchestrator initialization...")
    
    try:
        orchestrator = get_orchestrator()
        print(f"✅ Orchestrator created")
        
        # Check agent executor
        if orchestrator.agent_executor:
            print(f"✅ Agent executor initialized")
        else:
            print(f"⚠️  Agent executor not initialized (may be normal if no tools)")
        
        # Check tool registry
        registry = get_tool_registry()
        tools = registry.get_tools()
        print(f"✅ Tool registry has {len(tools)} tools")
        
        # Check LLM provider
        if orchestrator.llm_provider:
            health = orchestrator.llm_provider.health_check()
            print(f"✅ LLM provider: {health}")
        else:
            print(f"⚠️  LLM provider not initialized")
        
        # Get status
        status = await orchestrator.get_status()
        print(f"✅ Status check passed")
        print(f"   Tools: {status.get('tools', {})}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_init())
```

**Run it:**
```bash
cd deepiri/diri-cyrex
python test_orchestrator_init.py
```

**Expected output:**
- ✅ Orchestrator created
- ✅ Agent executor initialized (if tools exist)
- ✅ Tool registry working
- ✅ LLM provider health check

---

### Step 3: Create and Test a Sample Tool (10 minutes)

**Create a test tool file:**

```python
# test_tool_integration.py
import asyncio
from app.core.orchestrator import get_orchestrator
from app.core.tool_registry import get_tool_registry
from langchain_core.tools import Tool

# Simple test tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Input should be the city name."""
    return f"The weather in {city} is sunny and 72°F with light winds."

async def test_tool_execution():
    print("Testing tool execution...")
    
    # Register tool
    registry = get_tool_registry()
    weather_tool = Tool(
        name="get_weather",
        description="Get the current weather for a city. Input should be the city name.",
        func=get_weather
    )
    registry.register_tool(weather_tool)
    print(f"✅ Registered tool: {weather_tool.name}")
    
    # Get orchestrator (will reinitialize with new tool)
    orchestrator = get_orchestrator()
    
    # Force re-setup chains to pick up new tool
    orchestrator._setup_chains()
    
    if not orchestrator.agent_executor:
        print("❌ Agent executor not created. Check logs for errors.")
        return
    
    print(f"✅ Agent executor ready")
    
    # Test request with tools
    print("\n📤 Making request with tools...")
    result = await orchestrator.process_request(
        "What's the weather in San Francisco?",
        use_tools=True
    )
    
    print(f"\n📥 Response:")
    print(f"   Success: {result.get('success')}")
    print(f"   Response: {result.get('response', '')[:200]}...")
    print(f"   Duration: {result.get('duration_ms', 0):.2f}ms")
    
    if result.get('success'):
        print("\n✅ Tool execution test PASSED")
    else:
        print(f"\n❌ Tool execution test FAILED: {result.get('error')}")

if __name__ == "__main__":
    asyncio.run(test_tool_execution())
```

**Run it:**
```bash
python test_tool_integration.py
```

**Expected behavior:**
- Tool is registered
- Agent executor is created
- Request triggers tool usage
- Response includes tool result

---

### Step 4: Test with OpenAI (If Available) (5 minutes)

**If you have OPENAI_API_KEY set:**

```python
# test_openai_agent.py
import asyncio
from app.core.orchestrator import get_orchestrator
from app.core.tool_registry import get_tool_registry
from langchain_core.tools import Tool

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Input should be a valid Python expression."""
    try:
        result = eval(expression)
        return f"The result is {result}"
    except Exception as e:
        return f"Error: {str(e)}"

async def test_openai_agent():
    # Register calculator tool
    registry = get_tool_registry()
    calc_tool = Tool(
        name="calculator",
        description="Evaluate a mathematical expression. Input should be a valid Python expression.",
        func=calculator
    )
    registry.register_tool(calc_tool)
    
    orchestrator = get_orchestrator()
    orchestrator._setup_chains()
    
    if not orchestrator.agent_executor:
        print("❌ Agent executor not created")
        return
    
    # Test with OpenAI function calling
    result = await orchestrator.process_request(
        "What is 15 * 23 + 100?",
        use_tools=True
    )
    
    print(f"Response: {result.get('response')}")
    print(f"Success: {result.get('success')}")

if __name__ == "__main__":
    asyncio.run(test_openai_agent())
```

---

### Step 5: Test with Local LLM (Ollama) (10 minutes)

**If you have Ollama running:**

```python
# test_ollama_agent.py
import asyncio
from app.core.orchestrator import get_orchestrator
from app.core.tool_registry import get_tool_registry
from app.integrations.local_llm import get_local_llm, LLMBackend
from langchain_core.tools import Tool

def search_knowledge(query: str) -> str:
    """Search internal knowledge base. Input should be a search query."""
    return f"Found information about: {query}"

async def test_ollama_agent():
    # Get local LLM
    local_llm = get_local_llm(
        backend="ollama",
        model_name="llama3:8b"  # or your model
    )
    
    if not local_llm:
        print("❌ Local LLM not available. Is Ollama running?")
        return
    
    # Register tool
    registry = get_tool_registry()
    search_tool = Tool(
        name="search_knowledge",
        description="Search internal knowledge base. Input should be a search query.",
        func=search_knowledge
    )
    registry.register_tool(search_tool)
    
    # Create orchestrator with local LLM
    from app.core.orchestrator import WorkflowOrchestrator
    orchestrator = WorkflowOrchestrator(llm_provider=local_llm)
    
    if not orchestrator.agent_executor:
        print("❌ Agent executor not created")
        return
    
    # Test with ReAct agent
    result = await orchestrator.process_request(
        "Search for information about Python",
        use_tools=True
    )
    
    print(f"Response: {result.get('response')}")
    print(f"Success: {result.get('success')}")

if __name__ == "__main__":
    asyncio.run(test_ollama_agent())
```

---

### Step 6: Check Logs for Errors (5 minutes)

**Start the service and check logs:**

```bash
cd deepiri/diri-cyrex
python -m uvicorn app.main:app --reload
```

**Look for:**
- ✅ "Created OpenAI functions agent with X tools" or
- ✅ "Created ReAct agent with X tools"
- ❌ Any import errors
- ❌ Any agent creation failures

**Common issues:**
1. **Missing langchain packages** → Install dependencies
2. **No tools registered** → Agent executor will be None (normal)
3. **LLM not available** → Check OPENAI_API_KEY or Ollama connection
4. **Import errors** → Check Python environment

---

### Step 7: Test via API Endpoint (5 minutes)

**If the service is running, test via API:**

```bash
# Test orchestration endpoint
curl -X POST "http://localhost:8000/orchestration/process" \
  -H "Content-Type: application/json" \
  -H "x-api-key: change-me" \
  -d '{
    "user_input": "What is 2+2?",
    "use_tools": true
  }'
```

**Or use the orchestration API route:**
```python
# Via Python
import requests

response = requests.post(
    "http://localhost:8000/orchestration/process",
    json={
        "user_input": "What is 2+2?",
        "use_tools": True
    },
    headers={"x-api-key": "change-me"}
)

print(response.json())
```

---

## 🔍 Troubleshooting

### Issue: Agent executor is None

**Possible causes:**
1. No tools registered → Register at least one tool
2. LangChain agents not installed → Install dependencies
3. LLM not available → Check LLM provider initialization

**Fix:**
```python
# Check tool registry
from app.core.tool_registry import get_tool_registry
registry = get_tool_registry()
print(f"Tools: {len(registry.get_tools())}")

# Check LLM
from app.core.orchestrator import get_orchestrator
orchestrator = get_orchestrator()
print(f"LLM: {orchestrator.llm_provider}")
```

### Issue: Import errors

**Fix:**
```bash
pip install langchain langchain-core langchain-community langchain-openai
```

### Issue: Tool not being called

**Check:**
1. Tool is registered in tool registry
2. `use_tools=True` in request
3. Agent executor is initialized
4. Tool description is clear (LLM needs to understand when to use it)

---

## ✅ Success Criteria

After completing these steps, you should have:

1. ✅ Orchestrator initializes successfully
2. ✅ Agent executor is created (if tools exist)
3. ✅ Tools can be registered
4. ✅ Tools are executed when requested
5. ✅ Responses include tool results
6. ✅ Error handling works

---

## 📝 Next Steps After Testing

Once basic testing passes:

1. **Register your actual tools** - Add real tools for your use case
2. **Add tool error handling** - Enhance error messages
3. **Add tool monitoring** - Track tool usage metrics
4. **Optimize prompts** - Fine-tune agent prompts for your domain
5. **Add conversation memory** - Enable multi-turn conversations

---

## 🚨 If Something Fails

1. **Check logs** - Look for error messages
2. **Verify dependencies** - Ensure all packages are installed
3. **Test LLM connection** - Verify OpenAI API key or Ollama is running
4. **Check tool registration** - Ensure tools are properly registered
5. **Review code** - Check orchestrator.py for any syntax errors

---

**Time Estimate**: 30-45 minutes for all steps
**Priority**: HIGH - Verify implementation works before moving forward

