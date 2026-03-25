# Test Logging Guide

Complete guide to understanding and inspecting logs during test execution.

## 📋 Table of Contents

1. [Log Locations](#log-locations)
2. [Log Types](#log-types)
3. [What to Inspect](#what-to-inspect)
4. [Common Log Patterns](#common-log-patterns)
5. [Debugging Failed Tests](#debugging-failed-tests)
6. [Log Configuration](#log-configuration)

---

## 📁 Log Locations

### Console Output (Primary)
**Location**: Standard output (stdout/stderr) during test execution

**When to check**: Always - this is the primary source of test information

**How to view**:
```bash
# Run tests with verbose output
python3 scripts/run_tests.py --category all --verbose

# Or with pytest directly
pytest -v -s  # -s shows print statements and logs
```

### Application Logs
**Location**: Configured via `LOG_FILE` environment variable (default: None)

**Default location**: If `LOG_FILE` is set, logs go to that file
- Example: `logs/cyrex.log`
- Example: `logs/test.log`

**When to check**: When tests interact with application code that logs

**How to view**:
```bash
# If LOG_FILE is set
tail -f logs/cyrex.log

# Or check the file
cat logs/cyrex.log
```

### Pytest Logs
**Location**: Console output (pytest captures logs by default)

**When to check**: Always during test execution

**How to view**:
```bash
# Show logs during test execution
pytest --log-cli-level=DEBUG

# Show logs and print statements
pytest -s --log-cli-level=INFO
```

### Test-Specific Log Files
**Location**: Created by test fixtures if configured

**When to check**: When tests create temporary log files

**How to view**:
```bash
# Check test output directory
ls -la tests/output/
```

---

## 🔍 Log Types

### 1. Test Execution Logs

**What they show**:
- Test discovery and execution
- Test pass/fail status
- Test duration
- Fixture setup/teardown

**Example**:
```
tests/test_orchestrator.py::TestOrchestratorInitialization::test_orchestrator_creation PASSED [ 10%]
tests/test_orchestrator.py::TestToolIntegration::test_tool_registration PASSED [ 20%]
```

**Key indicators**:
- `PASSED` - Test succeeded
- `FAILED` - Test failed (check error details)
- `SKIPPED` - Test was skipped (check reason)
- `ERROR` - Test setup/teardown error

### 2. Application Logs (from code under test)

**What they show**:
- Orchestrator initialization
- LLM provider connections
- Tool registration
- Agent executor creation
- RAG chain setup
- Vector store operations

**Logger names to watch**:
- `cyrex.orchestrator` - Orchestrator operations
- `cyrex.tool_registry` - Tool registration/execution
- `cyrex.local_llm` - Local LLM operations
- `cyrex.openai_wrapper` - OpenAI operations
- `cyrex.milvus_store` - Vector store operations
- `cyrex.rag_bridge` - RAG operations

**Example**:
```json
{
  "event": "Created OpenAI functions agent with 2 tools",
  "logger": "cyrex.orchestrator",
  "level": "info",
  "timestamp": "2025-01-15T10:30:45.123Z"
}
```

### 3. Error Logs

**What they show**:
- Exceptions and stack traces
- Import errors
- Connection failures
- Configuration issues

**Key indicators**:
- `ERROR` level logs
- Exception stack traces
- `Traceback` messages

**Example**:
```
ERROR cyrex.orchestrator: Failed to setup chains: ImportError: No module named 'langchain.agents'
Traceback (most recent call last):
  File "app/core/orchestrator.py", line 198, in _setup_chains
    from langchain.agents import create_openai_functions_agent
ImportError: No module named 'langchain.agents'
```

### 4. Warning Logs

**What they show**:
- Missing optional dependencies
- Fallback behaviors
- Deprecated features
- Configuration issues

**Key indicators**:
- `WARNING` level logs
- Messages about missing features
- Fallback notifications

**Example**:
```
WARNING cyrex.orchestrator: LangChain agents not available: No module named 'langchain.agents'
WARNING cyrex.local_llm: Could not verify Ollama connection at http://localhost:11434
```

---

## 🔎 What to Inspect

### For Test Failures

#### 1. Check Test Output
```bash
# Run with verbose output to see full error
pytest -v tests/test_orchestrator.py::TestOrchestratorInitialization::test_orchestrator_creation
```

**Look for**:
- Assertion errors
- Exception messages
- Stack traces
- Missing dependencies

#### 2. Check Application Logs
**Look for**:
- Initialization errors
- Connection failures
- Import errors
- Configuration problems

**Key loggers**:
- `cyrex.orchestrator` - Orchestrator setup issues
- `cyrex.tool_registry` - Tool registration problems
- `cyrex.local_llm` - LLM connection issues

#### 3. Check Warning Messages
**Look for**:
- Missing optional dependencies
- Fallback behaviors
- Configuration warnings

**Common warnings**:
```
WARNING: LangChain agents not available
WARNING: Ollama not available
WARNING: Milvus not available
```

### For Test Success Verification

#### 1. Check Initialization Logs
**Look for**:
```
INFO cyrex.orchestrator: Using OpenAI: gpt-4
INFO cyrex.orchestrator: Created OpenAI functions agent with 2 tools
INFO cyrex.tool_registry: Registered tool: get_weather
```

#### 2. Check Execution Logs
**Look for**:
```
INFO cyrex.orchestrator: Using agent executor with tools for request
INFO cyrex.orchestrator: Agent executor completed with 1 tool calls
```

#### 3. Check Health Status
**Look for**:
```
INFO cyrex.local_llm: Successfully verified Ollama connection at http://localhost:11434
INFO cyrex.milvus_store: Connected to Milvus at localhost:19530
```

---

## 📊 Common Log Patterns

### Successful Test Run

```
INFO cyrex.orchestrator: Orchestrator created
INFO cyrex.tool_registry: Registered tool: get_weather
INFO cyrex.orchestrator: Created OpenAI functions agent with 1 tools
INFO cyrex.orchestrator: Using agent executor with tools for request
INFO cyrex.orchestrator: Agent executor completed with 1 tool calls
```

### Missing Dependencies

```
WARNING cyrex.orchestrator: LangChain agents not available: No module named 'langchain.agents'
WARNING cyrex.orchestrator: Tools available but agents not available or no tools registered
```

### LLM Connection Issues

```
WARNING cyrex.local_llm: Could not verify Ollama connection at http://localhost:11434
WARNING cyrex.local_llm: Will attempt to use it anyway
ERROR cyrex.local_llm: LLM invocation failed: Connection refused
```

### Tool Execution Issues

```
INFO cyrex.tool_registry: Executed tool: get_weather
ERROR cyrex.tool_registry: Tool execution failed: get_weather, error: KeyError: 'city'
```

### Agent Executor Issues

```
WARNING cyrex.orchestrator: Agent executor failed: AgentExecutor error, falling back to direct LLM
INFO cyrex.orchestrator: Falling back to direct LLM call
```

---

## 🐛 Debugging Failed Tests

### Step 1: Enable Verbose Logging

```bash
# Set log level to DEBUG
export LOG_LEVEL=DEBUG

# Run tests with verbose output
python3 scripts/run_tests.py --category orchestrator --verbose
```

### Step 2: Check Specific Test

```bash
# Run single test with full output
pytest -v -s tests/test_orchestrator.py::TestOrchestratorInitialization::test_orchestrator_creation

# With debug logging
LOG_LEVEL=DEBUG pytest -v -s tests/test_orchestrator.py::TestOrchestratorInitialization::test_orchestrator_creation
```

### Step 3: Inspect Log Output

**Look for these patterns**:

1. **Import Errors**:
   ```
   ImportError: No module named 'langchain.agents'
   ```
   **Fix**: Install missing package: `pip install langchain`

2. **Connection Errors**:
   ```
   ConnectionError: Connection refused
   ```
   **Fix**: Start required service (Ollama, Milvus, etc.)

3. **Configuration Errors**:
   ```
   ValueError: OPENAI_API_KEY not configured
   ```
   **Fix**: Set required environment variables

4. **Tool Errors**:
   ```
   ValueError: Tool get_weather not found
   ```
   **Fix**: Check tool registration in test

### Step 4: Check Application State

**Inspect logs for**:
- Component initialization status
- Connection status
- Configuration values

**Example log inspection**:
```bash
# Filter logs by component
grep "cyrex.orchestrator" logs/cyrex.log | tail -20

# Filter by log level
grep "ERROR" logs/cyrex.log | tail -20

# Filter by test context
grep "test_orchestrator" logs/cyrex.log
```

---

## ⚙️ Log Configuration

### Environment Variables

**Set log level**:
```bash
export LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

**Set log file**:
```bash
export LOG_FILE=logs/test.log
```

**In test environment** (via `conftest.py`):
```python
@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FILE", "logs/test.log")
```

### Pytest Log Configuration

**In `pytest.ini`** (add if needed):
```ini
[pytest]
log_cli = true
log_cli_level = INFO
log_cli_format = %(asctime)s [%(levelname)8s] %(name)s: %(message)s
log_cli_date_format = %Y-%m-%d %H:%M:%S
```

### Structured Logging

The application uses `structlog` for structured logging. Logs are in JSON format when written to files.

**Example structured log**:
```json
{
  "event": "Agent executor completed with 1 tool calls",
  "logger": "cyrex.orchestrator",
  "level": "info",
  "timestamp": "2025-01-15T10:30:45.123Z",
  "request_id": "req_1234567890",
  "tool_calls": 1
}
```

---

## 📝 Log Inspection Commands

### Quick Log Checks

```bash
# View recent logs
tail -f logs/cyrex.log

# Search for errors
grep -i error logs/cyrex.log | tail -20

# Search for warnings
grep -i warning logs/cyrex.log | tail -20

# Search by component
grep "cyrex.orchestrator" logs/cyrex.log | tail -20

# Search by test
grep "test_orchestrator" logs/cyrex.log
```

### During Test Execution

```bash
# Run tests and save output
python3 scripts/run_tests.py --category all --verbose 2>&1 | tee test_output.log

# Run with debug logging
LOG_LEVEL=DEBUG python3 scripts/run_tests.py --category orchestrator --verbose 2>&1 | tee debug_test.log
```

### Filter Logs by Component

```bash
# Orchestrator logs
grep "cyrex.orchestrator" logs/cyrex.log

# Tool registry logs
grep "cyrex.tool_registry" logs/cyrex.log

# LLM provider logs
grep "cyrex.local_llm\|cyrex.openai_wrapper" logs/cyrex.log

# Vector store logs
grep "cyrex.milvus_store" logs/cyrex.log
```

---

## 🎯 Key Log Messages to Watch

### Initialization

**Success**:
- `✅ Orchestrator created`
- `✅ Agent executor initialized`
- `✅ Tool registry has X tools`
- `✅ LLM provider: {'status': 'healthy'}`

**Failure**:
- `❌ Agent executor not created`
- `⚠️ LLM provider not initialized`
- `WARNING: LangChain agents not available`

### Tool Execution

**Success**:
- `✅ Registered tool: tool_name`
- `✅ Executed tool: tool_name`
- `✅ Agent executor completed with X tool calls`

**Failure**:
- `❌ Tool execution failed`
- `ERROR: Tool tool_name not found`
- `WARNING: Agent executor failed, falling back to direct LLM`

### LLM Operations

**Success**:
- `✅ Using OpenAI: gpt-4`
- `✅ Using local LLM: ollama`
- `✅ Successfully verified Ollama connection`

**Failure**:
- `❌ Local LLM not available`
- `ERROR: LLM invocation failed`
- `WARNING: Could not verify Ollama connection`

### RAG Operations

**Success**:
- `✅ Connected to Milvus at localhost:19530`
- `✅ Retrieved X documents from Milvus`

**Failure**:
- `WARNING: Milvus not available`
- `ERROR: RAG retrieval failed`
- `WARNING: RAG retrieval failed, falling back`

---

## 🔧 Troubleshooting with Logs

### Test Fails: "Agent executor not created"

**Check logs for**:
1. LangChain availability:
   ```
   WARNING: LangChain agents not available
   ```
   **Fix**: `pip install langchain langchain-community`

2. Tool registration:
   ```
   WARNING: Tools available but agents not available
   ```
   **Fix**: Check tool registry has tools registered

3. LLM provider:
   ```
   WARNING: LLM provider not available
   ```
   **Fix**: Configure OPENAI_API_KEY or start Ollama

### Test Fails: "Tool execution failed"

**Check logs for**:
1. Tool not found:
   ```
   ERROR: Tool tool_name not found
   ```
   **Fix**: Register tool before use

2. Tool execution error:
   ```
   ERROR: Tool execution failed: tool_name, error: ...
   ```
   **Fix**: Check tool implementation

3. Agent executor error:
   ```
   WARNING: Agent executor failed, falling back
   ```
   **Fix**: Check agent executor setup

### Test Fails: "LLM not available"

**Check logs for**:
1. Connection issues:
   ```
   WARNING: Could not verify Ollama connection
   ```
   **Fix**: Start Ollama: `ollama serve`

2. API key missing:
   ```
   WARNING: OPENAI_API_KEY not configured
   ```
   **Fix**: Set OPENAI_API_KEY environment variable

3. Import errors:
   ```
   ImportError: No module named 'langchain_openai'
   ```
   **Fix**: `pip install langchain-openai`

---

## 📚 Additional Resources

- [Test Runner Guide](TEST_RUNNER_GUIDE.md) - How to run tests
- [Setup Guide](SETUP_GUIDE.md) - Test setup instructions
- [README](README.md) - Test suite overview

---

## 💡 Tips

1. **Always run with `--verbose`** when debugging
2. **Set `LOG_LEVEL=DEBUG`** for maximum detail
3. **Use `-s` flag with pytest** to see print statements
4. **Check logs immediately after test failure** - they contain context
5. **Filter logs by component** to narrow down issues
6. **Look for WARNING messages** - they often explain failures
7. **Check both console output and log files** if configured

---

---

## 🚀 Quick Reference: Common Scenarios

### Scenario 1: Test Fails with "Agent executor not created"

**What to check**:
```bash
# 1. Check for LangChain import errors
grep -i "langchain.*not available" test_output.log

# 2. Check tool registration
grep "Registered tool" test_output.log

# 3. Check LLM provider status
grep "LLM provider" test_output.log
```

**Expected logs if working**:
```
INFO cyrex.tool_registry: Registered tool: get_weather
INFO cyrex.orchestrator: Created OpenAI functions agent with 1 tools
```

**If missing**:
```
WARNING: LangChain agents not available
WARNING: Tools available but agents not available
```

### Scenario 2: Test Fails with "Tool execution failed"

**What to check**:
```bash
# 1. Check tool registration
grep "Registered tool" test_output.log

# 2. Check tool execution
grep "Executed tool\|Tool execution failed" test_output.log

# 3. Check agent executor
grep "Agent executor\|agent executor" test_output.log
```

**Expected logs if working**:
```
INFO cyrex.tool_registry: Executed tool: get_weather
INFO cyrex.orchestrator: Agent executor completed with 1 tool calls
```

**If failing**:
```
ERROR cyrex.tool_registry: Tool execution failed: get_weather, error: ...
WARNING cyrex.orchestrator: Agent executor failed, falling back
```

### Scenario 3: Test Fails with "LLM not available"

**What to check**:
```bash
# 1. Check LLM provider initialization
grep "LLM provider\|Using.*LLM" test_output.log

# 2. Check connection status
grep "Ollama\|OpenAI\|connection" test_output.log

# 3. Check API key
grep "API.*key\|OPENAI" test_output.log
```

**Expected logs if working**:
```
INFO cyrex.orchestrator: Using OpenAI: gpt-4
# OR
INFO cyrex.local_llm: Successfully verified Ollama connection
INFO cyrex.orchestrator: Using local LLM: ollama
```

**If failing**:
```
WARNING: OPENAI_API_KEY not configured
WARNING: Could not verify Ollama connection
ERROR: LLM provider not initialized
```

### Scenario 4: Test Passes but No Tool Usage

**What to check**:
```bash
# 1. Check if agent executor was created
grep "Agent executor\|agent executor" test_output.log

# 2. Check if tools were registered
grep "Registered tool" test_output.log

# 3. Check if use_tools was True
grep "use_tools\|Using agent executor" test_output.log
```

**Expected logs if tools used**:
```
INFO cyrex.orchestrator: Using agent executor with tools for request
INFO cyrex.orchestrator: Agent executor completed with X tool calls
```

**If tools not used**:
```
# No "Using agent executor" message
# Direct LLM call instead
```

---

## 📋 Log Inspection Checklist

When a test fails, check these in order:

- [ ] **Test output** - Read the full error message
- [ ] **Stack trace** - Identify where the error occurred
- [ ] **Warning messages** - Often explain why something didn't work
- [ ] **Component initialization** - Check if components started correctly
- [ ] **Connection status** - Verify external services are available
- [ ] **Configuration** - Check environment variables and settings
- [ ] **Dependencies** - Verify all required packages are installed

---

**Last Updated**: 2025-01-15
**Maintained by**: Test Suite Documentation

