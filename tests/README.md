# Test Suite for diri-cyrex

Comprehensive test suite for LangChain infrastructure, orchestrator, tools, and API endpoints.

## Quick Links

- **[Test Runner Guide](TEST_RUNNER_GUIDE.md)** - Complete guide for using the test runner
- **[Quick Reference](TEST_RUNNER_QUICK_REF.md)** - Quick command reference
- **[Setup Guide](SETUP_GUIDE.md)** - Step-by-step setup and testing instructions
- **[Logging Guide](TEST_LOGGING_GUIDE.md)** - What logs to inspect and how to debug tests
- **[Test Runner Script](../scripts/run_tests.py)** - Master test runner with interactive CLI

## Test Files

### `test_orchestrator.py`
Tests for `WorkflowOrchestrator` initialization, tool integration, and agent execution.

**Test Classes:**
- `TestOrchestratorInitialization` - Basic orchestrator setup
- `TestToolIntegration` - Tool registration and execution
- `TestAgentExecutor` - Agent executor creation and usage
- `TestRAGIntegration` - RAG chain setup and usage
- `TestErrorHandling` - Error handling and edge cases
- `TestStreaming` - Streaming functionality
- `TestIntegration` - Full integration tests

### `test_tool_integration.py`
Comprehensive tool integration tests.

**Test Classes:**
- `TestToolRegistration` - Tool registration and management
- `TestToolExecution` - Direct tool execution
- `TestToolWithOrchestrator` - Tools with orchestrator
- `TestToolErrorHandling` - Tool error handling
- `TestToolIntegrationScenarios` - Integration scenarios

### `test_ollama_agent.py`
Tests for Ollama/local LLM agent integration using ReAct pattern.

**Test Classes:**
- `TestOllamaConnection` - Ollama connection and availability
- `TestOllamaAgent` - ReAct agent with Ollama
- `TestOllamaErrorHandling` - Error handling with Ollama
- `TestOllamaIntegration` - Integration tests
- `TestOllamaPerformance` - Performance tests

### `test_orchestration_api.py`
API endpoint tests for `/orchestration/*` endpoints.

**Test Classes:**
- `TestOrchestrationProcessEndpoint` - `/orchestration/process` endpoint
- `TestOrchestrationStatusEndpoint` - `/orchestration/status` endpoint
- `TestOrchestrationWorkflowEndpoint` - `/orchestration/workflow` endpoint
- `TestOrchestrationLocalLLM` - Local LLM integration
- `TestOrchestrationAPIValidation` - Input validation
- `TestOrchestrationAPIIntegration` - Full integration tests

## Running Tests

### Run All Tests
```bash
cd deepiri/diri-cyrex
pytest

# Or use the test runner
python3 scripts/run_tests.py --category all
```

### Run Specific Test File
```bash
pytest tests/test_orchestrator.py
pytest tests/test_tool_integration.py
pytest tests/test_ollama_agent.py
pytest tests/test_orchestration_api.py
```

### Run Specific Test Class
```bash
pytest tests/test_orchestrator.py::TestOrchestratorInitialization
```

### Run Specific Test
```bash
pytest tests/test_orchestrator.py::TestOrchestratorInitialization::test_orchestrator_creation
```

### Run with Markers

**Integration tests only:**
```bash
pytest -m integration
```

**Skip slow tests:**
```bash
pytest -m "not slow"
```

**AI-related tests:**
```bash
pytest -m ai
```

### Run with Verbose Output
```bash
pytest -v
```

### Run with Coverage
```bash
pytest --cov=app --cov-report=html
```

## Test Markers

- `@pytest.mark.integration` - Integration tests (may require external services)
- `@pytest.mark.slow` - Slow tests (may take longer to run)
- `@pytest.mark.asyncio` - Async tests (automatically handled by pytest-asyncio)

## Prerequisites

### Required Packages
```bash
pip install pytest pytest-asyncio pytest-cov
```

### Optional (for full test coverage)
- Ollama running locally (for Ollama agent tests)
- OpenAI API key (for OpenAI agent tests)
- Milvus/Chroma (for vector store tests)

## Test Fixtures

Shared fixtures are in `conftest.py`:
- `mock_llm_provider` - Mock LLM provider
- `mock_vector_store` - Mock vector store
- `sample_weather_tool` - Sample weather tool
- `sample_calculator_tool` - Sample calculator tool
- `sample_search_tool` - Sample search tool
- `clean_tool_registry` - Clean tool registry instance

## Skipping Tests

Tests that require external services will be skipped if:
- LangChain agents are not installed
- Ollama is not available
- LLM provider is not configured

This is expected behavior - tests gracefully skip when dependencies are missing.

## Writing New Tests

### Example Test Structure
```python
import pytest
from app.core.orchestrator import WorkflowOrchestrator

@pytest.mark.asyncio
class TestMyFeature:
    """Test description"""
    
    async def test_my_feature(self, mock_llm_provider):
        """Test specific feature"""
        orchestrator = WorkflowOrchestrator(llm_provider=mock_llm_provider)
        result = await orchestrator.process_request("test")
        assert result is not None
```

### Best Practices
1. Use fixtures from `conftest.py` when possible
2. Mark integration tests with `@pytest.mark.integration`
3. Mark slow tests with `@pytest.mark.slow`
4. Use descriptive test names
5. Test both success and error cases
6. Mock external dependencies

## Troubleshooting

### Tests Fail with Import Errors
```bash
# Install missing dependencies
pip install -r requirements.txt
```

### Ollama Tests Skip
- Ensure Ollama is running: `ollama serve`
- Check model is available: `ollama list`
- Set `LOCAL_LLM_MODEL` environment variable

### LangChain Agent Tests Skip
```bash
# Install LangChain agents
pip install langchain langchain-community
```

### Async Test Issues
- Ensure `pytest-asyncio` is installed
- Check `asyncio_mode = auto` in `pytest.ini`

## Continuous Integration

These tests are designed to run in CI/CD pipelines:
- Fast unit tests run first
- Integration tests run if external services are available
- Tests gracefully skip missing dependencies
- All tests should pass or skip (never fail due to missing deps)

## Coverage Goals

- **Unit Tests**: 80%+ coverage
- **Integration Tests**: Cover all major workflows
- **API Tests**: Cover all endpoints
- **Error Handling**: Test all error paths

