# Cyrex Test Organization

## Test Structure

All test scripts are organized in the `tests/` folder:

```
tests/
├── integration/          # End-to-end integration tests
│   ├── test_agent_communication.py      # Agent-to-agent group chat communication
│   ├── test_group_chat.py               # Group chat functionality (async)
│   ├── test_group_chat_simple.py        # Group chat functionality (sync)
│   ├── test_agent_integration.py        # Agent tool registration and prompts
│   ├── test_api_integration.py          # API endpoint integration
│   └── test_langgraph.py                # LangGraph tool calling
├── ai/                  # AI/ML model tests
│   ├── test_task_classifier.py
│   ├── test_rag.py
│   ├── test_hybrid_ai.py
│   └── test_challenge_generator.py
└── test_comprehensive.py    # Comprehensive system tests
```

## Running Tests

### Integration Tests

**Agent-to-Agent Communication:**
```bash
# Comprehensive test (requires requests)
python tests/integration/test_agent_communication.py

# Simple test (uses requests)
python tests/integration/test_group_chat_simple.py

# Async test (uses aiohttp)
python tests/integration/test_group_chat.py
```

**Agent Integration:**
```bash
# Test tool registration and prompt loading
python tests/integration/test_agent_integration.py

# Test API endpoints
python tests/integration/test_api_integration.py

# Test LangGraph tool calling
python tests/integration/test_langgraph.py
```

### Prerequisites

- Backend running at `http://localhost:8000`
- Required Python packages:
  - `requests` (for most tests)
  - `aiohttp` (for async tests)

### Test Categories

1. **Integration Tests** (`tests/integration/`)
   - End-to-end functionality tests
   - API integration tests
   - Agent communication tests
   - Tool calling tests

2. **AI Tests** (`tests/ai/`)
   - Model-specific tests
   - RAG functionality
   - Classifier tests
   - Challenge generation

3. **Comprehensive Tests** (`tests/`)
   - Full system tests
   - Health checks
   - Guardrail tests

## Notes

- All test scripts in the root directory have been moved to `tests/integration/`
- Tests that are runtime functions used in `/app` remain in their original locations
- Integration tests require the backend to be running
- Some tests may require specific models to be available in Ollama

