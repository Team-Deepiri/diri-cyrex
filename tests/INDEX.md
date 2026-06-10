# Test Suite Index

All test-related files are now organized in the `tests/` directory.

## Test Files

### Core Test Files
- `test_orchestrator.py` - Orchestrator and agent executor tests
- `test_tool_integration.py` - Tool registration and execution tests
- `test_ollama_agent.py` - Ollama/local LLM agent tests
- `test_orchestration_api.py` - API endpoint tests
- `test_health.py` - Health check tests
- `test_comprehensive.py` - Comprehensive integration tests

### Test Configuration
- `conftest.py` - Shared pytest fixtures
- `__init__.py` - Test package initialization

### Test Subdirectories
- `ai/` - AI-specific tests (bandit, challenge generator, RAG, etc.)
- `integration/` - Full integration tests

## Documentation

- **[README.md](README.md)** - Main test suite documentation
- **[TEST_RUNNER_GUIDE.md](TEST_RUNNER_GUIDE.md)** - Complete guide for using the test runner
- **[TEST_RUNNER_QUICK_REF.md](TEST_RUNNER_QUICK_REF.md)** - Quick command reference
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Step-by-step setup and testing instructions
- **[TEST_LOGGING_GUIDE.md](TEST_LOGGING_GUIDE.md)** - What logs to inspect and how to debug tests

## Test Runner

The master test runner script is in the scripts directory:
- `../scripts/run_tests.py` - Interactive CLI test runner

## Quick Start

```bash
# From project root
python3 scripts/run_tests.py                    # Interactive mode
python3 scripts/run_tests.py --category all     # Run all tests
python3 scripts/run_tests.py --list             # List available tests
```

See [TEST_RUNNER_QUICK_REF.md](TEST_RUNNER_QUICK_REF.md) for more commands.

