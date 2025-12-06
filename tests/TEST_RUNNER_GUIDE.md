# Test Runner Guide

Master test runner script with interactive CLI for running tests.

## Quick Start

### Interactive Mode (Recommended)
```bash
python3 scripts/run_tests.py
# or
python3 scripts/run_tests.py -i
```

This will show you an interactive menu to:
1. Select by category
2. Select by file name
3. Select specific test class or function
4. Run all tests
5. Exit

## Command Line Options

### List Available Tests
```bash
python3 scripts/run_tests.py --list
```

### Run by Category
```bash
# Run orchestrator tests
python3 scripts/run_tests.py --category orchestrator

# Run tool integration tests
python3 scripts/run_tests.py --category tools

# Run Ollama agent tests
python3 scripts/run_tests.py --category ollama

# Run API endpoint tests
python3 scripts/run_tests.py --category api

# Run integration tests only
python3 scripts/run_tests.py --category integration

# Run all tests
python3 scripts/run_tests.py --category all
```

### Run by File
```bash
# Run specific test file
python3 scripts/run_tests.py --file orchestrator
python3 scripts/run_tests.py --file tools
python3 scripts/run_tests.py --file ollama
python3 scripts/run_tests.py --file api
```

### Run Specific Test
```bash
# Run specific test class
python3 scripts/run_tests.py --file orchestrator --test "TestOrchestratorInitialization"

# Run specific test method
python3 scripts/run_tests.py --file orchestrator --test "TestOrchestratorInitialization::test_orchestrator_creation"
```

### Advanced Options

**Verbose Output:**
```bash
python3 scripts/run_tests.py --category all --verbose
```

**With Coverage:**
```bash
python3 scripts/run_tests.py --category all --coverage
```

**Skip Slow Tests:**
```bash
python3 scripts/run_tests.py --category all --no-slow
```

**JSON Output:**
```bash
python3 scripts/run_tests.py --category all --format json
```

## Test Categories

### orchestrator
- Tests for WorkflowOrchestrator
- Initialization, tool integration, agent execution
- File: `tests/test_orchestrator.py`

### tools
- Tool registration and execution tests
- Tool integration with orchestrator
- File: `tests/test_tool_integration.py`

### ollama
- Ollama/local LLM agent tests
- ReAct agent pattern tests
- File: `tests/test_ollama_agent.py`

### api
- API endpoint tests
- `/orchestration/*` endpoints
- File: `tests/test_orchestration_api.py`

### integration
- Full integration tests
- May require external services
- Uses `-m integration` marker

### all
- All test files
- Complete test suite

## Examples

### Example 1: Quick Test Run
```bash
# Run orchestrator tests with verbose output
python3 scripts/run_tests.py --category orchestrator --verbose
```

### Example 2: Full Test Suite with Coverage
```bash
# Run all tests with coverage report
python3 scripts/run_tests.py --category all --coverage
```

### Example 3: Specific Test
```bash
# Run only the orchestrator creation test
python3 scripts/run_tests.py --file orchestrator --test "TestOrchestratorInitialization::test_orchestrator_creation" --verbose
```

### Example 4: Integration Tests Only
```bash
# Run only integration tests
python3 scripts/run_tests.py --category integration
```

### Example 5: Fast Tests (Skip Slow)
```bash
# Run all tests except slow ones
python3 scripts/run_tests.py --category all --no-slow
```

## Interactive Mode Features

When running in interactive mode, you'll see:

1. **Available Test Categories** - List of all categories
2. **Selection Options** - Choose how to select tests
3. **File Selection** - Pick specific test files
4. **Test Path Selection** - Choose specific test classes/functions

## Output

### Standard Output
- Color-coded results
- Summary of passed/failed tests
- Error messages for failures

### Coverage Output
- Terminal coverage report
- HTML coverage report in `htmlcov/` directory
- Missing line coverage details

### JSON Output
- JSON report file: `test-report.json`
- Machine-readable format
- Useful for CI/CD integration

## Tips

1. **Start with interactive mode** to explore available tests
2. **Use `--list`** to see all available options
3. **Use `--no-slow`** for faster test runs during development
4. **Use `--coverage`** before committing to ensure good coverage
5. **Use `--verbose`** when debugging test failures

## Troubleshooting

### Tests Not Found
- Check that test files exist in `tests/` directory
- Verify you're in the `diri-cyrex` directory

### Import Errors
- Ensure dependencies are installed: `pip install -r requirements.txt`
- Check that you're using the correct Python environment

### Ollama Tests Skip
- This is normal if Ollama is not running
- Tests gracefully skip missing dependencies

### Permission Errors
- On Unix/Linux, you may need: `chmod +x run_tests.py`
- Or run with: `python3 scripts/run_tests.py`

## Integration with CI/CD

The test runner can be used in CI/CD pipelines:

```bash
# In CI/CD script
python3 scripts/run_tests.py --category all --format json --no-slow
```

This provides:
- Fast test execution
- JSON output for parsing
- Exit code for pass/fail detection

