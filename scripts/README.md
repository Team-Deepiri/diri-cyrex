# Scripts Directory

Utility scripts for diri-cyrex service.

## Available Scripts

### `run_tests.py`
Master test runner with interactive CLI for running tests.

**Usage:**
```bash
# Interactive mode
python3 scripts/run_tests.py

# Run all tests
python3 scripts/run_tests.py --category all

# List available tests
python3 scripts/run_tests.py --list
```

See [tests/TEST_RUNNER_GUIDE.md](../tests/TEST_RUNNER_GUIDE.md) for complete documentation.

### `install-git-hooks.sh`
Installs Git hooks for branch protection.

### `setup_local_model.sh`
Sets up local LLM models (Ollama, etc.).

## Running Scripts

All scripts should be run from the project root directory:

```bash
cd deepiri/diri-cyrex
python3 scripts/run_tests.py
```

