# Development Tools

Development and testing utilities for diri-cyrex.

## Available Scripts

### `run_tests.py`
Master test runner with interactive CLI for running tests.

**Usage:**
```bash
# Interactive mode
python3 scripts/dev/run_tests.py

# Run all tests
python3 scripts/dev/run_tests.py --category all

# Run specific category
python3 scripts/dev/run_tests.py --category unit

# List available tests
python3 scripts/dev/run_tests.py --list
```

**Categories:**
- `unit` - Unit tests
- `integration` - Integration tests
- `api` - API endpoint tests
- `all` - All tests

See [tests/TEST_RUNNER_GUIDE.md](../../tests/TEST_RUNNER_GUIDE.md) for complete documentation.

### `check_modelkit_integration.py`
Checks ModelKit integration and connectivity.

**Usage:**
```bash
python3 scripts/dev/check_modelkit_integration.py
```

**What it does:**
- Tests ModelKit service connection
- Verifies API endpoints
- Checks authentication
- Validates model availability

### `install-git-hooks.sh`
Installs Git hooks for branch protection and code quality.

**Usage:**
```bash
bash scripts/dev/install-git-hooks.sh
```

**What it does:**
- Installs pre-commit hooks
- Sets up branch protection rules
- Configures code quality checks

### `setup-hooks.sh`
Convenience script to set up Git hooks (calls install-git-hooks.sh).

**Usage:**
```bash
bash scripts/dev/setup-hooks.sh
```

### `cyrex_watcher.py`
Intelligent file watcher for development that automatically reloads the application on code changes.

**Usage:**
```bash
python3 scripts/dev/cyrex_watcher.py
```

**Features:**
- Watches for code changes in the app directory
- Automatically reloads uvicorn on file changes
- Ignores cache files, logs, and non-code files
- Handles port conflicts gracefully

## Running Tests

### Quick Test
```bash
python3 scripts/dev/run_tests.py
```

### Full Test Suite
```bash
python3 scripts/dev/run_tests.py --category all --verbose
```

### Specific Test File
```bash
pytest tests/unit/test_specific.py -v
```

## Development Workflow

1. **Before committing:**
   ```bash
   bash scripts/dev/install-git-hooks.sh
   ```

2. **Run tests:**
   ```bash
   python3 scripts/dev/run_tests.py --category unit
   ```

3. **Check integrations:**
   ```bash
   python3 scripts/dev/check_modelkit_integration.py
   ```

## Troubleshooting

### Tests Failing
1. Check dependencies: `pip install -r requirements.txt`
2. Verify environment variables are set
3. Check service connections (Ollama, Milvus, etc.)

### Git Hooks Not Working
1. Verify hooks are installed: `ls -la .git/hooks/`
2. Reinstall: `bash scripts/dev/install-git-hooks.sh`
3. Check permissions: `chmod +x .git/hooks/*`

### ModelKit Connection Issues
1. Verify ModelKit service is running
2. Check authentication credentials
3. Verify network connectivity

