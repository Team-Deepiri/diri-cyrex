"""
Testing API Routes
Exposes test execution functionality via REST API
Integrates run_tests.py functionality into the backend
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import subprocess
import os
import json
import time
import threading
import queue
from pathlib import Path
from ..logging_config import get_logger

logger = get_logger("cyrex.testing_api")

router = APIRouter(prefix="/testing", tags=["testing"])

# Get project root directory
# In Docker: __file__ is /app/app/routes/testing_api.py, so parent.parent.parent = /app/
# In local: __file__ is .../diri-cyrex/app/routes/testing_api.py, so parent.parent.parent = .../diri-cyrex/
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
APP_DIR = Path(__file__).parent.parent.absolute()  # /app/app or .../diri-cyrex/app

# Try multiple possible locations for tests directory
# 1. Sibling to app directory (Docker: /app/tests, Local: .../diri-cyrex/tests)
# 2. Inside app directory (Local fallback: .../diri-cyrex/app/tests)
# 3. At project root (Docker: /app/tests, Local: .../diri-cyrex/tests)
possible_test_dirs = [
    APP_DIR.parent / "tests",  # Most common: sibling to app/
    APP_DIR / "tests",  # Fallback: inside app/
    PROJECT_ROOT / "tests",  # At project root
]

# Find the first existing tests directory (lenient check - just needs to be a directory)
# We'll do a more thorough check in the status endpoint
TESTS_DIR = None
for test_dir in possible_test_dirs:
    try:
        test_dir_path = test_dir.resolve()  # Resolve to absolute path
        if test_dir_path.exists() and test_dir_path.is_dir():
            # Just check if it's a directory - don't require test files to exist
            # This allows the directory to be detected even if it's empty or being populated
            TESTS_DIR = test_dir_path
            logger.info(f"Found tests directory at: {TESTS_DIR}")
            break
    except Exception as e:
        logger.debug(f"Error checking test directory {test_dir}: {e}")
        continue

# If none found, default to the most likely location (sibling to app/)
if TESTS_DIR is None:
    TESTS_DIR = (APP_DIR.parent / "tests").resolve()
    logger.info(
        f"Tests directory not found in any expected location. "
        f"Searched: {[str(d.resolve()) for d in possible_test_dirs]}. "
        f"Defaulting to: {TESTS_DIR}. "
        f"Project root: {PROJECT_ROOT}, App dir: {APP_DIR}."
    )


def _find_tests_directory():
    """Dynamically find tests directory - re-checks on each call"""
    for test_dir in possible_test_dirs:
        try:
            test_dir_path = test_dir.resolve()
            if test_dir_path.exists() and test_dir_path.is_dir():
                return test_dir_path
        except Exception:
            continue
    # Fallback to default
    return (APP_DIR.parent / "tests").resolve()

# Test categories and their files (from run_tests.py)
TEST_CATEGORIES = {
    "orchestrator": {
        "name": "Orchestrator Tests",
        "description": "Tests for WorkflowOrchestrator initialization, tool integration, and agent execution",
        "files": ["tests/test_orchestrator.py"],
        "markers": []
    },
    "tools": {
        "name": "Tool Integration Tests",
        "description": "Tests for tool registration, execution, and integration",
        "files": ["tests/test_tool_integration.py"],
        "markers": []
    },
    "ollama": {
        "name": "Ollama Agent Tests",
        "description": "Tests for Ollama/local LLM agent integration using ReAct pattern",
        "files": ["tests/test_ollama_agent.py"],
        "markers": []
    },
    "api": {
        "name": "API Endpoint Tests",
        "description": "Tests for /orchestration/* API endpoints",
        "files": ["tests/test_orchestration_api.py"],
        "markers": []
    },
    "integration": {
        "name": "Integration Tests",
        "description": "Full integration tests (may require external services)",
        "files": [
            "tests/test_orchestrator.py",
            "tests/test_tool_integration.py",
            "tests/test_ollama_agent.py",
            "tests/test_orchestration_api.py"
        ],
        "markers": ["-m", "integration"]
    },
    "all": {
        "name": "All Tests",
        "description": "Run all test files",
        "files": [
            "tests/test_orchestrator.py",
            "tests/test_tool_integration.py",
            "tests/test_ollama_agent.py",
            "tests/test_orchestration_api.py",
            "tests/test_health.py",
            "tests/test_comprehensive.py"
        ],
        "markers": []
    }
}

# Individual test files
TEST_FILES = {
    "orchestrator": "tests/test_orchestrator.py",
    "tools": "tests/test_tool_integration.py",
    "ollama": "tests/test_ollama_agent.py",
    "api": "tests/test_orchestration_api.py",
    "health": "tests/test_health.py",
    "comprehensive": "tests/test_comprehensive.py"
}


class TestRunRequest(BaseModel):
    """Request model for running tests"""
    category: Optional[str] = Field(None, description="Test category to run")
    file: Optional[str] = Field(None, description="Specific test file to run")
    test_path: Optional[str] = Field(None, description="Specific test class or function (e.g., TestClass::test_method)")
    verbose: bool = Field(True, description="Verbose output")
    coverage: bool = Field(False, description="Run with coverage")
    skip_slow: bool = Field(False, description="Skip slow tests")
    timeout: Optional[int] = Field(15, description="Timeout per test in seconds")
    output_format: str = Field("json", description="Output format: json or standard")


class TestListResponse(BaseModel):
    """Response model for listing tests"""
    categories: Dict[str, Any]
    files: Dict[str, str]


@router.get("/list", response_model=TestListResponse)
async def list_tests():
    """List all available test categories and files - fast endpoint, returns static data"""
    try:
        logger.info(f"Returning test list: {len(TEST_CATEGORIES)} categories, {len(TEST_FILES)} files")
        response = TestListResponse(
            categories=TEST_CATEGORIES,
            files=TEST_FILES
        )
        logger.debug(f"Response categories keys: {list(response.categories.keys())}")
        logger.debug(f"Response files keys: {list(response.files.keys())}")
        return response
    except Exception as e:
        logger.error(f"Error in list_tests: {e}", exc_info=True)
        # Return empty response on error rather than failing
        return TestListResponse(
            categories={},
            files={}
        )


@router.post("/run")
async def run_tests(request: TestRunRequest):
    """
    Run tests synchronously and return results.
    For real-time streaming, use /run/stream endpoint.
    """
    try:
        # Determine what to run
        test_files = []
        markers = []
        
        if request.category:
            if request.category not in TEST_CATEGORIES:
                raise HTTPException(status_code=400, detail=f"Unknown category: {request.category}")
            category = TEST_CATEGORIES[request.category]
            test_files = category["files"]
            markers = category.get("markers", [])
        elif request.file:
            if request.file not in TEST_FILES:
                raise HTTPException(status_code=400, detail=f"Unknown file: {request.file}")
            file_path = TEST_FILES[request.file]
            if request.test_path:
                test_files = [f"{file_path}::{request.test_path}"]
            else:
                test_files = [file_path]
        else:
            # Default to all tests
            test_files = TEST_CATEGORIES["all"]["files"]
        
        # Re-check tests directory location dynamically
        current_tests_dir = _find_tests_directory()
        
        # Check if tests directory exists
        if not current_tests_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Tests directory not found at {current_tests_dir}. Please create the tests directory and add test files."
            )
        
        # Build pytest command
        cmd = ["python3", "-m", "pytest"]
        
        # Add test files
        resolved_files = []
        for file_path in test_files:
            if file_path == "tests/" or file_path == "tests":
                if current_tests_dir.exists():
                    resolved_files.append("tests/")
                    continue
            
            if "::" in file_path:
                parts = file_path.split("::", 1)
                base_path = parts[0]
                test_path = parts[1]
                resolved_base = Path(PROJECT_ROOT / base_path).resolve()
                if resolved_base.exists():
                    resolved_files.append(f"{base_path}::{test_path}")
            else:
                resolved_path = Path(PROJECT_ROOT / file_path).resolve()
                if resolved_path.exists():
                    resolved_files.append(file_path)
        
        # Check if any test files were found
        if not resolved_files:
            # Try fallback: use tests directory if it exists
            if current_tests_dir.exists():
                resolved_files = ["tests/"]
                logger.info(f"No specific test files found, using tests directory: {current_tests_dir}")
            else:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"No test files found. Requested: {test_files}. "
                        f"Tests directory exists: {current_tests_dir.exists()}. "
                        f"Tests directory path: {current_tests_dir}. "
                        f"Project root: {PROJECT_ROOT}. "
                        f"Please ensure tests directory exists and contains test files."
                    )
                )
        
        cmd.extend(resolved_files)
        
        # Add markers
        if markers:
            cmd.extend(markers)
        
        # Skip slow tests if requested
        if request.skip_slow:
            cmd.extend(["-m", "not slow"])
        
        # Verbose output
        if request.verbose:
            cmd.append("-v")
        
        # Coverage
        if request.coverage:
            cmd.extend(["--cov=app", "--cov-report=term-missing", "--cov-report=json:coverage.json"])
        
        # Output format
        if request.output_format == "json":
            cmd.extend(["--json-report", "--json-report-file=test-report.json"])
        
        # Color output
        cmd.append("--color=yes")
        cmd.append("-s")  # Don't capture output
        cmd.append("--tb=short")  # Shorter tracebacks
        
        # Timeout - only add if pytest-timeout is available
        if request.timeout and request.timeout > 0:
            # Check if pytest-timeout is available by trying to import it
            try:
                import importlib
                importlib.import_module("pytest_timeout")
                # Plugin is available, add timeout arguments
                cmd.extend(["--timeout", str(request.timeout), "--timeout-method", "thread"])
            except ImportError:
                logger.warning("pytest-timeout plugin not installed, skipping timeout option. Install with: pip install pytest-timeout")
            except Exception as e:
                logger.warning(f"Could not check for pytest-timeout plugin: {e}, skipping timeout option")
        
        logger.info(f"Running tests with command: {' '.join(cmd)}")
        
        # Run tests
        original_cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute overall timeout
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            
            # Try to parse JSON report if available
            report_data = None
            if request.output_format == "json":
                report_path = PROJECT_ROOT / "test-report.json"
                if report_path.exists():
                    try:
                        with open(report_path, 'r') as f:
                            report_data = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to parse JSON report: {e}")
            
            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd),
                "report": report_data
            }
        finally:
            os.chdir(original_cwd)
            
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Tests exceeded timeout")
    except Exception as e:
        logger.error(f"Error running tests: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error running tests: {str(e)}")


@router.post("/run/stream")
async def run_tests_stream(request: TestRunRequest):
    """
    Run tests with real-time streaming output.
    Returns Server-Sent Events (SSE) stream.
    """
    async def generate():
        try:
            # Determine what to run (same logic as /run)
            test_files = []
            markers = []
            
            if request.category:
                if request.category not in TEST_CATEGORIES:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Unknown category: {request.category}'})}\n\n"
                    return
                category = TEST_CATEGORIES[request.category]
                test_files = category["files"]
                markers = category.get("markers", [])
            elif request.file:
                if request.file not in TEST_FILES:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Unknown file: {request.file}'})}\n\n"
                    return
                file_path = TEST_FILES[request.file]
                if request.test_path:
                    test_files = [f"{file_path}::{request.test_path}"]
                else:
                    test_files = [file_path]
            else:
                test_files = TEST_CATEGORIES["all"]["files"]
            
            # Re-check tests directory location dynamically
            current_tests_dir = _find_tests_directory()
            
            # Check if tests directory exists
            if not current_tests_dir.exists():
                yield f"data: {json.dumps({'type': 'error', 'message': f'Tests directory not found at {current_tests_dir}. Please create the tests directory and add test files.'})}\n\n"
                return
            
            # Build pytest command
            cmd = ["python3", "-m", "pytest"]
            
            # Add test files
            resolved_files = []
            for file_path in test_files:
                if file_path == "tests/" or file_path == "tests":
                    if current_tests_dir.exists():
                        resolved_files.append("tests/")
                        continue
                
                if "::" in file_path:
                    parts = file_path.split("::", 1)
                    base_path = parts[0]
                    test_path = parts[1]
                    resolved_base = Path(PROJECT_ROOT / base_path).resolve()
                    if resolved_base.exists():
                        resolved_files.append(f"{base_path}::{test_path}")
                else:
                    resolved_path = Path(PROJECT_ROOT / file_path).resolve()
                    if resolved_path.exists():
                        resolved_files.append(file_path)
            
            # Check if any test files were found
            if not resolved_files:
                # Try fallback: use tests directory if it exists
                if current_tests_dir.exists():
                    resolved_files = ["tests/"]
                    logger.info(f"No specific test files found, using tests directory: {current_tests_dir}")
                else:
                    error_msg = (
                        f"No test files found. Requested: {test_files}. "
                        f"Tests directory exists: {current_tests_dir.exists()}. "
                        f"Tests directory path: {current_tests_dir}. "
                        f"Project root: {PROJECT_ROOT}. "
                        f"Please ensure tests directory exists and contains test files."
                    )
                    yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                    return
            
            cmd.extend(resolved_files)
            
            if markers:
                cmd.extend(markers)
            
            if request.skip_slow:
                cmd.extend(["-m", "not slow"])
            
            if request.verbose:
                cmd.append("-v")
            
            if request.coverage:
                cmd.extend(["--cov=app", "--cov-report=term-missing"])
            
            cmd.append("--color=yes")
            cmd.append("-s")
            cmd.append("--tb=short")
            
            # Timeout - only add if pytest-timeout is available
            if request.timeout and request.timeout > 0:
                try:
                    import importlib
                    importlib.import_module("pytest_timeout")
                    # Plugin is available, add timeout arguments
                    cmd.extend(["--timeout", str(request.timeout), "--timeout-method", "thread"])
                except ImportError:
                    logger.warning("pytest-timeout plugin not installed, skipping timeout option. Install with: pip install pytest-timeout")
                except Exception as e:
                    logger.warning(f"Could not check for pytest-timeout plugin: {e}, skipping timeout option")
            
            # Send start event
            yield f"data: {json.dumps({'type': 'start', 'command': ' '.join(cmd)})}\n\n"
            
            # Run tests with streaming output
            original_cwd = os.getcwd()
            os.chdir(PROJECT_ROOT)
            
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"}
                )
                
                # Stream output line by line
                for line in process.stdout:
                    # Remove ANSI color codes for cleaner output
                    cleaned_line = line.rstrip()
                    yield f"data: {json.dumps({'type': 'output', 'line': cleaned_line})}\n\n"
                
                # Wait for process to complete
                return_code = process.wait()
                
                # Send completion event
                yield f"data: {json.dumps({'type': 'complete', 'return_code': return_code, 'success': return_code == 0})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            logger.error(f"Error in test stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# Cache pytest-timeout availability check to avoid repeated slow imports
_pytest_timeout_available_cache = None

def _check_pytest_timeout_cached():
    """Check if pytest-timeout is available (cached)"""
    global _pytest_timeout_available_cache
    if _pytest_timeout_available_cache is not None:
        return _pytest_timeout_available_cache
    
    try:
        import importlib
        importlib.import_module("pytest_timeout")
        _pytest_timeout_available_cache = True
    except ImportError:
        _pytest_timeout_available_cache = False
    except Exception:
        _pytest_timeout_available_cache = False
    
    return _pytest_timeout_available_cache


@router.get("/status")
async def get_test_status():
    """Get status of test infrastructure - fast endpoint, no blocking operations"""
    import asyncio
    # Return static data immediately, then try to enhance with filesystem checks
    # This ensures the frontend always gets useful data quickly
    static_response = {
        "project_root": str(PROJECT_ROOT),
        "tests_dir_exists": True,  # Assume it exists since tests are running
        "tests_dir": str(APP_DIR.parent / "tests"),  # Default path
        "test_file_count": len(TEST_FILES),  # Use known test files count
        "available_categories": list(TEST_CATEGORIES.keys()),
        "available_files": list(TEST_FILES.keys()),
        "categories": list(TEST_CATEGORIES.keys()),
        "files": list(TEST_FILES.keys()),
        "pytest_timeout_available": _check_pytest_timeout_cached(),
        "status": "ok"
    }
    
    try:
        # Try to enhance with filesystem checks, but don't wait long
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _get_test_status_internal),
                timeout=0.5  # Very short timeout - just enhance if possible
            )
            # Merge filesystem results with static data
            static_response.update({
                "tests_dir_exists": result.get("tests_dir_exists", True),
                "tests_dir": result.get("tests_dir", static_response["tests_dir"]),
                "test_file_count": result.get("test_file_count", static_response["test_file_count"]),
            })
            logger.debug(f"Test status check completed: tests_dir_exists={result.get('tests_dir_exists')}")
        except asyncio.TimeoutError:
            # Timeout is fine - we already have static data
            logger.debug("Test status filesystem check timed out, using static data")
        except Exception as e:
            logger.debug(f"Test status filesystem check failed: {e}, using static data")
        
        return static_response
    except Exception as e:
        logger.error(f"Error in get_test_status: {e}", exc_info=True)
        # Return static data even on error
        return static_response

def _get_test_status_internal():
    """Internal test status check - synchronous to avoid async issues - optimized for speed"""
    try:
        # Re-check tests directory location dynamically (in case volume mount changed)
        current_tests_dir = _find_tests_directory()
        
        # Fast synchronous check - no async operations that could block
        # Use try/except with timeouts to ensure we never block
        tests_dir_exists = False
        test_file_count = 0
        
        if current_tests_dir:
            try:
                # Fast exists() check - this is very fast
                tests_dir_exists = current_tests_dir.exists()
                
                # Count test files if directory exists - use listdir for speed instead of glob
                if tests_dir_exists:
                    try:
                        # Use listdir + filter instead of glob for better performance
                        # This is much faster than glob for large directories
                        import os
                        if os.path.isdir(str(current_tests_dir)):
                            # Quick listdir - this is very fast
                            files = os.listdir(str(current_tests_dir))
                            # Quick filter for test_*.py files - just count, don't load
                            # Use a simple loop with early exit for better performance
                            count = 0
                            for f in files:
                                if f.startswith("test_") and f.endswith(".py"):
                                    count += 1
                                    if count >= 50:  # Cap at 50 for display
                                        break
                            test_file_count = count
                    except (OSError, PermissionError, Exception) as e:
                        # If listdir fails, just skip counting - directory exists is enough
                        logger.debug(f"Could not count test files in {current_tests_dir}: {e}")
                        test_file_count = 0  # Set to 0 to indicate we couldn't count
            except Exception:
                pass  # Ignore errors - return what we can
        
        # Check pytest-timeout availability (cached, fast)
        pytest_timeout_available = False
        try:
            pytest_timeout_available = _check_pytest_timeout_cached()
        except Exception:
            pass  # Ignore errors
        
        # Get categories and files (these are static, fast)
        try:
            categories_list = list(TEST_CATEGORIES.keys())
            files_list = list(TEST_FILES.keys())
        except Exception:
            # Fallback if TEST_CATEGORIES or TEST_FILES not defined
            categories_list = []
            files_list = []
        
        # Return immediately with available metadata
        return {
            "project_root": str(PROJECT_ROOT),
            "tests_dir_exists": tests_dir_exists,
            "tests_dir": str(current_tests_dir) if current_tests_dir else "Not found",
            "test_file_count": test_file_count,
            "available_categories": categories_list,
            "available_files": files_list,
            "categories": categories_list,  # For backward compatibility
            "files": files_list,  # For backward compatibility
            "pytest_timeout_available": pytest_timeout_available,
            "status": "ok" if tests_dir_exists else "tests_dir_missing"
        }
    except Exception as e:
        logger.error(f"Error in _get_test_status_internal: {e}", exc_info=True)
        # Return minimal error response quickly
        return {
            "project_root": str(PROJECT_ROOT) if 'PROJECT_ROOT' in globals() else "unknown",
            "tests_dir_exists": False,
            "tests_dir": "error",
            "test_file_count": 0,
            "available_categories": [],
            "available_files": [],
            "categories": [],
            "files": [],
            "pytest_timeout_available": False,
            "status": "error",
            "error": str(e)
        }

