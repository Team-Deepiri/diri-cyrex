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

# Test categories and their files - dynamically discovered from actual test files
def _discover_test_files():
    """Discover test files from the tests directory"""
    test_files = {}
    categories = {}
    
    # Re-check tests directory location dynamically
    current_tests_dir = _find_tests_directory()
    
    if not current_tests_dir or not current_tests_dir.exists():
        logger.warning(f"Tests directory not found at {current_tests_dir}, using default test files")
        # Return defaults if directory doesn't exist
        return _get_default_test_config()
    
    # Discover test files recursively
    try:
        for test_file in current_tests_dir.rglob("test_*.py"):
            # Get relative path from PROJECT_ROOT
            try:
                rel_path = test_file.relative_to(PROJECT_ROOT)
                rel_path_str = str(rel_path).replace("\\", "/")  # Normalize path separators
                
                # Create a key from the filename (without extension)
                key = test_file.stem  # e.g., "test_health" from "test_health.py"
                
                # Skip if already added (prefer shorter paths)
                if key not in test_files or len(rel_path_str) < len(test_files[key]):
                    test_files[key] = rel_path_str
            except ValueError:
                # File is not relative to PROJECT_ROOT, skip it
                continue
        
        # Build categories based on directory structure
        categories = _build_categories_from_files(test_files, current_tests_dir)
        
        logger.info(f"Discovered {len(test_files)} test files in {current_tests_dir}")
        return categories, test_files
        
    except Exception as e:
        logger.error(f"Error discovering test files: {e}", exc_info=True)
        return _get_default_test_config()

def _build_categories_from_files(test_files: Dict[str, str], tests_dir: Path) -> Dict[str, Any]:
    """Build test categories from discovered files"""
    categories = {}
    
    # Group files by directory
    by_dir = {}
    for key, file_path in test_files.items():
        # Extract directory from path (e.g., "tests/ai" from "tests/ai/test_rag.py")
        parts = file_path.split("/")
        if len(parts) > 2:  # Has subdirectory
            dir_name = parts[1]  # e.g., "ai" or "integration"
            if dir_name not in by_dir:
                by_dir[dir_name] = []
            by_dir[dir_name].append(file_path)
        else:  # Root tests directory
            if "root" not in by_dir:
                by_dir["root"] = []
            by_dir["root"].append(file_path)
    
    # Create category for each directory
    for dir_name, files in by_dir.items():
        if dir_name == "root":
            category_key = "core"
            category_name = "Core Tests"
            category_desc = "Core functionality tests"
        elif dir_name == "ai":
            category_key = "ai"
            category_name = "AI Tests"
            category_desc = "AI and ML model tests"
        elif dir_name == "service":
            category_key = "service"
            category_name = "Service Tests"
            category_desc = "Service and ML model tests"
        elif dir_name == "integration":
            category_key = "integration"
            category_name = "Integration Tests"
            category_desc = "Full integration tests (may require external services)"
        else:
            category_key = dir_name
            category_name = f"{dir_name.title()} Tests"
            category_desc = f"Tests in {dir_name} directory"
        
        categories[category_key] = {
            "name": category_name,
            "description": category_desc,
            "files": sorted(files),
            "markers": []
        }
    
    # Add "all" category
    all_files = sorted(test_files.values())
    categories["all"] = {
        "name": "All Tests",
        "description": "Run all test files",
        "files": all_files,
        "markers": []
    }
    
    return categories

def _get_default_test_config():
    """Return default test configuration if discovery fails"""
    default_categories = {
        "core": {
            "name": "Core Tests",
            "description": "Core functionality tests",
            "files": [
                "tests/test_health.py",
                "tests/test_comprehensive.py",
                "tests/test_cyrex_guard.py"
            ],
            "markers": []
        },
        "service": {
            "name": "Service Tests",
            "description": "Service and ML model tests",
            "files": [
                "tests/service/test_rag.py",
                "tests/service/test_hybrid_ai.py",
                "tests/service/test_task_classifier.py",
                "tests/service/test_challenge_generator.py",
                "tests/service/test_bandit.py"
            ],
            "markers": []
        },
        "integration": {
            "name": "Integration Tests",
            "description": "Full integration tests (may require external services)",
            "files": [
                "tests/integration/test_api_integration.py",
                "tests/integration/test_agent_integration.py",
                "tests/integration/test_agent_communication.py",
                "tests/integration/test_full_pipeline.py",
                "tests/integration/test_group_chat.py",
                "tests/integration/test_group_chat_simple.py",
                "tests/integration/test_langgraph.py"
            ],
            "markers": []
        },
        "all": {
            "name": "All Tests",
            "description": "Run all test files",
            "files": [
                "tests/test_health.py",
                "tests/test_comprehensive.py",
                "tests/test_cyrex_guard.py",
                "tests/service/test_rag.py",
                "tests/service/test_hybrid_ai.py",
                "tests/service/test_task_classifier.py",
                "tests/service/test_challenge_generator.py",
                "tests/service/test_bandit.py",
                "tests/integration/test_api_integration.py",
                "tests/integration/test_agent_integration.py",
                "tests/integration/test_agent_communication.py",
                "tests/integration/test_full_pipeline.py",
                "tests/integration/test_group_chat.py",
                "tests/integration/test_group_chat_simple.py",
                "tests/integration/test_langgraph.py"
            ],
            "markers": []
        }
    }
    
    default_files = {
        "health": "tests/test_health.py",
        "comprehensive": "tests/test_comprehensive.py",
        "cyrex_guard": "tests/test_cyrex_guard.py",
        "rag": "tests/service/test_rag.py",
        "hybrid_ai": "tests/service/test_hybrid_ai.py",
        "task_classifier": "tests/service/test_task_classifier.py",
        "challenge_generator": "tests/service/test_challenge_generator.py",
        "bandit": "tests/service/test_bandit.py",
        "api_integration": "tests/integration/test_api_integration.py",
        "agent_integration": "tests/integration/test_agent_integration.py",
        "agent_communication": "tests/integration/test_agent_communication.py",
        "full_pipeline": "tests/integration/test_full_pipeline.py",
        "group_chat": "tests/integration/test_group_chat.py",
        "group_chat_simple": "tests/integration/test_group_chat_simple.py",
        "langgraph": "tests/integration/test_langgraph.py"
    }
    
    return default_categories, default_files

# Discover test files on module load - wrap in try/except to prevent module load failures
try:
    TEST_CATEGORIES, TEST_FILES = _discover_test_files()
    logger.info(f"Initialized test configuration: {len(TEST_CATEGORIES)} categories, {len(TEST_FILES)} files")
except Exception as e:
    logger.error(f"Failed to discover test files on module load: {e}", exc_info=True)
    # Use defaults to ensure module loads successfully
    TEST_CATEGORIES, TEST_FILES = _get_default_test_config()
    logger.info(f"Using default test configuration: {len(TEST_CATEGORIES)} categories, {len(TEST_FILES)} files")


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


@router.get("/list")
async def list_tests():
    """List all available test categories and files - dynamically discovers test files"""
    try:
        # Re-discover test files on each request to ensure we have the latest
        # This allows the list to update if test files are added/removed
        categories, files = _discover_test_files()
        
        logger.info(f"Returning test list: {len(categories)} categories, {len(files)} files")
        logger.debug(f"Categories: {list(categories.keys())}")
        logger.debug(f"Files: {list(files.keys())}")
        
        # Ensure we always return valid data
        if not categories:
            logger.warning("No categories found, using defaults")
            categories, files = _get_default_test_config()
        
        # Return as dict to ensure JSON serialization works
        return {
            "categories": categories,
            "files": files
        }
    except Exception as e:
        logger.error(f"Error in list_tests: {e}", exc_info=True)
        # Try to return defaults on error
        try:
            categories, files = _get_default_test_config()
            return {
                "categories": categories,
                "files": files
            }
        except Exception as fallback_error:
            logger.error(f"Error getting default test config: {fallback_error}", exc_info=True)
            # Last resort: return empty but valid structure
            return {
                "categories": {},
                "files": {}
            }


def _get_current_test_config():
    """Get current test configuration, refreshing if needed"""
    # For now, use module-level variables for performance
    # These are refreshed in the /list endpoint
    return TEST_CATEGORIES, TEST_FILES

def _parse_pytest_output(output_text: str) -> Dict[str, Any]:
    """Parse pytest output to extract test results"""
    import re
    
    # Initialize summary
    summary = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "error": 0,
        "warnings": 0,
        "total": 0,
        "warning": None
    }
    
    # Patterns to match pytest output
    # Example: "1 passed, 1 skipped, 3 warnings in 12.30s"
    # Or: "1 failed, 2 passed in 5.20s"
    summary_pattern = r'(\d+)\s+(passed|failed|skipped|error|warnings)'
    
    # Find all matches
    matches = re.findall(summary_pattern, output_text.lower())
    
    for count, status in matches:
        count = int(count)
        if status == "passed":
            summary["passed"] = count
        elif status == "failed":
            summary["failed"] = count
        elif status == "skipped":
            summary["skipped"] = count
        elif status == "error":
            summary["error"] = count
        elif status == "warnings":
            summary["warnings"] = count
    
    # Calculate total
    summary["total"] = summary["passed"] + summary["failed"] + summary["skipped"] + summary["error"]
    
    # Check for async test warnings
    if "PytestUnhandledCoroutineWarning" in output_text or "async def functions are not natively supported" in output_text:
        summary["warning"] = "Async tests detected but pytest-asyncio may not be properly configured. Install: pip install pytest-asyncio"
    
    # Check if all tests were skipped
    if summary["total"] > 0 and summary["passed"] == 0 and summary["failed"] == 0 and summary["skipped"] > 0:
        summary["warning"] = "All tests were skipped. Check test configuration, dependencies, and markers."
    
    return summary

@router.post("/run")
async def run_tests(request: TestRunRequest):
    """
    Run tests synchronously and return results.
    For real-time streaming, use /run/stream endpoint.
    """
    try:
        # Get current test configuration
        categories, files = _get_current_test_config()
        
        # Determine what to run
        test_files = []
        markers = []
        
        if request.category:
            if request.category not in categories:
                raise HTTPException(status_code=400, detail=f"Unknown category: {request.category}")
            category = categories[request.category]
            test_files = category["files"]
            markers = category.get("markers", [])
        elif request.file:
            if request.file not in files:
                raise HTTPException(status_code=400, detail=f"Unknown file: {request.file}")
            file_path = files[request.file]
            if request.test_path:
                test_files = [f"{file_path}::{request.test_path}"]
            else:
                test_files = [file_path]
        else:
            # Default to all tests
            if "all" not in categories:
                # Fallback: discover files if "all" category doesn't exist
                categories, files = _discover_test_files()
            test_files = categories.get("all", {}).get("files", [])
        
        # Re-check tests directory location dynamically
        current_tests_dir = _find_tests_directory()
        
        # Check if tests directory exists
        if not current_tests_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Tests directory not found at {current_tests_dir}. Please create the tests directory and add test files."
            )
        
        # Build pytest command - try to use the same Python that's running this script
        # This ensures we use the correct virtual environment
        import sys
        python_executable = sys.executable  # Use the same Python interpreter
        cmd = [python_executable, "-m", "pytest"]
        
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
        
        # Add pytest-asyncio mode if available (for async tests)
        # Check if pytest-asyncio is installed
        try:
            import importlib
            importlib.import_module("pytest_asyncio")
            # pytest-asyncio is available, add async mode
            cmd.extend(["-p", "asyncio", "--asyncio-mode=auto"])
            logger.debug("pytest-asyncio detected, enabling async test support")
        except ImportError:
            logger.debug("pytest-asyncio not available, async tests may be skipped")
        except Exception as e:
            logger.debug(f"Could not check for pytest-asyncio: {e}")
        
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
            
            # Parse pytest output to detect skipped tests and actual results
            # Pytest returns 0 even when tests are skipped, so we need to parse the output
            output_text = result.stdout + result.stderr
            test_summary = _parse_pytest_output(output_text)
            
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
            
            # Determine success: passed > 0 and failed == 0 (skipped is OK)
            # If all tests were skipped, that's a warning, not success
            actual_success = (
                test_summary["passed"] > 0 and 
                test_summary["failed"] == 0 and
                test_summary["error"] == 0
            )
            
            # If all tests were skipped, mark as partial success with warning
            if test_summary["passed"] == 0 and test_summary["skipped"] > 0 and test_summary["failed"] == 0:
                actual_success = False  # All skipped is not success
                test_summary["warning"] = "All tests were skipped. Check test configuration and dependencies."
            
            return {
                "success": actual_success,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd),
                "report": report_data,
                "test_summary": test_summary
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
            
            # Get current test configuration
            categories, files = _get_current_test_config()
            
            if request.category:
                if request.category not in categories:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Unknown category: {request.category}'})}\n\n"
                    return
                category = categories[request.category]
                test_files = category["files"]
                markers = category.get("markers", [])
            elif request.file:
                if request.file not in files:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Unknown file: {request.file}'})}\n\n"
                    return
                file_path = files[request.file]
                if request.test_path:
                    test_files = [f"{file_path}::{request.test_path}"]
                else:
                    test_files = [file_path]
            else:
                if "all" not in categories:
                    # Fallback: discover files if "all" category doesn't exist
                    categories, files = _discover_test_files()
                test_files = categories.get("all", {}).get("files", [])
            
            # Re-check tests directory location dynamically
            current_tests_dir = _find_tests_directory()
            
            # Check if tests directory exists
            if not current_tests_dir.exists():
                yield f"data: {json.dumps({'type': 'error', 'message': f'Tests directory not found at {current_tests_dir}. Please create the tests directory and add test files.'})}\n\n"
                return
            
            # Build pytest command - try to use the same Python that's running this script
            import sys
            python_executable = sys.executable  # Use the same Python interpreter
            cmd = [python_executable, "-m", "pytest"]
            
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
            
            # Add pytest-asyncio mode if available (for async tests)
            try:
                import importlib
                importlib.import_module("pytest_asyncio")
                cmd.extend(["-p", "asyncio", "--asyncio-mode=auto"])
            except (ImportError, Exception):
                pass  # pytest-asyncio not available, continue without it
            
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
                
                # Parse output to detect skipped tests
                # Note: We can't parse the full output here since it's streamed, but we can check the return code
                # The frontend will parse the output from the streamed lines
                # For now, we'll send the return code and let the frontend handle parsing
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
    try:
        # Get current test configuration safely
        try:
            categories_count = len(TEST_CATEGORIES) if TEST_CATEGORIES else 0
            files_count = len(TEST_FILES) if TEST_FILES else 0
            categories_list = list(TEST_CATEGORIES.keys()) if TEST_CATEGORIES else []
            files_list = list(TEST_FILES.keys()) if TEST_FILES else []
        except Exception as e:
            logger.warning(f"Error accessing TEST_CATEGORIES/TEST_FILES: {e}")
            categories_count = 0
            files_count = 0
            categories_list = []
            files_list = []
        
        # Return static data immediately, then try to enhance with filesystem checks
        # This ensures the frontend always gets useful data quickly
        static_response = {
            "project_root": str(PROJECT_ROOT),
            "tests_dir_exists": False,  # Will be updated by filesystem check
            "tests_dir": str(APP_DIR.parent / "tests"),  # Default path
            "test_file_count": files_count,
            "available_categories": categories_count,
            "available_files": files_count,
            "categories": categories_list,
            "files": files_list,
            "pytest_timeout_available": _check_pytest_timeout_cached(),
            "status": "ok"
        }
        
        # Try to enhance with filesystem checks, but don't wait long
        try:
            loop = asyncio.get_event_loop()
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, _get_test_status_internal),
                    timeout=0.5  # Very short timeout - just enhance if possible
                )
                # Merge filesystem results with static data
                static_response.update({
                    "tests_dir_exists": result.get("tests_dir_exists", False),
                    "tests_dir": result.get("tests_dir", static_response["tests_dir"]),
                    "test_file_count": result.get("test_file_count", static_response["test_file_count"]),
                    "available_categories": len(result.get("available_categories", [])),
                    "available_files": len(result.get("available_files", [])),
                })
                logger.debug(f"Test status check completed: tests_dir_exists={result.get('tests_dir_exists')}")
            except asyncio.TimeoutError:
                # Timeout is fine - we already have static data
                logger.debug("Test status filesystem check timed out, using static data")
            except Exception as e:
                logger.debug(f"Test status filesystem check failed: {e}, using static data")
        except Exception as e:
            logger.debug(f"Error in status enhancement: {e}, using static data")
        
        return static_response
    except Exception as e:
        logger.error(f"Error in get_test_status: {e}", exc_info=True)
        # Return minimal valid response on error
        return {
            "project_root": str(PROJECT_ROOT) if 'PROJECT_ROOT' in globals() else "unknown",
            "tests_dir_exists": False,
            "tests_dir": str(APP_DIR.parent / "tests") if 'APP_DIR' in globals() else "unknown",
            "test_file_count": 0,
            "available_categories": 0,
            "available_files": 0,
            "categories": [],
            "files": [],
            "pytest_timeout_available": False,
            "status": "error",
            "error": str(e)
        }

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

