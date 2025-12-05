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
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
TESTS_DIR = PROJECT_ROOT / "tests"

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
    """List all available test categories and files"""
    return TestListResponse(
        categories=TEST_CATEGORIES,
        files=TEST_FILES
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
        
        # Build pytest command
        cmd = ["python3", "-m", "pytest"]
        
        # Add test files
        resolved_files = []
        for file_path in test_files:
            if file_path == "tests/" or file_path == "tests":
                if TESTS_DIR.exists():
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
        
        if not resolved_files:
            if TESTS_DIR.exists():
                resolved_files = ["tests/"]
            else:
                raise HTTPException(status_code=404, detail="No test files found")
        
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
        
        # Timeout
        if request.timeout and request.timeout > 0:
            cmd.extend(["--timeout", str(request.timeout), "--timeout-method", "thread"])
        
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
            
            # Build pytest command
            cmd = ["python3", "-m", "pytest"]
            
            # Add test files
            resolved_files = []
            for file_path in test_files:
                if file_path == "tests/" or file_path == "tests":
                    if TESTS_DIR.exists():
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
            
            if not resolved_files:
                if TESTS_DIR.exists():
                    resolved_files = ["tests/"]
                else:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'No test files found'})}\n\n"
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
            
            if request.timeout and request.timeout > 0:
                cmd.extend(["--timeout", str(request.timeout), "--timeout-method", "thread"])
            
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


@router.get("/status")
async def get_test_status():
    """Get status of test infrastructure"""
    return {
        "project_root": str(PROJECT_ROOT),
        "tests_dir_exists": TESTS_DIR.exists(),
        "tests_dir": str(TESTS_DIR),
        "available_categories": list(TEST_CATEGORIES.keys()),
        "available_files": list(TEST_FILES.keys())
    }

