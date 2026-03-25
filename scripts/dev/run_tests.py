#!/usr/bin/env python3
"""
Master Test Runner for diri-cyrex
Interactive CLI to run tests with selection options

Usage:
    python3 scripts/run_tests.py                    # Interactive mode
    python3 scripts/run_tests.py --category all     # Run all tests
    python3 scripts/run_tests.py --file orchestrator # Run specific file
    python3 scripts/run_tests.py --list             # List available tests
"""
import sys
import subprocess
import argparse
import os
import time
import threading
import queue
from pathlib import Path
from typing import List, Optional
import json

# Get project root directory (parent of scripts directory)
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.absolute()

# Ensure we can find the tests directory
TESTS_DIR = PROJECT_ROOT / "tests"


# Test categories and their files
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


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def list_categories():
    """List all available test categories"""
    print_header("Available Test Categories")
    for key, category in TEST_CATEGORIES.items():
        print(f"{Colors.BOLD}{key:15}{Colors.ENDC} - {category['name']}")
        print(f"{' '*17}{category['description']}")
        print(f"{' '*17}Files: {len(category['files'])}")
        print()


def list_files():
    """List all available test files"""
    print_header("Available Test Files")
    for key, file_path in TEST_FILES.items():
        # Check if file exists relative to project root
        resolved_path = Path(PROJECT_ROOT / file_path).resolve()
        exists = resolved_path.exists()
        status = f"{Colors.OKGREEN}✓{Colors.ENDC}" if exists else f"{Colors.FAIL}✗{Colors.ENDC}"
        print(f"{status} {Colors.BOLD}{key:15}{Colors.ENDC} - {file_path}")


def interactive_select():
    """Interactive test selection"""
    print_header("Interactive Test Selection")
    
    # Show categories
    list_categories()
    
    print(f"{Colors.BOLD}Selection Options:{Colors.ENDC}")
    print("1. Select by category (orchestrator, tools, ollama, api, integration, all)")
    print("2. Select by file name")
    print("3. Select specific test class or function")
    print("4. Run all tests")
    print("5. Exit")
    
    choice = input(f"\n{Colors.OKCYAN}Enter your choice (1-5): {Colors.ENDC}").strip()
    
    if choice == "1":
        return select_by_category()
    elif choice == "2":
        return select_by_file()
    elif choice == "3":
        return select_specific_test()
    elif choice == "4":
        return ["tests/"]
    elif choice == "5":
        print_info("Exiting...")
        sys.exit(0)
    else:
        print_error("Invalid choice")
        return None


def select_by_category():
    """Select tests by category"""
    print(f"\n{Colors.BOLD}Available Categories:{Colors.ENDC}")
    for key in TEST_CATEGORIES.keys():
        print(f"  - {key}")
    
    selection = input(f"\n{Colors.OKCYAN}Enter category (comma-separated for multiple): {Colors.ENDC}").strip()
    categories = [c.strip() for c in selection.split(",")]
    
    test_files = []
    markers = []
    
    for cat in categories:
        if cat in TEST_CATEGORIES:
            test_files.extend(TEST_CATEGORIES[cat]["files"])
            if TEST_CATEGORIES[cat]["markers"]:
                markers.extend(TEST_CATEGORIES[cat]["markers"])
        else:
            print_warning(f"Unknown category: {cat}")
    
    return test_files, markers


def select_by_file():
    """Select tests by file"""
    list_files()
    
    selection = input(f"\n{Colors.OKCYAN}Enter file key(s) (comma-separated): {Colors.ENDC}").strip()
    file_keys = [f.strip() for f in selection.split(",")]
    
    test_files = []
    for key in file_keys:
        if key in TEST_FILES:
            file_path = TEST_FILES[key]
            # Check relative to project root
            resolved_path = Path(PROJECT_ROOT / file_path).resolve()
            if resolved_path.exists():
                test_files.append(file_path)
            else:
                print_warning(f"File not found: {file_path} (checked: {resolved_path})")
        else:
            print_warning(f"Unknown file key: {key}")
    
    return test_files, []


def select_specific_test():
    """Select specific test class or function"""
    list_files()
    
    file_key = input(f"\n{Colors.OKCYAN}Enter file key: {Colors.ENDC}").strip()
    if file_key not in TEST_FILES:
        print_error(f"Unknown file key: {file_key}")
        return None, []
    
    file_path = TEST_FILES[file_key]
    # Check relative to project root
    resolved_path = Path(PROJECT_ROOT / file_path).resolve()
    if not resolved_path.exists():
        print_error(f"File not found: {file_path} (checked: {resolved_path})")
        return None, []
    
    test_path = input(f"{Colors.OKCYAN}Enter test path (e.g., TestClass::test_method or TestClass): {Colors.ENDC}").strip()
    
    if test_path:
        full_path = f"{file_path}::{test_path}"
    else:
        full_path = file_path
    
    return [full_path], []


def run_tests(test_files: List[str], markers: List[str] = None, verbose: bool = False, 
              coverage: bool = False, slow: bool = True, output_format: str = "standard",
              timeout: Optional[int] = None):
    """Run pytest with given parameters"""
    # Change to project root directory
    original_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    
    try:
        cmd = ["python3", "-m", "pytest"]
        
        # Add test files - resolve paths relative to project root
        if test_files:
            resolved_files = []
            for file_path in test_files:
                # Special case: "tests/" directory
                if file_path == "tests/" or file_path == "tests":
                    if TESTS_DIR.exists():
                        resolved_files.append("tests/")
                        continue
                    else:
                        print_warning(f"Tests directory not found: {TESTS_DIR}")
                        continue
                
                # If it's a full path with ::, split it
                if "::" in file_path:
                    parts = file_path.split("::", 1)
                    base_path = parts[0]
                    test_path = parts[1]
                    # Resolve base path relative to project root
                    resolved_base = Path(PROJECT_ROOT / base_path).resolve()
                    if resolved_base.exists():
                        resolved_files.append(f"{base_path}::{test_path}")
                    else:
                        print_warning(f"Test file not found: {base_path} (checked: {resolved_base})")
                else:
                    # Resolve relative to project root
                    resolved_path = Path(PROJECT_ROOT / file_path).resolve()
                    if resolved_path.exists():
                        # Use relative path for pytest (works better)
                        resolved_files.append(file_path)
                    else:
                        # Try as absolute path
                        abs_path = Path(file_path).resolve()
                        if abs_path.exists():
                            # Convert to relative path from project root
                            try:
                                rel_path = abs_path.relative_to(PROJECT_ROOT)
                                resolved_files.append(str(rel_path))
                            except ValueError:
                                # Path is outside project root, use as-is
                                resolved_files.append(str(abs_path))
                        else:
                            print_warning(f"Test file not found: {file_path} (checked: {resolved_path})")
            
            if resolved_files:
                cmd.extend(resolved_files)
            else:
                # Fallback to tests directory
                if TESTS_DIR.exists():
                    cmd.append("tests/")
                else:
                    print_error(f"Tests directory not found: {TESTS_DIR}")
                    os.chdir(original_cwd)
                    return 1
        else:
            # No files specified, use tests directory
            if TESTS_DIR.exists():
                cmd.append("tests/")
            else:
                print_error(f"Tests directory not found: {TESTS_DIR}")
                os.chdir(original_cwd)
                return 1
        
        # Add markers
        if markers:
            cmd.extend(markers)
        
        # Skip slow tests if requested
        if not slow:
            cmd.extend(["-m", "not slow"])
        
        # Verbose output - always show test names and progress
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-v")  # Always verbose so user can see what's happening
            # Remove -q flag - we want to see test progress
        
        # Coverage
        if coverage:
            cmd.extend(["--cov=app", "--cov-report=term-missing", "--cov-report=html"])
        
        # Output format
        if output_format == "json":
            cmd.extend(["--json-report", "--json-report-file=test-report.json"])
        
        # Color output
        cmd.append("--color=yes")
        
        # Show test progress in real-time (no buffering)
        cmd.append("-s")  # Don't capture output (show print statements)
        cmd.append("--tb=short")  # Shorter tracebacks
        
        # Add timeout if specified (requires pytest-timeout plugin)
        # Use pytest-timeout for per-test timeouts
        if timeout and timeout > 0:
            cmd.extend(["--timeout", str(timeout), "--timeout-method", "thread"])
        
        print_header("Running Tests")
        print_info(f"Command: {' '.join(cmd)}")
        print_info(f"Working directory: {PROJECT_ROOT}")
        if timeout and timeout > 0:
            print_info(f"Timeout: {timeout} seconds per test")
        print()
        
        # Run with timeout at subprocess level as well
        # Overall timeout is 10x per-test timeout, or 30 minutes if no per-test timeout
        overall_timeout = (timeout * 10) if (timeout and timeout > 0) else 1800
        try:
            # Stream output in real-time so user can see progress
            process = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,  # Line buffered
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            
            # Stream output line by line in a separate thread
            output_queue = queue.Queue()
            def read_output():
                try:
                    for line in process.stdout:
                        output_queue.put(line)
                except:
                    pass
                output_queue.put(None)  # Sentinel
            
            reader_thread = threading.Thread(target=read_output, daemon=True)
            reader_thread.start()
            
            # Print output as it comes
            start_time = time.time()
            while True:
                try:
                    line = output_queue.get(timeout=0.1)
                    if line is None:
                        break
                    print(line, end='', flush=True)
                except queue.Empty:
                    # Check if process is still running
                    if process.poll() is not None:
                        # Process finished, drain remaining output
                        for line in process.stdout:
                            print(line, end='', flush=True)
                        break
                    # Check overall timeout
                    if time.time() - start_time > overall_timeout:
                        raise subprocess.TimeoutExpired(cmd, overall_timeout)
            
            reader_thread.join(timeout=1)
            return process.returncode
            
        except subprocess.TimeoutExpired:
            if 'process' in locals():
                process.kill()
                process.wait()
            print_error(f"\nTests exceeded overall timeout of {overall_timeout} seconds")
            print_warning("Some tests may be hanging. Try running with --timeout 10 for faster feedback.")
            return 124  # Standard timeout exit code
        
    except KeyboardInterrupt:
        print_warning("\nTests interrupted by user")
        return 130
    except Exception as e:
        print_error(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Restore original directory
        os.chdir(original_cwd)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Master Test Runner for diri-cyrex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python3 scripts/run_tests.py

  # Run specific category
  python3 scripts/run_tests.py --category orchestrator

  # Run specific file
  python3 scripts/run_tests.py --file orchestrator

  # Run with coverage
  python3 scripts/run_tests.py --category all --coverage

  # Run specific test
  python3 scripts/run_tests.py --file orchestrator --test "TestOrchestratorInitialization::test_orchestrator_creation"

  # Skip slow tests
  python3 scripts/run_tests.py --category all --no-slow
        """
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive mode (default if no other options specified)"
    )
    
    parser.add_argument(
        "-c", "--category",
        choices=list(TEST_CATEGORIES.keys()),
        help="Run tests by category"
    )
    
    parser.add_argument(
        "-f", "--file",
        choices=list(TEST_FILES.keys()),
        help="Run tests from specific file"
    )
    
    parser.add_argument(
        "-t", "--test",
        help="Run specific test (e.g., TestClass::test_method)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage report"
    )
    
    parser.add_argument(
        "--no-slow",
        action="store_true",
        help="Skip slow tests"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available categories and files"
    )
    
    parser.add_argument(
        "--format",
        choices=["standard", "json"],
        default="standard",
        help="Output format"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,  # 15 seconds default per test (faster feedback)
        help="Timeout per test in seconds (default: 15, use 0 to disable)"
    )
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        list_categories()
        print()
        list_files()
        return 0
    
    # Determine what to run
    test_files = []
    markers = []
    
    if args.category:
        # Run by category
        category = TEST_CATEGORIES[args.category]
        test_files = category["files"]
        markers = category.get("markers", [])
    elif args.file:
        # Run by file
        if args.test:
            # Specific test in file
            test_files = [f"{TEST_FILES[args.file]}::{args.test}"]
        else:
            # Entire file
            test_files = [TEST_FILES[args.file]]
    elif args.test:
        # Specific test (need to find which file)
        print_warning("Use --file with --test to specify test location")
        return 1
    elif args.interactive or (not args.category and not args.file):
        # Interactive mode
        result = interactive_select()
        if result is None:
            return 1
        if isinstance(result, tuple):
            test_files, markers = result
        else:
            test_files = result
            markers = []
    
    # Filter out non-existent files - resolve paths relative to project root
    existing_files = []
    for file_path in test_files:
        # Special case: "tests/" directory
        if file_path == "tests/" or file_path == "tests":
            if TESTS_DIR.exists():
                existing_files.append("tests/")
                continue
        
        # Check if it's a test path with ::
        if "::" in file_path:
            # Split to check base file
            base_path = file_path.split("::")[0]
            resolved_path = Path(PROJECT_ROOT / base_path).resolve()
            if resolved_path.exists():
                existing_files.append(file_path)
        else:
            # Resolve relative to project root
            resolved_path = Path(PROJECT_ROOT / file_path).resolve()
            if resolved_path.exists():
                existing_files.append(file_path)
            # Also try as-is in case it's already absolute
            elif Path(file_path).exists():
                existing_files.append(file_path)
    
    # If no files specified, default to tests directory
    if not existing_files:
        if test_files:
            # Files were specified but none found
            print_error("No test files found")
            print_info(f"Project root: {PROJECT_ROOT}")
            print_info(f"Tests directory exists: {TESTS_DIR.exists()}")
            print_info(f"Searched for: {test_files}")
            # Show what files do exist
            if TESTS_DIR.exists():
                print_info(f"Available test files in {TESTS_DIR}:")
                for test_file in TESTS_DIR.glob("test_*.py"):
                    print_info(f"  - {test_file.name}")
            return 1
        else:
            # No files specified, use tests directory
            if TESTS_DIR.exists():
                existing_files = ["tests/"]
            else:
                print_error(f"Tests directory not found: {TESTS_DIR}")
                return 1
    
    # Run tests
    return run_tests(
        existing_files,
        markers=markers,
        verbose=args.verbose,
        coverage=args.coverage,
        slow=not args.no_slow,
        output_format=args.format,
        timeout=args.timeout
    )


if __name__ == "__main__":
    sys.exit(main())

