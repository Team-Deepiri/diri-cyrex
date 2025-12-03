#!/usr/bin/env python3
"""
Master Test Runner for diri-cyrex
Interactive CLI to run tests with selection options

Usage:
    python run_tests.py                    # Interactive mode
    python run_tests.py --category all     # Run all tests
    python run_tests.py --file orchestrator # Run specific file
    python run_tests.py --list             # List available tests
"""
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional
import json


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
        exists = Path(file_path).exists()
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
            if Path(file_path).exists():
                test_files.append(file_path)
            else:
                print_warning(f"File not found: {file_path}")
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
    if not Path(file_path).exists():
        print_error(f"File not found: {file_path}")
        return None, []
    
    test_path = input(f"{Colors.OKCYAN}Enter test path (e.g., TestClass::test_method or TestClass): {Colors.ENDC}").strip()
    
    if test_path:
        full_path = f"{file_path}::{test_path}"
    else:
        full_path = file_path
    
    return [full_path], []


def run_tests(test_files: List[str], markers: List[str] = None, verbose: bool = False, 
              coverage: bool = False, slow: bool = True, output_format: str = "standard"):
    """Run pytest with given parameters"""
    cmd = ["python", "-m", "pytest"]
    
    # Add test files
    if test_files:
        cmd.extend(test_files)
    else:
        cmd.append("tests/")
    
    # Add markers
    if markers:
        cmd.extend(markers)
    
    # Skip slow tests if requested
    if not slow:
        cmd.extend(["-m", "not slow"])
    
    # Verbose output
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")  # Quiet mode
    
    # Coverage
    if coverage:
        cmd.extend(["--cov=app", "--cov-report=term-missing", "--cov-report=html"])
    
    # Output format
    if output_format == "json":
        cmd.extend(["--json-report", "--json-report-file=test-report.json"])
    
    # Color output
    cmd.append("--color=yes")
    
    print_header("Running Tests")
    print_info(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print_warning("\nTests interrupted by user")
        return 130
    except Exception as e:
        print_error(f"Error running tests: {e}")
        return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Master Test Runner for diri-cyrex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python run_tests.py

  # Run specific category
  python run_tests.py --category orchestrator

  # Run specific file
  python run_tests.py --file orchestrator

  # Run with coverage
  python run_tests.py --category all --coverage

  # Run specific test
  python run_tests.py --file orchestrator --test "TestOrchestratorInitialization::test_orchestrator_creation"

  # Skip slow tests
  python run_tests.py --category all --no-slow
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
    
    # Filter out non-existent files
    existing_files = [f for f in test_files if Path(f).exists() or "::" in f]
    if not existing_files and test_files:
        print_error("No test files found")
        return 1
    
    # Run tests
    return run_tests(
        existing_files,
        markers=markers,
        verbose=args.verbose,
        coverage=args.coverage,
        slow=not args.no_slow,
        output_format=args.format
    )


if __name__ == "__main__":
    sys.exit(main())

