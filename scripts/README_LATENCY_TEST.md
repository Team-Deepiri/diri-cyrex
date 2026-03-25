# Latency Test Script

## Overview

The unified latency test script (`test_latency.py`) consolidates all latency testing functionality into a single CLI tool.

## Usage

### Basic Usage

```bash
# Run comprehensive test suite (default)
python scripts/test_latency.py

# Or explicitly
python scripts/test_latency.py --test comprehensive
```

### Test Specific Components

```bash
# Test Ollama directly
python scripts/test_latency.py --test ollama --prompt "Hello" --iterations 5

# Test orchestrator (no tools, no RAG)
python scripts/test_latency.py --test orchestrator --prompt "Hi"

# Test orchestrator with tools
python scripts/test_latency.py --test orchestrator --use-tools --prompt "Hello"

# Test orchestrator with RAG
python scripts/test_latency.py --test orchestrator --use-tools --use-rag --prompt "What is AI?"
```

### Custom Configuration

```bash
# Custom URLs
python scripts/test_latency.py \
  --ollama-url http://localhost:11434 \
  --orchestrator-url http://localhost:8000

# Custom model
python scripts/test_latency.py --test ollama --model llama3:8b-q4_0

# More iterations for better statistics
python scripts/test_latency.py --test comprehensive --iterations 10
```

## Command Line Options

```
--test {ollama,orchestrator,comprehensive,all}
    Test type to run (default: comprehensive)

--prompt TEXT
    Test prompt (default: "Hello, how are you?")

--iterations INTEGER
    Number of iterations per test (default: 3)

--ollama-url TEXT
    Ollama API URL (default: http://localhost:11434)

--orchestrator-url TEXT
    Orchestrator API URL (default: http://localhost:8000)

--use-tools
    Use tools in orchestrator test (default: True)

--use-rag
    Use RAG in orchestrator test (default: False)

--model TEXT
    Ollama model to test (default: llama3:8b)
```

## Output

The script provides:
- Individual test results with min/max/average
- Summary statistics
- Overhead analysis (orchestrator vs direct Ollama)
- Performance recommendations

## Documentation

For detailed latency analysis and optimization strategies, see:
- `docs/operations/LATENCY_ANALYSIS_AND_OPTIMIZATION.md`

