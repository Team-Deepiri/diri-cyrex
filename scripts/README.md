# Scripts Directory

Utility scripts for diri-cyrex service, organized by category.

## 📁 Directory Structure

```
scripts/
├── gpu/          # GPU configuration and setup scripts
├── llm/          # LLM/Ollama management scripts
├── dev/          # Development tools and utilities
└── docs/         # Documentation and guides
```

## 🚀 Quick Start

### GPU Setup
- **NVIDIA (Linux/WSL)**: `gpu/install-nvidia-container-toolkit.sh`
- **Apple Silicon (macOS - MPS)**: `gpu/configure-mps-macos.sh`

### LLM Setup
- **Install Ollama**: `llm/install-local-llm.sh`
- **Check Models**: `llm/check-ollama-models.sh`
- **Test Connection**: `llm/test-ollama-connection.sh`

### Development
- **Run Tests**: `dev/run_tests.py`
- **Install Git Hooks**: `dev/install-git-hooks.sh`

## 📚 Documentation

See `docs/README.md` for detailed documentation and guides.

## 🔧 Categories

### `gpu/` - GPU Configuration
Scripts for configuring GPU acceleration:
- NVIDIA Container Toolkit installation (Linux/WSL)
- MPS (Metal Performance Shaders) setup (macOS)
- GPU detection and testing

### `llm/` - LLM Management
Scripts for managing local LLMs (Ollama, etc.):
- Ollama installation and setup
- Model management and checking
- Connection testing
- Docker integration

### `dev/` - Development Tools
Development and testing utilities:
- Test runners
- Git hooks installation
- Integration checks

### `docs/` - Documentation
Guides and documentation:
- Setup guides
- Troubleshooting
- Quick reference

## 💡 Usage

All scripts should be run from the project root directory:

```bash
# Example: Install NVIDIA Container Toolkit
cd deepiri-platform/diri-cyrex
bash scripts/gpu/install-nvidia-container-toolkit.sh

# Example: Check Ollama models
bash scripts/llm/check-ollama-models.sh

# Example: Run tests
python3 scripts/dev/run_tests.py
```

## 📝 Notes

- **Windows Users**: Most scripts are bash-based. Use WSL2 or Git Bash.
- **macOS Users**: Use the MPS configuration script for Apple Silicon GPU support.
- **Linux Users**: Use NVIDIA Container Toolkit script for GPU support in Docker.

For detailed information about each script, see the README in each subdirectory.
