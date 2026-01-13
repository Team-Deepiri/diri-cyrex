#!/bin/bash
# Configure MPS (Metal Performance Shaders) for macOS with Ollama, PyTorch, and Milvus
# This script sets up GPU acceleration for Apple Silicon Macs

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🍎 MPS (Metal Performance Shaders) Configuration for macOS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is for macOS only"
    echo "   For Linux/NVIDIA: use install-nvidia-container-toolkit.sh"
    exit 1
fi

echo "✅ macOS detected"
echo ""

# Check for Apple Silicon
CHIP_TYPE=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
if echo "$CHIP_TYPE" | grep -qi "Apple"; then
    echo "✅ Apple Silicon detected: $CHIP_TYPE"
    APPLE_SILICON=true
else
    echo "⚠️  Apple Silicon not detected: $CHIP_TYPE"
    echo "   MPS is only available on Apple Silicon (M1/M2/M3/M4)"
    echo "   Intel Macs will use CPU only"
    APPLE_SILICON=false
fi
echo ""

# Check Python
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo "✅ Python found: $PYTHON_VERSION"
else
    echo "❌ Python 3 not found"
    echo "   Install with: brew install python3"
    exit 1
fi

# Check pip
if command_exists pip3; then
    echo "✅ pip3 found"
else
    echo "❌ pip3 not found"
    echo "   Install with: python3 -m ensurepip --upgrade"
    exit 1
fi

echo ""

# Check PyTorch installation
echo "🔍 Checking PyTorch installation..."
PYTORCH_INSTALLED=false
MPS_AVAILABLE=false

if python3 -c "import torch" 2>/dev/null; then
    PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    echo "✅ PyTorch installed: $PYTORCH_VERSION"
    PYTORCH_INSTALLED=true
    
    # Check MPS availability
    if python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
        MPS_AVAILABLE=true
        echo "✅ MPS backend is available"
    else
        echo "⚠️  MPS backend not available"
        if [ "$APPLE_SILICON" = false ]; then
            echo "   (Expected - MPS requires Apple Silicon)"
        else
            echo "   (This may indicate PyTorch needs to be updated)"
        fi
    fi
else
    echo "⚠️  PyTorch not installed"
    echo "   Will install PyTorch with MPS support"
fi

echo ""

# Check Ollama
echo "🔍 Checking Ollama installation..."
OLLAMA_INSTALLED=false
if command_exists ollama; then
    OLLAMA_VERSION=$(ollama --version 2>/dev/null | head -n1 || echo "unknown")
    echo "✅ Ollama found: $OLLAMA_VERSION"
    OLLAMA_INSTALLED=true
else
    echo "⚠️  Ollama not found"
    echo "   Install with: brew install ollama"
    echo "   Or download from: https://ollama.ai"
fi

echo ""

# Check Milvus (for vector store)
echo "🔍 Checking Milvus setup..."
MILVUS_AVAILABLE=false
if command_exists docker; then
    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "milvus"; then
        echo "✅ Milvus container found"
        MILVUS_AVAILABLE=true
    else
        echo "⚠️  Milvus container not running"
        echo "   Milvus will be started via docker-compose"
    fi
else
    echo "⚠️  Docker not found"
    echo "   Milvus requires Docker"
fi

echo ""

# Configuration summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Configuration Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Apple Silicon: $([ "$APPLE_SILICON" = true ] && echo "✅ Yes" || echo "❌ No")"
echo "PyTorch: $([ "$PYTORCH_INSTALLED" = true ] && echo "✅ Installed ($PYTORCH_VERSION)" || echo "⚠️  Not installed")"
echo "MPS Available: $([ "$MPS_AVAILABLE" = true ] && echo "✅ Yes" || echo "❌ No")"
echo "Ollama: $([ "$OLLAMA_INSTALLED" = true ] && echo "✅ Installed" || echo "⚠️  Not installed")"
echo "Milvus: $([ "$MILVUS_AVAILABLE" = true ] && echo "✅ Available" || echo "⚠️  Not running")"
echo ""

# Installation/Configuration steps
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 Configuration Steps"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

NEEDS_INSTALL=false

# Install/Update PyTorch
if [ "$PYTORCH_INSTALLED" = false ] || [ "$MPS_AVAILABLE" = false ]; then
    echo "1️⃣  Installing/Updating PyTorch with MPS support..."
    echo ""
    
    if [ "$APPLE_SILICON" = true ]; then
        echo "   Installing PyTorch for Apple Silicon (MPS support)..."
        pip3 install --upgrade torch torchvision torchaudio
        echo "   ✅ PyTorch installed/updated"
        
        # Verify MPS
        if python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())" 2>/dev/null; then
            MPS_AVAILABLE=true
            echo "   ✅ MPS backend verified"
        else
            echo "   ⚠️  MPS still not available - may need system restart"
        fi
    else
        echo "   Installing PyTorch (CPU only - Intel Mac)..."
        pip3 install --upgrade torch torchvision torchaudio
        echo "   ✅ PyTorch installed (CPU version)"
    fi
    echo ""
    NEEDS_INSTALL=true
fi

# Install Ollama if needed
if [ "$OLLAMA_INSTALLED" = false ]; then
    echo "2️⃣  Installing Ollama..."
    echo ""
    
    if command_exists brew; then
        echo "   Installing via Homebrew..."
        brew install ollama
    else
        echo "   Homebrew not found. Please install Ollama manually:"
        echo "   1. Download from: https://ollama.ai"
        echo "   2. Or install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo "   3. Then run: brew install ollama"
    fi
    echo ""
    NEEDS_INSTALL=true
fi

# Configure environment variables
echo "3️⃣  Configuring environment variables..."
echo ""

ENV_FILE="$HOME/.deepiri-mps-env"
cat > "$ENV_FILE" << 'EOF'
# Deepiri MPS Configuration for macOS
# Source this file: source ~/.deepiri-mps-env

# PyTorch MPS device
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Ollama configuration
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_NUM_GPU=1  # Use GPU if available

# Milvus configuration (if running locally)
export MILVUS_HOST=localhost
export MILVUS_PORT=19530

# Python path for embeddings
export SENTENCE_TRANSFORMERS_HOME=~/.cache/sentence_transformers
export HF_HOME=~/.cache/huggingface
EOF

echo "   ✅ Environment file created: $ENV_FILE"
echo ""

# Add to shell profile
SHELL_PROFILE=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_PROFILE="$HOME/.zshrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_PROFILE="$HOME/.bash_profile"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_PROFILE="$HOME/.bashrc"
fi

if [ -n "$SHELL_PROFILE" ]; then
    if ! grep -q "deepiri-mps-env" "$SHELL_PROFILE" 2>/dev/null; then
        echo "" >> "$SHELL_PROFILE"
        echo "# Deepiri MPS Configuration" >> "$SHELL_PROFILE"
        echo "source ~/.deepiri-mps-env 2>/dev/null || true" >> "$SHELL_PROFILE"
        echo "   ✅ Added to $SHELL_PROFILE"
    else
        echo "   ℹ️  Already configured in $SHELL_PROFILE"
    fi
else
    echo "   ⚠️  Shell profile not found - manually source the env file:"
    echo "      source $ENV_FILE"
fi

echo ""

# Test PyTorch MPS
echo "4️⃣  Testing PyTorch MPS..."
echo ""

if [ "$APPLE_SILICON" = true ]; then
    source "$ENV_FILE" 2>/dev/null || true
    
    python3 << 'PYTHON_TEST'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")

if torch.backends.mps.is_available():
    print("Testing MPS device...")
    try:
        x = torch.randn(3, 3, device='mps')
        y = torch.randn(3, 3, device='mps')
        z = x @ y
        print("✅ MPS test successful!")
        print(f"   Device: {z.device}")
        print(f"   Result shape: {z.shape}")
    except Exception as e:
        print(f"❌ MPS test failed: {e}")
        sys.exit(1)
else:
    print("⚠️  MPS not available - will use CPU")
    print("   This is normal for Intel Macs")
PYTHON_TEST

    if [ $? -eq 0 ]; then
        echo "   ✅ PyTorch MPS test passed"
    else
        echo "   ⚠️  PyTorch MPS test had issues"
    fi
else
    echo "   ℹ️  Skipping MPS test (Intel Mac - CPU only)"
fi

echo ""

# Test Ollama
echo "5️⃣  Testing Ollama..."
echo ""

if [ "$OLLAMA_INSTALLED" = true ]; then
    if pgrep -x "ollama" > /dev/null; then
        echo "   ✅ Ollama is running"
        
        # Test Ollama API
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "   ✅ Ollama API is accessible"
        else
            echo "   ⚠️  Ollama API not responding - may need to start Ollama"
            echo "      Run: ollama serve"
        fi
    else
        echo "   ⚠️  Ollama is not running"
        echo "      Start with: ollama serve"
        echo "      Or it will start automatically when you use it"
    fi
else
    echo "   ⚠️  Ollama not installed - skipping test"
fi

echo ""

# Docker Compose configuration note
echo "6️⃣  Docker Compose Configuration"
echo ""

if [ -f "docker-compose.dev.yml" ] || [ -f "../docker-compose.dev.yml" ]; then
    echo "   ✅ docker-compose.dev.yml found"
    echo ""
    echo "   For macOS, ensure your docker-compose.dev.yml has:"
    echo "   - Ollama service configured for localhost:11434"
    echo "   - Milvus service (if using Docker)"
    echo ""
    echo "   Note: Docker containers on macOS typically can't access GPU"
    echo "   For best performance, run Ollama natively (not in Docker)"
else
    echo "   ⚠️  docker-compose.dev.yml not found in current directory"
fi

echo ""

# Final instructions
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Configuration Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$NEEDS_INSTALL" = true ]; then
    echo "📝 Next Steps:"
    echo ""
    echo "1. Restart your terminal or source the environment:"
    echo "   source ~/.deepiri-mps-env"
    echo ""
    echo "2. Start Ollama (if not running):"
    echo "   ollama serve"
    echo ""
    echo "3. Pull a model to test:"
    echo "   ollama pull llama3:8b"
    echo ""
    echo "4. Test PyTorch MPS in Python:"
    echo "   python3 -c \"import torch; print('MPS:', torch.backends.mps.is_available())\""
    echo ""
else
    echo "✅ Everything is already configured!"
    echo ""
    echo "To use MPS in your Python code:"
    echo "  import torch"
    echo "  device = 'mps' if torch.backends.mps.is_available() else 'cpu'"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

