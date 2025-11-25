#!/bin/bash
# Setup script for local LLM (Ollama) - Recommended for cost savings
# This script installs Ollama and pulls the recommended model (llama3:8b)

set -e

echo "🚀 Setting up local LLM for Cyrex..."
echo ""

# Detect WSL (Windows Subsystem for Linux)
IS_WSL=false
if [[ -f /proc/version ]] && grep -qi microsoft /proc/version; then
    IS_WSL=true
    echo "🔍 WSL (Windows Subsystem for Linux) detected"
    echo ""
fi

# Check if Ollama is already installed
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is already installed"
    ollama --version
else
    echo "📦 Installing Ollama..."
    
    # Detect OS and install Ollama
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$IS_WSL" == true ]]; then
        # Linux or WSL
        if [[ "$IS_WSL" == true ]]; then
            echo "📝 Installing Ollama in WSL environment..."
            echo "   Note: Ollama will run in WSL, accessible from Windows at http://localhost:11434"
        fi
        
        # Check if curl is available
        if ! command -v curl &> /dev/null; then
            echo "❌ curl not found. Installing curl..."
            if command -v apt-get &> /dev/null; then
                sudo apt-get update && sudo apt-get install -y curl
            elif command -v yum &> /dev/null; then
                sudo yum install -y curl
            else
                echo "❌ Cannot install curl automatically. Please install curl manually and rerun this script."
                exit 1
            fi
        fi
        
        # Install Ollama
        curl -fsSL https://ollama.com/install.sh | sh
        
        if [[ "$IS_WSL" == true ]]; then
            echo ""
            echo "💡 WSL Installation Notes:"
            echo "   - Ollama service will start automatically"
            echo "   - Access from Windows: http://localhost:11434"
            echo "   - Access from WSL: http://localhost:11434"
            echo "   - To start Ollama manually: ollama serve"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo "❌ Homebrew not found. Please install Ollama manually from https://ollama.com/download"
            exit 1
        fi
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        # Windows (native, not WSL)
        echo "❌ Native Windows detected (not WSL)."
        echo "   Options:"
        echo "   1. Run this script in WSL (recommended)"
        echo "   2. Download and install Ollama for Windows from https://ollama.com/download"
        echo "      After installation, run: ollama pull llama3:8b"
        exit 1
    else
        echo "❌ Unsupported OS. Please install Ollama manually from https://ollama.com/download"
        exit 1
    fi
fi

echo ""
echo "🔄 Starting Ollama service..."
# Start Ollama in background if not running
if ! pgrep -x "ollama" > /dev/null; then
    if [[ "$IS_WSL" == true ]]; then
        echo "   Starting Ollama in WSL..."
        # In WSL, we might need to start it differently
        nohup ollama serve > /dev/null 2>&1 &
        sleep 5
        echo "✅ Ollama service started in WSL"
        echo "   Access from Windows: http://localhost:11434"
    else
        ollama serve &
        sleep 3
        echo "✅ Ollama service started"
    fi
else
    echo "✅ Ollama service is already running"
fi

echo ""
echo "📥 Pulling recommended model: llama3:8b"
echo "   This may take a few minutes depending on your internet connection...(4GB download)"
ollama pull llama3:8b

echo ""
echo "✅ Model downloaded successfully!"
echo ""
echo "🔍 Verifying installation..."
ollama list

echo ""
echo "✅ Local LLM setup complete!"
echo ""

if [[ "$IS_WSL" == true ]]; then
    echo "📝 Next steps (WSL):"
    echo "   1. Ensure Ollama is running: ollama serve"
    echo "   2. Configure your .env file in WSL:"
    echo "      LOCAL_LLM_BACKEND=ollama"
    echo "      LOCAL_LLM_MODEL=llama3:8b"
    echo "      OLLAMA_BASE_URL=http://localhost:11434"
    echo "   3. Start Cyrex service (from WSL or Docker)"
    echo ""
    echo "💡 WSL Tips:"
    echo "   - Ollama runs in WSL but is accessible from Windows at http://localhost:11434"
    echo "   - If Cyrex runs in Docker, use 'host.docker.internal:11434' or 'localhost:11434'"
    echo "   - To check if Ollama is running: curl http://localhost:11434/api/tags"
else
    echo "📝 Next steps:"
    echo "   1. Ensure Ollama is running: ollama serve"
    echo "   2. Configure your .env file:"
    echo "      LOCAL_LLM_BACKEND=ollama"
    echo "      LOCAL_LLM_MODEL=llama3:8b"
    echo "      OLLAMA_BASE_URL=http://localhost:11434"
    echo "   3. Start Cyrex service"
fi

echo ""
echo "💡 Tip: The system will automatically use OpenAI if available, otherwise fall back to local LLM"

