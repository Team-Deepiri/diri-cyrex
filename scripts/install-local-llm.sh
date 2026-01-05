#!/bin/bash
# Install and setup local LLM (Ollama) as Docker container
# Combines setup and model checking functionality
# Works on Windows (WSL), Linux, and macOS

set -e

CONTAINER_NAME="deepiri-ollama-dev"
OLLAMA_PORT=11434

echo "🚀 Installing Local LLM (Ollama) for Cyrex"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect operating system
OS_TYPE="unknown"
IS_WSL=false
IS_WINDOWS=false

if [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS_TYPE="linux"
    # Check if running in WSL
    if grep -qEi "(Microsoft|WSL)" /proc/version 2>/dev/null || [ -n "$WSL_DISTRO_NAME" ]; then
        IS_WSL=true
        echo "🪟 Detected Windows Subsystem for Linux (WSL)"
    fi
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ -n "$OS" ]] && [[ "$OS" == "Windows_NT" ]]; then
    IS_WINDOWS=true
    OS_TYPE="windows"
    echo "🪟 Detected Windows"
fi

# Detection flags
HAS_NVIDIA_GPU=false
HAS_APPLE_SILICON=false
HAS_CPU_ONLY=false
DRIVERS_INSTALLED=false

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 GPU Detection and Driver Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Path 1: Check for Mac/Apple Silicon
if [ "$OS_TYPE" = "macos" ]; then
    echo "🍎 Detected macOS - Checking for Apple Silicon..."
    echo ""
    
    # Check for Apple Silicon (M1/M2/M3/M4)
    if sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -qi "Apple"; then
        CHIP_TYPE=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
        echo "✅ Apple Silicon detected: $CHIP_TYPE"
        echo ""
        HAS_APPLE_SILICON=true
        
        # Check if Docker Desktop is installed
        if command_exists docker; then
            echo "✅ Docker found"
            
            # Check Docker platform
            DOCKER_PLATFORM=$(docker version --format '{{.Server.Arch}}' 2>/dev/null || echo "unknown")
            echo "   Docker platform: $DOCKER_PLATFORM"
            
            # Check if Docker Desktop is running
            if docker info >/dev/null 2>&1; then
                echo "✅ Docker is running"
                
                # Check if Docker Desktop has Apple Silicon support enabled
                if [ "$DOCKER_PLATFORM" = "arm64" ] || [ "$DOCKER_PLATFORM" = "aarch64" ]; then
                    echo "✅ Docker is configured for Apple Silicon (arm64)"
                else
                    echo "⚠️  Docker may not be using Apple Silicon architecture"
                    echo "   Please ensure Docker Desktop is set to use Apple Silicon (arm64)"
                fi
            else
                echo "❌ Docker is not running"
                echo ""
                echo "Would you like to start Docker Desktop? (y/n)"
                read -r start_docker
                if [[ "$start_docker" =~ ^[Yy]$ ]]; then
                    echo ""
                    echo "🚀 Starting Docker Desktop..."
                    open -a Docker
                    echo "   Waiting for Docker to start (this may take a moment)..."
                    sleep 5
                    
                    # Wait for Docker to be ready
                    for i in {1..30}; do
                        if docker info >/dev/null 2>&1; then
                            echo "✅ Docker is now running"
                            break
                        fi
                        sleep 2
                    done
                else
                    echo "⚠️  Docker must be running to use Ollama. Please start Docker Desktop manually."
                    exit 1
                fi
            fi
        else
            echo "❌ Docker not found"
            echo ""
            echo "⚠️  Docker Desktop is required for Ollama on macOS"
            echo ""
            echo "Would you like to install Docker Desktop? (y/n)"
            read -r install_docker
            
            if [[ "$install_docker" =~ ^[Yy]$ ]]; then
                echo ""
                echo "📦 Installing Docker Desktop for Mac..."
                echo ""
                echo "Please download and install Docker Desktop from:"
                echo "   https://www.docker.com/products/docker-desktop/"
                echo ""
                echo "Or install via Homebrew:"
                echo "   brew install --cask docker"
                echo ""
                echo "After installation, please run this script again."
                exit 0
            else
                echo "⚠️  Docker Desktop is required. Exiting."
                exit 1
            fi
        fi
        
        echo ""
        echo "ℹ️  Apple Silicon (Metal) acceleration is automatically used by Ollama"
        echo "   No additional driver installation needed for macOS"
        echo ""
    else
        echo "⚠️  Intel Mac detected (not Apple Silicon)"
        echo "   Ollama will run on CPU"
        HAS_CPU_ONLY=true
    fi
fi

# Path 2: Check for NVIDIA GPUs (Linux/WSL/Windows)
if [ "$OS_TYPE" = "linux" ] || [ "$IS_WSL" = true ] || [ "$IS_WINDOWS" = true ]; then
    if [ "$IS_WSL" = true ]; then
        echo "🐧 Detected Linux/WSL - Checking for NVIDIA GPUs..."
    elif [ "$IS_WINDOWS" = true ]; then
        echo "🪟 Detected Windows - Checking for NVIDIA GPUs..."
    else
        echo "🐧 Detected Linux - Checking for NVIDIA GPUs..."
    fi
    echo ""
    
    # First check if nvidia-smi is available (most reliable, works in WSL)
    if command_exists nvidia-smi && nvidia-smi >/dev/null 2>&1; then
        echo "✅ NVIDIA GPU detected via nvidia-smi"
        echo ""
        echo "📊 GPU Information:"
        nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader,nounits | while IFS=, read -r index name driver mem; do
            echo "   GPU $index: $name (Driver: $driver, Memory: ${mem}MB)"
        done
        echo ""
        HAS_NVIDIA_GPU=true
        DRIVERS_INSTALLED=true
    # Fallback to lspci if nvidia-smi not available (for detection without drivers)
    elif command_exists lspci; then
        GPU_COUNT=$(lspci 2>/dev/null | grep -i "nvidia\|vga.*nvidia" | wc -l | tr -d ' ')
        if [ "$GPU_COUNT" -gt 0 ]; then
            echo "✅ Found $GPU_COUNT NVIDIA GPU(s) via PCI scan:"
            echo ""
            lspci 2>/dev/null | grep -i "nvidia\|vga.*nvidia" | while IFS= read -r line; do
                echo "   • $line"
            done
            echo ""
            HAS_NVIDIA_GPU=true
        else
            echo "⚠️  No NVIDIA GPUs detected"
            echo ""
            HAS_CPU_ONLY=true
        fi
    else
        echo "⚠️  Cannot detect GPUs (this is normal in some WSL/Windows environments)"
        echo "   Ollama will run on CPU"
        echo ""
        HAS_CPU_ONLY=true
    fi
fi

# Path 3: CPU-only fallback (if no GPU detected)
if [ "$HAS_NVIDIA_GPU" = false ] && [ "$HAS_APPLE_SILICON" = false ] && [ "$OS_TYPE" != "macos" ]; then
    HAS_CPU_ONLY=true
fi

# NVIDIA-specific setup (Linux/WSL only) - only if GPU detected but drivers not confirmed
if [ "$HAS_NVIDIA_GPU" = true ] && [ "$DRIVERS_INSTALLED" = false ] && [ "$OS_TYPE" = "linux" ]; then
    echo "🔍 Checking for NVIDIA drivers..."
    if command_exists nvidia-smi && nvidia-smi >/dev/null 2>&1; then
        echo "✅ NVIDIA drivers detected (nvidia-smi available)"
        echo ""
        echo "📊 GPU Information:"
        nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader,nounits | while IFS=, read -r index name driver mem; do
            echo "   GPU $index: $name (Driver: $driver, Memory: ${mem}MB)"
        done
        echo ""
        DRIVERS_INSTALLED=true
    else
        echo "❌ NVIDIA drivers not detected (nvidia-smi not found)"
        echo ""
        echo "⚠️  GPUs detected but drivers are not installed"
        echo ""
        echo "For WSL users: Install NVIDIA drivers on Windows host, not in WSL"
        echo "   Visit: https://www.nvidia.com/Download/index.aspx"
        echo ""
        echo "⚠️  Skipping driver installation. GPU acceleration may not be available."
        echo ""
    fi
fi

# Configure NVIDIA Container Toolkit runtime if drivers are installed (Linux/WSL only)
if [ "$DRIVERS_INSTALLED" = true ] && [ "$OS_TYPE" = "linux" ] && [ "$IS_WINDOWS" = false ]; then
    echo "🔧 Configuring Docker NVIDIA runtime..."
    echo ""
    
    # Check if nvidia-ctk is available
    if command_exists nvidia-ctk; then
        echo "✅ nvidia-ctk found, configuring runtime..."
        if sudo nvidia-ctk runtime configure --runtime=docker --set-as-default 2>/dev/null; then
            echo "✅ NVIDIA runtime configured successfully"
            echo ""
            echo "🔄 Restarting Docker service..."
            if sudo systemctl restart docker 2>/dev/null || sudo service docker restart 2>/dev/null; then
                echo "✅ Docker service restarted successfully"
            else
                echo "⚠️  Failed to restart Docker service. You may need to restart it manually."
            fi
        else
            echo "⚠️  Failed to configure NVIDIA runtime automatically."
            echo "   For WSL users: Ensure Docker Desktop has WSL 2 backend enabled"
            echo "   and GPU support is enabled in Docker Desktop settings."
        fi
    else
        echo "ℹ️  nvidia-ctk not found. For WSL users, GPU support is handled by Docker Desktop."
        echo "   Ensure 'Use the WSL 2 based engine' and GPU support are enabled in Docker Desktop."
    fi
    echo ""
fi

# Summary of detection
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Detection Summary:"
if [ "$HAS_NVIDIA_GPU" = true ] && [ "$DRIVERS_INSTALLED" = true ]; then
    echo "   ✅ NVIDIA GPU detected and configured"
elif [ "$HAS_APPLE_SILICON" = true ]; then
    echo "   ✅ Apple Silicon (M1/M2/M3/M4) detected"
elif [ "$HAS_CPU_ONLY" = true ]; then
    echo "   ℹ️  CPU-only mode (no GPU acceleration)"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if Docker is available
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🐳 Docker Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if ! command_exists docker; then
    echo "❌ Docker is not installed or not in PATH"
    echo ""
    if [ "$IS_WINDOWS" = true ] || [ "$IS_WSL" = true ]; then
        echo "Please install Docker Desktop for Windows:"
        echo "   https://www.docker.com/products/docker-desktop/"
        echo ""
        echo "For WSL users:"
        echo "   1. Install Docker Desktop for Windows"
        echo "   2. Enable 'Use the WSL 2 based engine' in Docker Desktop settings"
        echo "   3. Enable integration with your WSL distribution"
    else
        echo "Please install Docker and try again"
    fi
    exit 1
fi

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running"
    echo ""
    if [ "$IS_WINDOWS" = true ] || [ "$IS_WSL" = true ]; then
        echo "Please start Docker Desktop for Windows"
    elif [ "$OS_TYPE" = "macos" ]; then
        echo "Please start Docker Desktop for Mac"
    else
        echo "Please start Docker service:"
        echo "   sudo systemctl start docker"
    fi
    exit 1
fi

echo "✅ Docker is installed and running"
echo ""

# Check if docker compose is available
DOCKER_COMPOSE_CMD=""
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
    echo "✅ Docker Compose (v2) found"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
    echo "✅ Docker Compose (v1) found"
else
    echo "⚠️  Docker Compose not found, will use standalone container"
fi
echo ""

# Try to find docker-compose file (optional)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.dev.yml"
USE_COMPOSE=false

if [ -n "$DOCKER_COMPOSE_CMD" ] && [ -f "$COMPOSE_FILE" ] && grep -q "ollama:" "$COMPOSE_FILE"; then
    echo "📋 Found docker-compose.dev.yml with Ollama service"
    echo "   Using docker-compose for setup"
    USE_COMPOSE=true
else
    echo "📋 Using standalone Docker container setup"
    USE_COMPOSE=false
fi
echo ""

# Setup Ollama container
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Setting up Ollama Container"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if container already exists and is running
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "✅ Ollama container '$CONTAINER_NAME' is already running"
    echo ""
elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🔄 Found existing stopped container '$CONTAINER_NAME', starting it..."
    docker start "$CONTAINER_NAME"
    echo "✅ Container started"
    echo ""
else
    echo "📥 Pulling Ollama Docker image..."
    docker pull ollama/ollama:latest
    echo ""
    
    if [ "$USE_COMPOSE" = true ]; then
        echo "🔄 Starting Ollama container via docker-compose..."
        cd "$PROJECT_ROOT"
        $DOCKER_COMPOSE_CMD -f docker-compose.dev.yml up -d ollama
        echo ""
    else
        echo "🔄 Creating and starting Ollama container..."
        # Create volume for model persistence
        docker volume create ollama_dev_data 2>/dev/null || true
        
        # Determine GPU flags
        GPU_FLAGS=""
        if [ "$HAS_NVIDIA_GPU" = true ] && [ "$DRIVERS_INSTALLED" = true ]; then
            GPU_FLAGS="--gpus all"
        fi
        
        docker run -d \
            --name "$CONTAINER_NAME" \
            $GPU_FLAGS \
            -p "${OLLAMA_PORT}:11434" \
            -v ollama_dev_data:/root/.ollama \
            --restart unless-stopped \
            ollama/ollama:latest
        
        echo "✅ Container created and started"
        echo ""
    fi
fi

echo "⏳ Waiting for Ollama to be ready..."
sleep 5

# Wait for Ollama to be ready (max 60 seconds)
MAX_WAIT=60
WAIT_TIME=0
while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if docker exec "$CONTAINER_NAME" curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama container is ready!"
        break
    fi
    echo "   Waiting for Ollama to start... ($WAIT_TIME/$MAX_WAIT seconds)"
    sleep 2
    WAIT_TIME=$((WAIT_TIME + 2))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo "⚠️  Ollama container started but may not be fully ready yet"
    echo "   You can check status with: docker logs $CONTAINER_NAME"
    echo "   Continuing anyway..."
fi
echo ""

# Check for models
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Checking for Ollama models..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📦 Using container: $CONTAINER_NAME"
echo ""

# List models in the container
echo "📋 Checking for installed models..."
MODELS=$(docker exec "$CONTAINER_NAME" ollama list 2>/dev/null || echo "")

# Check if there are any models (ollama list outputs header + models, so we check for more than 1 line)
MODEL_COUNT=$(echo "$MODELS" | grep -v "^NAME" | grep -v "^$" | wc -l | tr -d ' ')

if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "⚠️  No models found in Ollama container"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⚠️  IMPORTANT: This project uses llama3:8b as the DEFAULT model"
    echo "   It is recommended to install llama3:8b for compatibility with this project"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Available models to install:"
    echo ""
    echo "  📌 RECOMMENDED (Project Default):"
    echo "    1) llama3:8b (4.7GB) - ⭐ DEFAULT - Used by this project"
    echo ""
    echo "  🦙 Llama Models:"
    echo "    2) llama3.2:1b (1.3GB) - Small, fast"
    echo "    3) llama3.2:3b (2.0GB) - Balanced"
    echo "    4) llama3.1:8b (4.7GB) - Latest Llama 3.1"
    echo "    5) llama3.1:70b (40GB) - Large, powerful"
    echo ""
    echo "  🌟 Mistral Models:"
    echo "    6) mistral:7b (4.1GB) - Efficient, high quality"
    echo "    7) mistral-nemo:12b (7.0GB) - Enhanced Mistral"
    echo ""
    echo "  💎 Gemma Models:"
    echo "    8) gemma2:2b (1.4GB) - Small, efficient"
    echo "    9) gemma2:9b (5.4GB) - Balanced"
    echo "   10) gemma2:27b (16GB) - Large, powerful"
    echo ""
    echo "  🧠 Phi Models:"
    echo "   11) phi3:mini (2.3GB) - Small, fast"
    echo "   12) phi3:medium (7.0GB) - Balanced"
    echo ""
    echo "  💻 Code Models:"
    echo "   13) codellama:7b (3.8GB) - Code generation"
    echo "   14) codellama:13b (7.3GB) - Larger code model"
    echo "   15) deepseek-coder:6.7b (4.1GB) - Advanced coding"
    echo ""
    echo "  🌟 Other Models:"
    echo "   16) qwen2.5:7b (4.4GB) - Alibaba's model"
    echo "   17) neural-chat:7b (4.1GB) - Conversational AI"
    echo ""
    echo "  🔧 Custom:"
    echo "   18) Enter custom model name"
    echo ""
    echo "  ⏭️  Skip (install later):"
    echo "   19) Skip model installation"
    echo ""
    echo "Enter model number(s) to install (comma-separated, e.g., 1,6,13):"
    read -r selection
    
    # Parse comma-separated selection
    IFS=',' read -ra SELECTED <<< "$selection"
    
    # Model mapping
    declare -A MODEL_MAP=(
        ["1"]="llama3:8b"
        ["2"]="llama3.2:1b"
        ["3"]="llama3.2:3b"
        ["4"]="llama3.1:8b"
        ["5"]="llama3.1:70b"
        ["6"]="mistral:7b"
        ["7"]="mistral-nemo:12b"
        ["8"]="gemma2:2b"
        ["9"]="gemma2:9b"
        ["10"]="gemma2:27b"
        ["11"]="phi3:mini"
        ["12"]="phi3:medium"
        ["13"]="codellama:7b"
        ["14"]="codellama:13b"
        ["15"]="deepseek-coder:6.7b"
        ["16"]="qwen2.5:7b"
        ["17"]="neural-chat:7b"
    )
    
    MODELS_TO_PULL=()
    
    for num in "${SELECTED[@]}"; do
        num=$(echo "$num" | tr -d ' ') # Remove whitespace
        
        if [ "$num" == "18" ]; then
            echo ""
            echo "Enter custom model name (e.g., llama3:8b, mistral:7b):"
            read -r custom_model
            if [ -n "$custom_model" ]; then
                MODELS_TO_PULL+=("$custom_model")
            fi
        elif [ "$num" == "19" ]; then
            echo ""
            echo "⏭️  Skipping model installation"
            MODELS_TO_PULL=()
            break
        elif [ -n "${MODEL_MAP[$num]}" ]; then
            MODELS_TO_PULL+=("${MODEL_MAP[$num]}")
        else
            echo "⚠️  Invalid selection: $num (skipping)"
        fi
    done
    
    if [ ${#MODELS_TO_PULL[@]} -gt 0 ]; then
        echo ""
        echo "📥 Pulling ${#MODELS_TO_PULL[@]} model(s)..."
        echo "   This may take several minutes depending on your internet connection and model sizes..."
        echo ""
        
        SUCCESS_COUNT=0
        FAILED_COUNT=0
        
        for model in "${MODELS_TO_PULL[@]}"; do
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "📦 Pulling: $model"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            if docker exec "$CONTAINER_NAME" ollama pull "$model"; then
                echo ""
                echo "✅ Successfully pulled: $model"
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            else
                echo ""
                echo "❌ Failed to pull: $model"
                FAILED_COUNT=$((FAILED_COUNT + 1))
            fi
            echo ""
        done
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 Summary: $SUCCESS_COUNT succeeded, $FAILED_COUNT failed"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        if [ $SUCCESS_COUNT -gt 0 ]; then
            echo "✅ Model installation complete!"
        fi
        
        if [ $FAILED_COUNT -gt 0 ]; then
            echo "⚠️  Some models failed to install. You can try again later."
        fi
    fi
else
    echo "✅ Found $MODEL_COUNT model(s) in Ollama container:"
    echo ""
    echo "$MODELS"
    echo ""
    echo "ℹ️  To pull additional models, use:"
    echo "   docker exec -it $CONTAINER_NAME ollama pull <model-name>"
    echo ""
    echo "   Or run this script again to see the model selection menu"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Ollama Installation Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📝 Configuration:"
echo "   - Container name: $CONTAINER_NAME"
echo "   - Service URL: http://ollama:11434 (from other containers)"
echo "   - Host URL: http://localhost:${OLLAMA_PORT} (from host machine)"
echo ""

if [ "$USE_COMPOSE" = true ]; then
    echo "📝 Next steps:"
    echo "   1. The Ollama service is configured in docker-compose.dev.yml"
    echo "   2. Cyrex will automatically connect to Ollama at http://ollama:11434"
    echo "   3. Start Cyrex service:"
    echo "      docker compose -f docker-compose.dev.yml up -d cyrex"
    echo ""
else
    echo "📝 Next steps:"
    echo "   1. Ollama is running as a standalone container"
    echo "   2. Configure your application to connect to: http://localhost:${OLLAMA_PORT}"
    echo "   3. For Docker containers, use: http://host.docker.internal:${OLLAMA_PORT}"
    echo ""
fi

echo "💡 Tips:"
echo "   - View Ollama logs: docker logs $CONTAINER_NAME"
if [ "$USE_COMPOSE" = true ]; then
    echo "   - Restart Ollama: docker compose -f docker-compose.dev.yml restart ollama"
else
    echo "   - Restart Ollama: docker restart $CONTAINER_NAME"
fi
echo "   - Pull more models: docker exec -it $CONTAINER_NAME ollama pull <model-name>"
if [ "$USE_COMPOSE" = false ]; then
    echo "   - Models are persisted in the ollama_dev_data Docker volume"
fi
echo "   - Stop Ollama: docker stop $CONTAINER_NAME"
echo "   - Start Ollama: docker start $CONTAINER_NAME"
echo ""


