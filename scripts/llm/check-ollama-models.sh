#!/bin/bash
# Check if Ollama container has models, and prompt to pull models if none exist

set -e

CONTAINER_NAME="deepiri-ollama-dev"

echo "GPU Detection and Driver Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect operating system
OS_TYPE="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]] || grep -qEi "(Microsoft|WSL)" /proc/version 2>/dev/null; then
    OS_TYPE="linux"
fi

# Detection flags
HAS_NVIDIA_GPU=false
HAS_APPLE_SILICON=false
HAS_CPU_ONLY=false

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
                
                # Check if Rosetta is needed (for x86_64 containers on Apple Silicon)
                if command_exists arch; then
                    NATIVE_ARCH=$(arch)
                    if [ "$NATIVE_ARCH" = "arm64" ]; then
                        echo "✅ Running natively on arm64"
                    fi
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

# Path 2: Check for NVIDIA GPUs (Linux/WSL)
if [ "$OS_TYPE" = "linux" ]; then
    echo "🐧 Detected Linux/WSL - Checking for NVIDIA GPUs..."
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
        GPU_COUNT=$(lspci | grep -i "nvidia\|vga.*nvidia" | wc -l | tr -d ' ')
        if [ "$GPU_COUNT" -gt 0 ]; then
            echo "✅ Found $GPU_COUNT NVIDIA GPU(s) via PCI scan:"
            echo ""
            lspci | grep -i "nvidia\|vga.*nvidia" | while IFS= read -r line; do
                echo "   • $line"
            done
            echo ""
            HAS_NVIDIA_GPU=true
        else
            echo "⚠️  No NVIDIA GPUs detected via lspci"
            echo ""
            HAS_CPU_ONLY=true
        fi
    else
        echo "⚠️  lspci not found and nvidia-smi not available"
        echo "   Cannot detect GPUs (this is normal in some WSL environments)"
        echo ""
        HAS_CPU_ONLY=true
    fi
fi

# Path 3: CPU-only fallback (if no GPU detected)
if [ "$HAS_NVIDIA_GPU" = false ] && [ "$HAS_APPLE_SILICON" = false ] && [ "$OS_TYPE" != "macos" ]; then
    HAS_CPU_ONLY=true
fi

# NVIDIA-specific setup (Linux only) - only if GPU detected but drivers not confirmed
DRIVERS_INSTALLED=${DRIVERS_INSTALLED:-false}
if [ "$HAS_NVIDIA_GPU" = true ] && [ "$DRIVERS_INSTALLED" = false ]; then
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
        echo "Would you like to install NVIDIA drivers? (y/n)"
        read -r install_drivers
        
        if [[ "$install_drivers" =~ ^[Yy]$ ]]; then
            echo ""
            echo "📦 Installing NVIDIA drivers..."
            echo ""
            
            # Detect Linux distribution
            if [ -f /etc/os-release ]; then
                . /etc/os-release
                OS=$ID
            else
                echo "❌ Cannot detect Linux distribution"
                exit 1
            fi
            
            case $OS in
                ubuntu|debian)
                    echo "Detected Ubuntu/Debian. Installing NVIDIA drivers..."
                    sudo apt-get update
                    sudo apt-get install -y nvidia-driver-535 nvidia-utils-535
                    ;;
                fedora|rhel|centos)
                    echo "Detected Fedora/RHEL/CentOS. Installing NVIDIA drivers..."
                    sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia
                    ;;
                arch|manjaro)
                    echo "Detected Arch/Manjaro. Installing NVIDIA drivers..."
                    sudo pacman -S --noconfirm nvidia nvidia-utils
                    ;;
                *)
                    echo "⚠️  Unsupported distribution: $OS"
                    echo "   Please install NVIDIA drivers manually for your distribution"
                    echo "   Visit: https://www.nvidia.com/Download/index.aspx"
                    exit 1
                    ;;
            esac
            
            echo ""
            echo "✅ Driver installation initiated. You may need to reboot your system."
            echo "   After reboot, run this script again to configure Docker runtime."
            echo ""
            exit 0
        else
            echo ""
            echo "⚠️  Skipping driver installation. GPU acceleration will not be available."
            echo ""
        fi
    fi
fi

# Configure NVIDIA Container Toolkit runtime if drivers are installed (Linux only)
if [ "$DRIVERS_INSTALLED" = true ] && [ "$OS_TYPE" = "linux" ]; then
    echo "🔧 Configuring Docker NVIDIA runtime..."
    echo ""
    
    # Check if nvidia-ctk is available
    if command_exists nvidia-ctk; then
        echo "✅ nvidia-ctk found, configuring runtime..."
        if sudo nvidia-ctk runtime configure --runtime=docker --set-as-default; then
            echo "✅ NVIDIA runtime configured successfully"
            echo ""
            echo "🔄 Restarting Docker service..."
            if sudo systemctl restart docker 2>/dev/null || sudo service docker restart 2>/dev/null; then
                echo "✅ Docker service restarted successfully"
            else
                echo "⚠️  Failed to restart Docker service. You may need to restart it manually:"
                echo "   sudo systemctl restart docker"
                echo "   or"
                echo "   sudo service docker restart"
            fi
        else
            echo "⚠️  Failed to configure NVIDIA runtime. You may need to:"
            echo "   1. Install nvidia-container-toolkit:"
            echo "      sudo apt-get install -y nvidia-container-toolkit"
            echo "   2. Configure runtime manually:"
            echo "      sudo nvidia-ctk runtime configure --runtime=docker --set-as-default"
            echo "   3. Restart Docker:"
            echo "      sudo systemctl restart docker"
        fi
    else
        echo "⚠️  nvidia-ctk not found. Installing nvidia-container-toolkit..."
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "ℹ️  Why install NVIDIA Container Toolkit?"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "The NVIDIA Container Toolkit allows Docker containers to access your GPU."
        echo "Without it, Ollama and other GPU-accelerated containers will run on CPU only,"
        echo "which is 10-50x slower for AI workloads."
        echo ""
        echo "Benefits:"
        echo "  ✅ Ollama will automatically use your GPU for faster model inference"
        echo "  ✅ GPU acceleration works in all Docker containers (no --gpus flag needed)"
        echo "  ✅ Better performance: 20-100+ tokens/sec (GPU) vs 2-5 tokens/sec (CPU)"
        echo "  ✅ Uses GPU VRAM instead of system RAM for models"
        echo ""
        echo "This is a one-time setup. After installation, Docker will automatically"
        echo "configure containers to use your NVIDIA GPU when available."
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        # Detect Linux distribution for nvidia-container-toolkit installation
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=$ID
        else
            echo "❌ Cannot detect Linux distribution"
            OS="unknown"
        fi
        
        case $OS in
            ubuntu|debian)
                echo "Installing nvidia-container-toolkit for Ubuntu/Debian..."
                
                # Detect architecture
                ARCH=$(dpkg --print-architecture 2>/dev/null || uname -m)
                if [ "$ARCH" = "x86_64" ]; then
                    ARCH="amd64"
                fi
                echo "   Detected architecture: $ARCH"
                
                # Add GPG key
                echo "   Adding GPG key..."
                if curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey 2>/dev/null | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null; then
                    echo "✅ GPG key added"
                else
                    echo "⚠️  Failed to add GPG key"
                    echo "   Attempting installation without GPG verification..."
                fi
                
                # Use the official stable repository (distribution-agnostic)
                # This is the correct method per NVIDIA documentation
                REPO_FILE="/etc/apt/sources.list.d/nvidia-container-toolkit.list"
                REPO_LINE="deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/${ARCH} /"
                
                echo "   Configuring repository..."
                echo "$REPO_LINE" | sudo tee "$REPO_FILE" >/dev/null
                
                # Verify the repository file was created correctly
                if [ -f "$REPO_FILE" ] && grep -q "nvidia.github.io" "$REPO_FILE" 2>/dev/null; then
                    echo "✅ Repository configuration file created"
                    echo "   Updating package lists..."
                    if sudo apt-get update 2>&1 | grep -q "nvidia-container-toolkit\|Reading package lists"; then
                        echo "✅ Package lists updated"
                        echo "   Installing nvidia-container-toolkit..."
                        if sudo apt-get install -y nvidia-container-toolkit; then
                            echo "✅ nvidia-container-toolkit installed successfully"
                        else
                            echo "⚠️  Installation failed. You may need to install manually."
                        fi
                    else
                        echo "⚠️  Failed to update package lists or repository not accessible"
                        echo "   Repository file contents:"
                        cat "$REPO_FILE" 2>/dev/null || echo "   (file not readable)"
                        echo ""
                        echo "   Please try manual installation:"
                        echo "   Visit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
                    fi
                else
                    echo "❌ Failed to create repository configuration file"
                    echo "   Please install nvidia-container-toolkit manually"
                    echo "   Visit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
                fi
                ;;
            fedora|rhel|centos)
                echo "Installing nvidia-container-toolkit for Fedora/RHEL/CentOS..."
                curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
                    sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
                sudo dnf install -y nvidia-container-toolkit
                ;;
            arch|manjaro)
                echo "Installing nvidia-container-toolkit for Arch/Manjaro..."
                sudo pacman -S --noconfirm nvidia-container-toolkit
                ;;
            *)
                echo "⚠️  Unsupported distribution: $OS"
                echo "   Please install nvidia-container-toolkit manually"
                echo "   Visit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
                ;;
        esac
        
        echo ""
        if command_exists nvidia-ctk; then
            echo "✅ nvidia-ctk installed, configuring runtime..."
            if sudo nvidia-ctk runtime configure --runtime=docker --set-as-default; then
                echo "✅ NVIDIA runtime configured successfully"
                echo ""
                echo "🔄 Restarting Docker service..."
                if sudo systemctl restart docker 2>/dev/null || sudo service docker restart 2>/dev/null; then
                    echo "✅ Docker service restarted successfully"
                else
                    echo "⚠️  Failed to restart Docker service. Please restart manually."
                fi
            fi
        else
            echo "⚠️  nvidia-ctk installation may have failed. Please install manually."
        fi
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

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Checking for Ollama models..."
echo ""

# Check if the specified container exists and is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    # Try to find any ollama container
    OLLAMA_CONTAINER=$(docker ps --format '{{.Names}}' | grep -i ollama | head -n 1)
    
    if [ -z "$OLLAMA_CONTAINER" ]; then
        echo "❌ Error: No Ollama container found"
        echo "   Expected: $CONTAINER_NAME"
        echo "   Please start the Ollama service first"
        exit 1
    else
        echo "⚠️  Container '$CONTAINER_NAME' not found, using '$OLLAMA_CONTAINER' instead"
        CONTAINER_NAME="$OLLAMA_CONTAINER"
    fi
fi

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
        elif [ -n "${MODEL_MAP[$num]}" ]; then
            MODELS_TO_PULL+=("${MODEL_MAP[$num]}")
        else
            echo "⚠️  Invalid selection: $num (skipping)"
        fi
    done
    
    if [ ${#MODELS_TO_PULL[@]} -eq 0 ]; then
        echo ""
        echo "❌ No valid models selected. Exiting."
        exit 0
    fi
    
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
        
        if docker exec -it "$CONTAINER_NAME" ollama pull "$model"; then
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
    echo "📋 Current models in container:"
    docker exec "$CONTAINER_NAME" ollama list
    echo ""
    
    if [ $SUCCESS_COUNT -gt 0 ]; then
        echo "✅ Model installation complete!"
    fi
    
    if [ $FAILED_COUNT -gt 0 ]; then
        echo "⚠️  Some models failed to install. You can try again later."
        exit 1
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

