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

# Function to detect system RAM
detect_system_ram() {
    if [ "$OS_TYPE" = "macos" ]; then
        # macOS: Get total physical memory in GB
        RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
        SYSTEM_RAM_GB=$((RAM_BYTES / 1024 / 1024 / 1024))
    elif [ "$OS_TYPE" = "linux" ]; then
        # Linux: Get total memory in GB
        RAM_KB=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo "0")
        SYSTEM_RAM_GB=$((RAM_KB / 1024 / 1024))
    else
        SYSTEM_RAM_GB=0
    fi
    
    # Round to nearest integer
    if [ "$SYSTEM_RAM_GB" -lt 1 ]; then
        SYSTEM_RAM_GB=0
    fi
}

# Function to detect GPU VRAM
detect_gpu_vram() {
    GPU_VRAM_GB=0
    
    if [ "$HAS_NVIDIA_GPU" = true ] && command_exists nvidia-smi; then
        # Get VRAM in MB from first GPU, convert to GB
        # Handle both single and multiple GPU outputs
        VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | awk '{print $1}' | tr -d ' ' || echo "0")
        if [ -n "$VRAM_MB" ] && [ "$VRAM_MB" != "0" ] && [ "$VRAM_MB" != "" ]; then
            # Convert MB to GB (round down for safety)
            GPU_VRAM_GB=$((VRAM_MB / 1024))
        fi
    elif [ "$HAS_APPLE_SILICON" = true ]; then
        # Apple Silicon: Use unified memory (same as system RAM)
        # For Apple Silicon, the GPU shares system RAM, so we use system RAM as VRAM estimate
        GPU_VRAM_GB=$SYSTEM_RAM_GB
    fi
    
    # Ensure we have a reasonable value
    if [ "$GPU_VRAM_GB" -lt 1 ]; then
        GPU_VRAM_GB=0
    fi
}

# Function to categorize model based on hardware
# Returns: "recommended", "usable", "marginal", or "no"
# Based on exact setup requirements provided
categorize_model() {
    local model_name=$1
    local ram_gb=$2
    local vram_gb=$3
    
    # Determine setup category
    # Handle edge cases: 15GB VRAM is close to 16GB, treat as Setup 5
    # Also handle cases where RAM is close to 32GB (e.g., 30GB+)
    local setup="unknown"
    if ([ "$ram_gb" -ge 32 ] || [ "$ram_gb" -ge 30 ]) && ([ "$vram_gb" -ge 16 ] || [ "$vram_gb" -ge 15 ]); then
        setup="setup5"  # 32GB+ RAM + 15GB+ VRAM (best experience)
    elif [ "$ram_gb" -ge 32 ] && [ "$vram_gb" -ge 10 ]; then
        setup="setup4"  # 32GB RAM + 10GB+ VRAM
    elif [ "$ram_gb" -ge 32 ] && [ "$vram_gb" -ge 8 ]; then
        setup="setup3"  # 32GB RAM + 8GB+ VRAM
    # ADD THIS NEW CONDITION FOR YOUR CASE:
    elif [ "$vram_gb" -ge 15 ]; then
        setup="setup5"  # Treat 15GB VRAM as Setup 5 (Premium GPU Experience)
    elif [ "$ram_gb" -ge 16 ] && [ "$vram_gb" -ge 10 ]; then
        setup="setup2"  # 16GB RAM + 10GB+ VRAM
    elif [ "$ram_gb" -ge 16 ] && [ "$vram_gb" -ge 8 ]; then
        setup="setup1"  # 16GB RAM + 8GB+ VRAM
    elif [ "$ram_gb" -ge 16 ] || [ "$vram_gb" -ge 8 ]; then
        setup="basic"   # At least 16GB RAM or 8GB VRAM
    else
        setup="minimal" # Less than 16GB RAM and 8GB VRAM
    fi
    
    # Categorize based on model name and setup
    case "$model_name" in
        # Small models (1-3B) - Safe on all setups
        "llama3.2:1b"|"llama3.2:3b"|"gemma2:2b"|"phi3:mini")
            if [ "$ram_gb" -ge 8 ]; then
                echo "recommended"
            else
                echo "usable"
            fi
            ;;
        
        # 7B models - Safe on Setup 1+
        "mistral:7b"|"neural-chat:7b"|"qwen2.5:7b"|"gemma:7b"|"yi:6b"|"openchat:7b"|"zephyr:7b"|"nous-hermes:7b"|"mythomax:7b"|"dolphin-mistral:7b"|"orca-mini:7b")
            # For high-end GPU setups, prioritize VRAM over system RAM
            if [ "$setup" = "setup5" ] || [ "$setup" = "setup4" ] || [ "$setup" = "setup3" ] || [ "$setup" = "setup2" ]; then
                echo "recommended"
            elif [ "$vram_gb" -ge 8 ] && [ "$ram_gb" -ge 16 ]; then
                echo "recommended"
            elif [ "$vram_gb" -ge 8 ] || [ "$ram_gb" -ge 16 ]; then
                echo "usable"
            elif [ "$ram_gb" -ge 8 ]; then
                echo "marginal"
            else
                echo "no"
            fi
            ;;
        
        # 8B models
        "llama3:8b"|"llama3.1:8b")
            if [ "$setup" = "setup5" ] || [ "$setup" = "setup4" ] || [ "$setup" = "setup3" ] || [ "$setup" = "setup2" ]; then
                echo "recommended"
            elif [ "$setup" = "setup1" ]; then
                echo "usable"
            elif [ "$ram_gb" -ge 16 ]; then
                echo "marginal"
            else
                echo "no"
            fi
            ;;
        
        # 9B models
        "gemma2:9b"|"yi:9b")
            if [ "$setup" = "setup5" ] || [ "$setup" = "setup4" ] || [ "$setup" = "setup3" ] || [ "$setup" = "setup2" ]; then
                echo "recommended"
            elif [ "$setup" = "setup1" ]; then
                echo "usable"
            elif [ "$ram_gb" -ge 32 ]; then
                echo "marginal"
            else
                echo "no"
            fi
            ;;
        
        # 11-12B models
        "mistral-nemo:12b"|"falcon:11b")
            if [ "$setup" = "setup5" ] || [ "$setup" = "setup4" ] || [ "$setup" = "setup3" ] || [ "$setup" = "setup2" ]; then
                echo "recommended"
            elif [ "$ram_gb" -ge 32 ] && [ "$vram_gb" -ge 8 ]; then
                echo "usable"
            else
                echo "marginal"
            fi
            ;;
        
        # 13B models
        "vicuna:13b"|"openhermes:13b")
            if [ "$setup" = "setup5" ] || [ "$setup" = "setup4" ] || [ "$setup" = "setup3" ]; then
                echo "recommended"
            elif [ "$ram_gb" -ge 32 ]; then
                echo "usable"
            else
                echo "marginal"
            fi
            ;;
        
        # 27B models
        "gemma2:27b")
            if [ "$setup" = "setup5" ]; then
                echo "recommended"
            elif [ "$ram_gb" -ge 32 ] && [ "$vram_gb" -ge 10 ]; then
                echo "marginal"
            else
                echo "no"
            fi
            ;;
        
        # Mixture of experts
        "mixtral:8x7b")
            if [ "$setup" = "setup5" ] || [ "$setup" = "setup4" ]; then
                echo "marginal"
            else
                echo "no"
            fi
            ;;
        
        # 70B models - only for 48GB+ VRAM
        "llama3.1:70b")
            if [ "$vram_gb" -ge 48 ]; then
                echo "marginal"
            else
                echo "no"
            fi
            ;;
        
        # Coding models - 7B
        "codellama:7b"|"deepseek-coder:6.7b"|"qwen2.5-coder:7b"|"starcoder2:7b"|"wizardcoder:7b")
            # For high-end GPU setups, prioritize VRAM over system RAM
            if [ "$setup" = "setup5" ] || [ "$setup" = "setup4" ] || [ "$setup" = "setup3" ] || [ "$setup" = "setup2" ]; then
                echo "recommended"
            elif [ "$vram_gb" -ge 8 ] && [ "$ram_gb" -ge 16 ]; then
                echo "recommended"
            elif [ "$vram_gb" -ge 8 ] || [ "$ram_gb" -ge 16 ]; then
                echo "usable"
            else
                echo "marginal"
            fi
            ;;
        
        # Coding models - 13B
        "codellama:13b"|"wizardcoder:13b")
            if [ "$setup" = "setup5" ] || [ "$setup" = "setup4" ] || [ "$setup" = "setup3" ] || [ "$setup" = "setup2" ]; then
                echo "recommended"
            elif [ "$ram_gb" -ge 32 ]; then
                echo "usable"
            else
                echo "marginal"
            fi
            ;;
        
        # Phi3 medium
        "phi3:medium")
            if [ "$setup" = "setup5" ] || [ "$setup" = "setup4" ] || [ "$setup" = "setup3" ] || [ "$setup" = "setup2" ]; then
                echo "usable"
            elif [ "$setup" = "setup1" ]; then
                echo "usable"
            elif [ "$ram_gb" -ge 32 ]; then
                echo "marginal"
            else
                echo "no"
            fi
            ;;
        
        # Default for unknown models
        *)
            # Conservative default based on resources
            if [ "$setup" = "setup5" ]; then
                echo "recommended"
            elif [ "$setup" = "setup4" ] || [ "$setup" = "setup3" ]; then
                echo "usable"
            elif [ "$vram_gb" -ge 8 ] && [ "$ram_gb" -ge 16 ]; then
                echo "usable"
            else
                echo "marginal"
            fi
            ;;
    esac
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
HAS_AMD_GPU=false
# TODO: ROCm detection for AMD GPUs
# AMD GPU support via ROCm is not yet implemented
# Future enhancement: Detect AMD GPUs and ROCm installation
HAS_CPU_ONLY=false

# Hardware specs (will be detected)
GPU_VRAM_GB=0
SYSTEM_RAM_GB=0

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
        echo "ℹ️  Note: Unified memory is shared between CPU and GPU"
        echo "   Performance depends on system load; memory pressure + swap can hurt performance"
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

# Detect hardware specs
detect_system_ram
detect_gpu_vram

# Summary of detection
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Detection Summary:"
if [ "$HAS_NVIDIA_GPU" = true ] && [ "$DRIVERS_INSTALLED" = true ]; then
    echo "   ✅ NVIDIA GPU detected and configured"
    if [ "$GPU_VRAM_GB" -gt 0 ]; then
        echo "   💾 GPU VRAM: ${GPU_VRAM_GB} GB"
    fi
elif [ "$HAS_APPLE_SILICON" = true ]; then
    echo "   ✅ Apple Silicon (M1/M2/M3/M4) detected"
    if [ "$GPU_VRAM_GB" -gt 0 ]; then
        echo "   💾 Unified Memory (VRAM): ${GPU_VRAM_GB} GB"
    fi
elif [ "$HAS_CPU_ONLY" = true ]; then
    echo "   ℹ️  CPU-only mode (no GPU acceleration)"
    echo "   💡 Note: AMD GPU (ROCm) support is planned for future versions"
fi

if [ "$SYSTEM_RAM_GB" -gt 0 ]; then
    echo "   💾 System RAM: ${SYSTEM_RAM_GB} GB"
fi

# Determine setup category
SETUP_CATEGORY="unknown"
# Handle edge cases: 15GB VRAM is close to 16GB, treat as Setup 5
# Also handle cases where RAM is close to 32GB (e.g., 30GB+)
if ([ "$SYSTEM_RAM_GB" -ge 32 ] || [ "$SYSTEM_RAM_GB" -ge 30 ]) && ([ "$GPU_VRAM_GB" -ge 16 ] || [ "$GPU_VRAM_GB" -ge 15 ]); then
    SETUP_CATEGORY="setup5"
    echo "   🎯 Setup: ${SYSTEM_RAM_GB}GB RAM + ${GPU_VRAM_GB}GB VRAM (Best Experience - Setup 5 equivalent)"
elif [ "$SYSTEM_RAM_GB" -ge 32 ] && [ "$GPU_VRAM_GB" -ge 10 ]; then
    SETUP_CATEGORY="setup4"
    echo "   🎯 Setup: 32GB RAM + 10GB VRAM"
elif [ "$SYSTEM_RAM_GB" -ge 32 ] && [ "$GPU_VRAM_GB" -ge 8 ]; then
    SETUP_CATEGORY="setup3"
    echo "   🎯 Setup: 32GB RAM + 8GB VRAM"
# ADD THIS NEW CONDITION FOR YOUR CASE:
elif [ "$GPU_VRAM_GB" -ge 15 ]; then
    SETUP_CATEGORY="setup5"  # Treat 15GB VRAM as Setup 5
    echo "   🎯 Setup: ${SYSTEM_RAM_GB}GB RAM + ${GPU_VRAM_GB}GB VRAM (Premium GPU Experience)"
elif [ "$SYSTEM_RAM_GB" -ge 16 ] && [ "$GPU_VRAM_GB" -ge 10 ]; then
    SETUP_CATEGORY="setup2"
    echo "   🎯 Setup: 16GB RAM + 10GB VRAM"
elif [ "$SYSTEM_RAM_GB" -ge 16 ] && [ "$GPU_VRAM_GB" -ge 8 ]; then
    SETUP_CATEGORY="setup1"
    echo "   🎯 Setup: 16GB RAM + 8GB VRAM"
elif [ "$SYSTEM_RAM_GB" -ge 16 ] || [ "$GPU_VRAM_GB" -ge 8 ]; then
    SETUP_CATEGORY="basic"
    echo "   🎯 Setup: Basic (${SYSTEM_RAM_GB}GB RAM, ${GPU_VRAM_GB}GB VRAM)"
else
    SETUP_CATEGORY="minimal"
    echo "   🎯 Setup: Minimal (${SYSTEM_RAM_GB}GB RAM, ${GPU_VRAM_GB}GB VRAM)"
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

# Function to check if a model is installed
is_model_installed() {
    local model_name=$1
    echo "$MODELS" | grep -q "^${model_name}[[:space:]]" || echo "$MODELS" | grep -q "^${model_name}$"
}

# List models in the container
echo "📋 Checking for installed models..."
MODELS=$(docker exec "$CONTAINER_NAME" ollama list 2>/dev/null || echo "")

# Check if there are any models (ollama list outputs header + models, so we check for more than 1 line)
MODEL_COUNT=$(echo "$MODELS" | grep -v "^NAME" | grep -v "^$" | wc -l | tr -d ' ')

# Extract installed model names (for comparison)
INSTALLED_MODEL_NAMES=()
if [ "$MODEL_COUNT" -gt 0 ]; then
    echo "✅ Found $MODEL_COUNT model(s) in Ollama container:"
    echo ""
    echo "$MODELS"
    echo ""
    
    # Extract model names (first column, skip header)
    while IFS= read -r line; do
        if [[ "$line" =~ ^[[:space:]]*([^[:space:]]+) ]]; then
            model_name="${BASH_REMATCH[1]}"
            if [ "$model_name" != "NAME" ]; then
                INSTALLED_MODEL_NAMES+=("$model_name")
            fi
        fi
    done <<< "$MODELS"
fi

# Always show the menu, regardless of whether models exist
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "⚠️  No models found in Ollama container"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚠️  IMPORTANT: This project uses mistral:7b as the DEFAULT model"
echo "   It is recommended to install mistral:7b for compatibility with this project"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Define all models with their details
# Using | as delimiter since model names contain colons
#
# NOTE: Model list is curated & conservative
# Update periodically as Ollama registry evolves
# - New models are added regularly
# - Some tags may be deprecated or renamed
# - Check Ollama's model library for latest: https://ollama.com/library
#
declare -a MODEL_LIST=(
    "mistral:7b|4.1GB|⭐ DEFAULT - Used by this project"
    "llama3:8b|4.7GB|Alternative model"
    "llama3.2:1b|1.3GB|Small, fast"
    "llama3.2:3b|2.0GB|Balanced"
    "llama3.1:8b|4.7GB|Latest Llama 3.1"
    "llama3.1:70b|40GB|Large, powerful (requires 48GB+ VRAM)"
    "mistral:7b|4.1GB|Efficient, high quality"
    "mistral-nemo:12b|7.0GB|Enhanced Mistral"
    "mixtral:8x7b|26GB|Mixture of experts"
    "gemma2:2b|1.4GB|Small, efficient"
    "gemma2:9b|5.4GB|Balanced"
    "gemma2:27b|16GB|Large, powerful"
    "gemma:7b|4.6GB|Google's Gemma"
    "phi3:mini|2.3GB|Small, fast"
    "phi3:medium|7.0GB|Balanced"
    "codellama:7b|3.8GB|Code generation"
    "codellama:13b|7.3GB|Larger code model"
    "deepseek-coder:6.7b|4.1GB|Advanced coding"
    "qwen2.5:7b|4.4GB|Alibaba's model"
    "qwen2.5-coder:7b|4.4GB|Alibaba's coding model"
    "neural-chat:7b|4.1GB|Conversational AI"
    "yi:6b|3.8GB|Yi model 6B"
    "yi:9b|5.4GB|Yi model 9B"
    "openchat:7b|4.1GB|OpenChat model"
    "zephyr:7b|4.1GB|Zephyr model"
    "nous-hermes:7b|4.1GB|Nous Hermes"
    "mythomax:7b|4.1GB|MythoMax"
    "dolphin-mistral:7b|4.1GB|Dolphin Mistral"
    "orca-mini:7b|4.1GB|Orca Mini"
    "vicuna:13b|7.3GB|Vicuna 13B"
    "falcon:11b|6.0GB|Falcon 11B"
    "openhermes:13b|7.3GB|OpenHermes"
    "starcoder2:7b|4.1GB|StarCoder2"
    "wizardcoder:7b|4.1GB|WizardCoder 7B"
    "wizardcoder:13b|7.3GB|WizardCoder 13B"
)

# Calculate option numbers for special actions
TOTAL_MODELS=${#MODEL_LIST[@]}
CUSTOM_OPTION=$((TOTAL_MODELS + 1))
REMOVE_OPTION=$((TOTAL_MODELS + 2))
RECHECK_OPTION=$((TOTAL_MODELS + 3))

# Categorize models
declare -a RECOMMENDED_MODELS=()
declare -a USABLE_MODELS=()
declare -a MARGINAL_MODELS=()
declare -a NO_MODELS=()

MODEL_INDEX=1
for model_info in "${MODEL_LIST[@]}"; do
    IFS='|' read -r model_name model_size model_desc <<< "$model_info"
    category=$(categorize_model "$model_name" "$SYSTEM_RAM_GB" "$GPU_VRAM_GB")
    
    # Check if model is already installed
    INSTALLED_MARKER=""
    for installed_name in "${INSTALLED_MODEL_NAMES[@]}"; do
        if [ "$installed_name" = "$model_name" ]; then
            INSTALLED_MARKER=" [INSTALLED]"
            break
        fi
    done
    
    case "$category" in
        "recommended")
            RECOMMENDED_MODELS+=("$MODEL_INDEX|$model_name|$model_size|$model_desc|$INSTALLED_MARKER")
            ;;
        "usable")
            USABLE_MODELS+=("$MODEL_INDEX|$model_name|$model_size|$model_desc|$INSTALLED_MARKER")
            ;;
        "marginal")
            MARGINAL_MODELS+=("$MODEL_INDEX|$model_name|$model_size|$model_desc|$INSTALLED_MARKER")
            ;;
        "no")
            NO_MODELS+=("$MODEL_INDEX|$model_name|$model_size|$model_desc|$INSTALLED_MARKER")
            ;;
    esac
    MODEL_INDEX=$((MODEL_INDEX + 1))
done

echo "Available models to install (filtered for your hardware):"
echo ""

# Display recommended models
if [ ${#RECOMMENDED_MODELS[@]} -gt 0 ]; then
    echo "  ✅ RECOMMENDED (Best for your hardware):"
    for model_entry in "${RECOMMENDED_MODELS[@]}"; do
        IFS='|' read -r idx name size desc installed <<< "$model_entry"
        echo "    $idx) $name ($size) - $desc$installed"
    done
    echo ""
fi

# Display usable models
if [ ${#USABLE_MODELS[@]} -gt 0 ]; then
    echo "  ⚠️  USABLE (Will work but may be tight):"
    for model_entry in "${USABLE_MODELS[@]}"; do
        IFS='|' read -r idx name size desc installed <<< "$model_entry"
        echo "    $idx) $name ($size) - $desc$installed"
    done
    echo ""
fi

# Display marginal models
if [ ${#MARGINAL_MODELS[@]} -gt 0 ]; then
    echo "  ⚠️  MARGINAL (May be slow or unstable):"
    for model_entry in "${MARGINAL_MODELS[@]}"; do
        IFS='|' read -r idx name size desc installed <<< "$model_entry"
        echo "    $idx) $name ($size) - $desc$installed"
    done
    echo ""
fi

# Display not recommended models
if [ ${#NO_MODELS[@]} -gt 0 ]; then
    echo "  ❌ NOT RECOMMENDED (Insufficient hardware):"
    for model_entry in "${NO_MODELS[@]}"; do
        IFS='|' read -r idx name size desc installed <<< "$model_entry"
        echo "    $idx) $name ($size) - $desc$installed"
    done
    echo ""
fi

    echo "  🔧 Custom:"
    TOTAL_MODELS=${#MODEL_LIST[@]}
    CUSTOM_OPTION=$((TOTAL_MODELS + 1))
    REMOVE_OPTION=$((TOTAL_MODELS + 2))
    RECHECK_OPTION=$((TOTAL_MODELS + 3))
    echo "    $CUSTOM_OPTION) Enter custom model name"
    echo ""
    echo "  🗑️  Management:"
    echo "    $REMOVE_OPTION) Remove existing model(s)"
    echo "    $RECHECK_OPTION) Re-check hardware detection"
    echo ""

# Show best recommendation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ ${#RECOMMENDED_MODELS[@]} -gt 0 ]; then
    FIRST_REC=$(echo "${RECOMMENDED_MODELS[0]}" | cut -d'|' -f1)
    FIRST_REC_NAME=$(echo "${RECOMMENDED_MODELS[0]}" | cut -d'|' -f2)
    echo "💡 Best recommendation for your setup: Model #$FIRST_REC ($FIRST_REC_NAME)"
    
    # Find highest model in recommended or usable categories
    HIGHEST_MODEL=""
    HIGHEST_MODEL_CAT=""
    if [ ${#USABLE_MODELS[@]} -gt 0 ]; then
        # Get the last usable model (likely the largest)
        for model_entry in "${USABLE_MODELS[@]}"; do
            IFS='|' read -r idx name size desc installed <<< "$model_entry"
            HIGHEST_MODEL="$name"
            HIGHEST_MODEL_CAT="usable"
        done
    fi
    
    if [ -n "$HIGHEST_MODEL" ]; then
        echo "🎯 Highest model you can run: $HIGHEST_MODEL (see 'USABLE' section above)"
    fi
else
    echo "⚠️  No models are recommended for your current hardware"
    if [ ${#USABLE_MODELS[@]} -gt 0 ]; then
        FIRST_USABLE=$(echo "${USABLE_MODELS[0]}" | cut -d'|' -f2)
        echo "💡 Consider: $FIRST_USABLE (see 'USABLE' section above)"
    fi
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Enter option number(s) to install (comma-separated, e.g., 1,6,13) or management option ($REMOVE_OPTION,$RECHECK_OPTION):"
read -r selection
    
# Parse comma-separated selection
IFS=',' read -ra SELECTED <<< "$selection"

# Check for special management options first
HANDLE_REMOVE=false
HANDLE_RECHECK=false
MODELS_TO_PULL=()

for num in "${SELECTED[@]}"; do
    num=$(echo "$num" | tr -d ' ') # Remove whitespace
    
    if [ "$num" == "$REMOVE_OPTION" ]; then
        HANDLE_REMOVE=true
    elif [ "$num" == "$RECHECK_OPTION" ]; then
        HANDLE_RECHECK=true
    fi
done

# Handle re-check hardware
if [ "$HANDLE_RECHECK" = true ]; then
    echo ""
    echo "🔄 Re-checking hardware detection..."
    echo ""
    detect_system_ram
    detect_gpu_vram
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Updated Detection Summary:"
    if [ "$HAS_NVIDIA_GPU" = true ] && [ "$DRIVERS_INSTALLED" = true ]; then
        echo "   ✅ NVIDIA GPU detected and configured"
        if [ "$GPU_VRAM_GB" -gt 0 ]; then
            echo "   💾 GPU VRAM: ${GPU_VRAM_GB} GB"
        fi
    elif [ "$HAS_APPLE_SILICON" = true ]; then
        echo "   ✅ Apple Silicon (M1/M2/M3/M4) detected"
        if [ "$GPU_VRAM_GB" -gt 0 ]; then
            echo "   💾 Unified Memory (VRAM): ${GPU_VRAM_GB} GB"
        fi
    elif [ "$HAS_CPU_ONLY" = true ]; then
        echo "   ℹ️  CPU-only mode (no GPU acceleration)"
    fi
    
    if [ "$SYSTEM_RAM_GB" -gt 0 ]; then
        echo "   💾 System RAM: ${SYSTEM_RAM_GB} GB"
    fi
    
    # Determine setup category
    SETUP_CATEGORY="unknown"
    # Handle edge cases: 15GB VRAM is close to 16GB, treat as Setup 5
    # Also handle cases where RAM is close to 32GB (e.g., 30GB+)
    if ([ "$SYSTEM_RAM_GB" -ge 32 ] || [ "$SYSTEM_RAM_GB" -ge 30 ]) && ([ "$GPU_VRAM_GB" -ge 16 ] || [ "$GPU_VRAM_GB" -ge 15 ]); then
        SETUP_CATEGORY="setup5"
        echo "   🎯 Setup: ${SYSTEM_RAM_GB}GB RAM + ${GPU_VRAM_GB}GB VRAM (Best Experience - Setup 5 equivalent)"
    elif [ "$SYSTEM_RAM_GB" -ge 32 ] && [ "$GPU_VRAM_GB" -ge 10 ]; then
        SETUP_CATEGORY="setup4"
        echo "   🎯 Setup: 32GB RAM + 10GB VRAM"
    elif [ "$SYSTEM_RAM_GB" -ge 32 ] && [ "$GPU_VRAM_GB" -ge 8 ]; then
        SETUP_CATEGORY="setup3"
        echo "   🎯 Setup: 32GB RAM + 8GB VRAM"
    # ADD THIS NEW CONDITION FOR YOUR CASE:
    elif [ "$GPU_VRAM_GB" -ge 15 ]; then
        SETUP_CATEGORY="setup5"  # Treat 15GB VRAM as Setup 5
        echo "   🎯 Setup: ${SYSTEM_RAM_GB}GB RAM + ${GPU_VRAM_GB}GB VRAM (Premium GPU Experience)"
    elif [ "$SYSTEM_RAM_GB" -ge 16 ] && [ "$GPU_VRAM_GB" -ge 10 ]; then
        SETUP_CATEGORY="setup2"
        echo "   🎯 Setup: 16GB RAM + 10GB VRAM"
    elif [ "$SYSTEM_RAM_GB" -ge 16 ] && [ "$GPU_VRAM_GB" -ge 8 ]; then
        SETUP_CATEGORY="setup1"
        echo "   🎯 Setup: 16GB RAM + 8GB VRAM"
    elif [ "$SYSTEM_RAM_GB" -ge 16 ] || [ "$GPU_VRAM_GB" -ge 8 ]; then
        SETUP_CATEGORY="basic"
        echo "   🎯 Setup: Basic (${SYSTEM_RAM_GB}GB RAM, ${GPU_VRAM_GB}GB VRAM)"
    else
        SETUP_CATEGORY="minimal"
        echo "   🎯 Setup: Minimal (${SYSTEM_RAM_GB}GB RAM, ${GPU_VRAM_GB}GB VRAM)"
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # If only re-check was selected, exit
    if [ "$HANDLE_REMOVE" = false ] && [ ${#SELECTED[@]} -eq 1 ]; then
        echo "✅ Hardware detection complete. Run the script again to see updated recommendations."
        exit 0
    fi
fi

# Handle remove models
if [ "$HANDLE_REMOVE" = true ]; then
    echo ""
    echo "🗑️  Remove Existing Models"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    if [ ${#INSTALLED_MODEL_NAMES[@]} -eq 0 ]; then
        echo "ℹ️  No models installed to remove."
    else
        echo "Installed models:"
        REMOVE_INDEX=1
        declare -A REMOVE_MAP=()
        for installed_name in "${INSTALLED_MODEL_NAMES[@]}"; do
            echo "  $REMOVE_INDEX) $installed_name"
            REMOVE_MAP["$REMOVE_INDEX"]="$installed_name"
            REMOVE_INDEX=$((REMOVE_INDEX + 1))
        done
        echo ""
        echo "Enter model number(s) to remove (comma-separated, e.g., 1,2):"
        read -r remove_selection
        
        IFS=',' read -ra REMOVE_SELECTED <<< "$remove_selection"
        MODELS_TO_REMOVE=()
        
        for num in "${REMOVE_SELECTED[@]}"; do
            num=$(echo "$num" | tr -d ' ')
            if [ -n "${REMOVE_MAP[$num]}" ]; then
                MODELS_TO_REMOVE+=("${REMOVE_MAP[$num]}")
            else
                echo "⚠️  Invalid selection: $num (skipping)"
            fi
        done
        
        if [ ${#MODELS_TO_REMOVE[@]} -gt 0 ]; then
            echo ""
            echo "⚠️  You are about to remove ${#MODELS_TO_REMOVE[@]} model(s):"
            for model in "${MODELS_TO_REMOVE[@]}"; do
                echo "   • $model"
            done
            echo ""
            echo "Continue? (y/n)"
            read -r confirm_remove
            
            if [[ "$confirm_remove" =~ ^[Yy]$ ]]; then
                REMOVED_COUNT=0
                for model in "${MODELS_TO_REMOVE[@]}"; do
                    echo "🗑️  Removing: $model"
                    if docker exec "$CONTAINER_NAME" ollama rm "$model" 2>/dev/null; then
                        echo "   ✅ Removed: $model"
                        REMOVED_COUNT=$((REMOVED_COUNT + 1))
                    else
                        echo "   ❌ Failed to remove: $model"
                    fi
                done
                echo ""
                echo "✅ Removed $REMOVED_COUNT model(s)"
                echo ""
                echo "📋 Current models in container:"
                docker exec "$CONTAINER_NAME" ollama list
                echo ""
            else
                echo "Cancelled."
            fi
        fi
    fi
    
    # If only remove was selected, exit
    if [ ${#SELECTED[@]} -eq 1 ]; then
        exit 0
    fi
fi

# Model mapping - build dynamically from MODEL_LIST
declare -A MODEL_MAP=()
MAP_INDEX=1
for model_info in "${MODEL_LIST[@]}"; do
    IFS='|' read -r model_name model_size model_desc <<< "$model_info"
    MODEL_MAP["$MAP_INDEX"]="$model_name"
    MAP_INDEX=$((MAP_INDEX + 1))
done

WARNINGS=()

for num in "${SELECTED[@]}"; do
    num=$(echo "$num" | tr -d ' ') # Remove whitespace
    
    # Skip management options (already handled)
    if [ "$num" == "$REMOVE_OPTION" ] || [ "$num" == "$RECHECK_OPTION" ]; then
        continue
    fi
    
    if [ "$num" == "$CUSTOM_OPTION" ]; then
        echo ""
        echo "Enter custom model name (e.g., llama3:8b, mistral:7b):"
        read -r custom_model
        if [ -n "$custom_model" ]; then
            MODELS_TO_PULL+=("$custom_model")
            # Check custom model compatibility
            custom_category=$(categorize_model "$custom_model" "$SYSTEM_RAM_GB" "$GPU_VRAM_GB")
            if [ "$custom_category" = "marginal" ]; then
                WARNINGS+=("$custom_model may run slowly on your hardware")
            elif [ "$custom_category" = "no" ]; then
                WARNINGS+=("$custom_model is NOT recommended for your hardware (${SYSTEM_RAM_GB}GB RAM, ${GPU_VRAM_GB}GB VRAM)")
            fi
        fi
    elif [ -n "${MODEL_MAP[$num]}" ]; then
        model_name="${MODEL_MAP[$num]}"
        MODELS_TO_PULL+=("$model_name")
        
        # Check compatibility and warn if needed
        model_category=$(categorize_model "$model_name" "$SYSTEM_RAM_GB" "$GPU_VRAM_GB")
        if [ "$model_category" = "usable" ]; then
            WARNINGS+=("$model_name may be tight on your hardware")
        elif [ "$model_category" = "marginal" ]; then
            WARNINGS+=("$model_name may run slowly or be unstable on your hardware")
        elif [ "$model_category" = "no" ]; then
            WARNINGS+=("$model_name is NOT recommended for your hardware (${SYSTEM_RAM_GB}GB RAM, ${GPU_VRAM_GB}GB VRAM)")
        fi
    else
        echo "⚠️  Invalid selection: $num (skipping)"
    fi
done

# Show warnings if any
if [ ${#WARNINGS[@]} -gt 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⚠️  Compatibility Warnings:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for warning in "${WARNINGS[@]}"; do
        echo "   • $warning"
    done
    echo ""
    echo "Continue anyway? (y/n)"
    read -r continue_anyway
    if [[ ! "$continue_anyway" =~ ^[Yy]$ ]]; then
        echo "Cancelled. Please select models better suited for your hardware."
        exit 0
    fi
    echo ""
fi

# Only proceed with pulling if there are models to pull
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
elif [ "$HANDLE_REMOVE" = false ] && [ "$HANDLE_RECHECK" = false ]; then
    # Only show this message if user didn't select any management options or models
    echo ""
    echo "ℹ️  No models selected for installation."
    echo "   Select model numbers (1-17), custom model (18), remove models (19), or re-check hardware (20)"
fi

