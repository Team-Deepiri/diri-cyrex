#!/bin/bash
# Install NVIDIA Container Toolkit for Docker GPU support
# This allows Docker containers to access NVIDIA GPUs

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 NVIDIA Container Toolkit Installation"
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
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS_TYPE="windows"
fi

# Check if running on macOS (not supported for NVIDIA)
if [ "$OS_TYPE" = "macos" ]; then
    echo "⚠️  macOS detected - NVIDIA Container Toolkit is not available for macOS"
    echo ""
    echo "For macOS with Apple Silicon:"
    echo "  • PyTorch will automatically use MPS (Metal Performance Shaders)"
    echo "  • No additional setup needed for GPU acceleration"
    echo "  • Docker containers typically can't access GPU on macOS"
    echo ""
    echo "For best performance on macOS, run models natively (not in Docker)."
    exit 0
fi

# Check if running on Windows
if [ "$OS_TYPE" = "windows" ]; then
    echo "⚠️  Windows detected"
    echo ""
    echo "For Windows with WSL2:"
    echo "  • Run this script inside WSL2 (not in Windows)"
    echo "  • Ensure WSL2 has NVIDIA drivers installed"
    echo ""
    echo "For Windows with Docker Desktop:"
    echo "  • GPU support requires WSL2 backend"
    echo "  • Install NVIDIA drivers in WSL2"
    echo "  • Then run this script in WSL2"
    echo ""
    read -p "Are you running this in WSL2? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Please run this script inside WSL2 for GPU support."
        exit 1
    fi
    OS_TYPE="linux"
fi

# Check if running on Linux
if [ "$OS_TYPE" != "linux" ]; then
    echo "❌ Unsupported operating system: $OS_TYPE"
    echo "   NVIDIA Container Toolkit is only available for Linux/WSL2"
    exit 1
fi

echo "🐧 Linux/WSL2 detected"
echo ""

# Check if Docker is installed
if ! command_exists docker; then
    echo "❌ Docker is not installed"
    echo ""
    echo "Please install Docker first:"
    echo "  • Ubuntu/Debian: sudo apt-get install docker.io"
    echo "  • Or visit: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✅ Docker found: $(docker --version)"
echo ""

# Check if NVIDIA drivers are installed
if command_exists nvidia-smi; then
    echo "✅ NVIDIA drivers detected"
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader,nounits | \
        while IFS=, read -r index name driver mem; do
            echo "   GPU $index: $name (Driver: $driver, Memory: ${mem}MB)"
        done
    echo ""
else
    echo "⚠️  nvidia-smi not found - NVIDIA drivers may not be installed"
    echo ""
    echo "Please install NVIDIA drivers first:"
    echo "  • Ubuntu: sudo apt-get install nvidia-driver-535 (or latest)"
    echo "  • Or visit: https://www.nvidia.com/Download/index.aspx"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    echo ""
fi

# Check if already installed
if command_exists nvidia-ctk; then
    echo "✅ NVIDIA Container Toolkit is already installed"
    echo ""
    echo "Current version:"
    nvidia-ctk --version 2>/dev/null || echo "   (version info unavailable)"
    echo ""
    
    # Check if runtime is configured
    if docker info 2>/dev/null | grep -q "nvidia"; then
        echo "✅ NVIDIA runtime is configured"
    else
        echo "⚠️  NVIDIA runtime not configured, configuring now..."
        if sudo nvidia-ctk runtime configure --runtime=docker --set-as-default 2>/dev/null; then
            echo "✅ NVIDIA runtime configured successfully"
            echo ""
            echo "🔄 Restarting Docker service..."
            if sudo systemctl restart docker 2>/dev/null || sudo service docker restart 2>/dev/null; then
                echo "✅ Docker service restarted"
            else
                echo "⚠️  Please restart Docker manually: sudo systemctl restart docker"
            fi
        else
            echo "⚠️  Failed to configure runtime"
        fi
    fi
    
    # Test GPU access
    echo ""
    echo "🧪 Testing GPU access in Docker..."
    if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        echo "✅ GPU access test successful!"
        echo ""
        echo "Your Docker containers can now access NVIDIA GPUs."
        exit 0
    else
        echo "⚠️  GPU access test failed"
        echo "   This may indicate a configuration issue"
    fi
    exit 0
fi

# Installation section
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ℹ️  About NVIDIA Container Toolkit"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "The NVIDIA Container Toolkit allows Docker containers to access your GPU."
echo "Without it, GPU-accelerated containers will run on CPU only, which is"
echo "10-50x slower for AI workloads."
echo ""
echo "Benefits:"
echo "  ✅ Ollama and other AI tools will automatically use your GPU"
echo "  ✅ GPU acceleration works in all Docker containers"
echo "  ✅ Better performance: 20-100+ tokens/sec (GPU) vs 2-5 tokens/sec (CPU)"
echo "  ✅ Uses GPU VRAM instead of system RAM for models"
echo ""
echo "This is a one-time setup. After installation, Docker will automatically"
echo "configure containers to use your NVIDIA GPU when available."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

read -p "Continue with installation? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 0
fi

echo ""
echo "📦 Installing NVIDIA Container Toolkit..."
echo ""

# Detect Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    OS_VERSION=$VERSION_ID
else
    echo "❌ Cannot detect Linux distribution"
    echo "   Please install manually: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    exit 1
fi

echo "Detected: $OS $OS_VERSION"
echo ""

# Installation based on distribution
case $OS in
    ubuntu|debian)
        echo "Installing for Ubuntu/Debian..."
        echo ""
        
        # Detect architecture
        ARCH=$(dpkg --print-architecture 2>/dev/null || uname -m)
        if [ "$ARCH" = "x86_64" ]; then
            ARCH="amd64"
        elif [ "$ARCH" = "aarch64" ]; then
            ARCH="arm64"
        fi
        echo "   Architecture: $ARCH"
        
        # Add GPG key
        echo "   Adding GPG key..."
        if curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey 2>/dev/null | \
            sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null; then
            echo "   ✅ GPG key added"
        else
            echo "   ⚠️  Failed to add GPG key, continuing anyway..."
        fi
        
        # Configure repository
        REPO_FILE="/etc/apt/sources.list.d/nvidia-container-toolkit.list"
        REPO_LINE="deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/${ARCH} /"
        
        echo "   Configuring repository..."
        echo "$REPO_LINE" | sudo tee "$REPO_FILE" >/dev/null
        
        if [ -f "$REPO_FILE" ] && grep -q "nvidia.github.io" "$REPO_FILE" 2>/dev/null; then
            echo "   ✅ Repository configured"
        else
            echo "   ❌ Failed to configure repository"
            exit 1
        fi
        
        # Update package lists
        echo "   Updating package lists..."
        if sudo apt-get update >/dev/null 2>&1; then
            echo "   ✅ Package lists updated"
        else
            echo "   ⚠️  Package list update had warnings, continuing..."
        fi
        
        # Install nvidia-container-toolkit
        echo "   Installing nvidia-container-toolkit..."
        if sudo apt-get install -y nvidia-container-toolkit; then
            echo "   ✅ nvidia-container-toolkit installed successfully"
        else
            echo "   ❌ Installation failed"
            echo ""
            echo "Please install manually:"
            echo "   Visit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
            exit 1
        fi
        ;;
    
    fedora|rhel|centos)
        echo "Installing for Fedora/RHEL/CentOS..."
        echo ""
        
        # Configure repository
        echo "   Configuring repository..."
        if curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
            sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null; then
            echo "   ✅ Repository configured"
        else
            echo "   ❌ Failed to configure repository"
            exit 1
        fi
        
        # Install
        echo "   Installing nvidia-container-toolkit..."
        if sudo dnf install -y nvidia-container-toolkit 2>/dev/null || \
           sudo yum install -y nvidia-container-toolkit 2>/dev/null; then
            echo "   ✅ nvidia-container-toolkit installed successfully"
        else
            echo "   ❌ Installation failed"
            exit 1
        fi
        ;;
    
    arch|manjaro)
        echo "Installing for Arch/Manjaro..."
        echo ""
        
        if sudo pacman -S --noconfirm nvidia-container-toolkit; then
            echo "   ✅ nvidia-container-toolkit installed successfully"
        else
            echo "   ❌ Installation failed"
            exit 1
        fi
        ;;
    
    *)
        echo "❌ Unsupported distribution: $OS"
        echo ""
        echo "Please install manually:"
        echo "   Visit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
        exit 1
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 Configuring Docker Runtime"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Verify installation
if ! command_exists nvidia-ctk; then
    echo "❌ nvidia-ctk not found after installation"
    echo "   Installation may have failed"
    exit 1
fi

echo "✅ nvidia-ctk found"
echo ""

# Configure Docker runtime
echo "Configuring Docker to use NVIDIA runtime..."
if sudo nvidia-ctk runtime configure --runtime=docker --set-as-default; then
    echo "✅ NVIDIA runtime configured successfully"
else
    echo "❌ Failed to configure NVIDIA runtime"
    echo ""
    echo "Please configure manually:"
    echo "   sudo nvidia-ctk runtime configure --runtime=docker --set-as-default"
    exit 1
fi

echo ""
echo "🔄 Restarting Docker service..."
if sudo systemctl restart docker 2>/dev/null || sudo service docker restart 2>/dev/null; then
    echo "✅ Docker service restarted"
else
    echo "⚠️  Failed to restart Docker service automatically"
    echo ""
    echo "Please restart Docker manually:"
    echo "   sudo systemctl restart docker"
    echo "   or"
    echo "   sudo service docker restart"
    echo ""
    read -p "Press Enter after restarting Docker..."
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Testing GPU Access"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Testing GPU access in Docker container..."
if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    echo "✅ GPU access test successful!"
    echo ""
    echo "Your Docker containers can now access NVIDIA GPUs."
    echo ""
    if command -v deepiri-gpu >/dev/null 2>&1; then
        echo "Canonical Docker Compose GPU hints (from deepiri-gpu-utils):"
        echo "   deepiri-gpu compose-gpu          # human-readable"
        echo "   deepiri-gpu compose-gpu --json   # machine-readable"
        echo ""
    fi
    echo "Manual docker-compose.yml example (NVIDIA fallback):"
    echo "  deploy:"
    echo "    resources:"
    echo "      reservations:"
    echo "        devices:"
    echo "          - driver: nvidia"
    echo "            count: all"
    echo "            capabilities: [gpu]"
else
    echo "⚠️  GPU access test failed"
    echo ""
    echo "This may indicate:"
    echo "  1. Docker service needs to be restarted"
    echo "  2. NVIDIA drivers need to be reinstalled"
    echo "  3. System needs to be rebooted"
    echo ""
    echo "Try restarting Docker: sudo systemctl restart docker"
    echo "Or reboot your system and test again."
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Installation Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

