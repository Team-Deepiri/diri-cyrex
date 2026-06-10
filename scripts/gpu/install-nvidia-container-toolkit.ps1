# Install NVIDIA Container Toolkit for Docker GPU support (Windows/WSL2)
# This allows Docker containers to access NVIDIA GPUs

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "🚀 NVIDIA Container Toolkit Installation" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

# Check if running in WSL2
$isWSL = $false
if (Test-Path "/proc/version") {
    $procVersion = Get-Content "/proc/version" -Raw
    if ($procVersion -match "Microsoft|WSL") {
        $isWSL = $true
    }
}

if (-not $isWSL) {
    Write-Host "⚠️  This script should be run in WSL2, not Windows PowerShell" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "For Windows with WSL2:" -ForegroundColor Yellow
    Write-Host "  1. Open WSL2 terminal (Ubuntu, etc.)"
    Write-Host "  2. Run: bash scripts/install-nvidia-container-toolkit.sh"
    Write-Host ""
    Write-Host "For Windows with Docker Desktop:" -ForegroundColor Yellow
    Write-Host "  • GPU support requires WSL2 backend"
    Write-Host "  • Install NVIDIA drivers in WSL2"
    Write-Host "  • Then run the bash script in WSL2"
    Write-Host ""
    
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        exit 1
    }
}

# Check if Docker is installed
try {
    $dockerVersion = docker --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Docker found: $dockerVersion" -ForegroundColor Green
    } else {
        throw "Docker not found"
    }
} catch {
    Write-Host "❌ Docker is not installed" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Docker Desktop for Windows:"
    Write-Host "  https://docs.docker.com/desktop/install/windows-install/"
    exit 1
}

Write-Host ""

# Check if NVIDIA drivers are installed (in WSL2)
if ($isWSL) {
    try {
        $nvidiaSmi = nvidia-smi 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ NVIDIA drivers detected" -ForegroundColor Green
            Write-Host ""
            Write-Host "GPU Information:" -ForegroundColor Cyan
            nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader,nounits
            Write-Host ""
        } else {
            throw "nvidia-smi not found"
        }
    } catch {
        Write-Host "⚠️  nvidia-smi not found - NVIDIA drivers may not be installed" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Please install NVIDIA drivers in WSL2 first:"
        Write-Host "  • Download from: https://www.nvidia.com/Download/index.aspx"
        Write-Host "  • Or use: sudo apt-get install nvidia-driver-535"
        Write-Host ""
        
        $continue = Read-Host "Continue anyway? (y/n)"
        if ($continue -ne "y" -and $continue -ne "Y") {
            exit 1
        }
        Write-Host ""
    }
}

# Check if already installed
try {
    $nvidiaCtk = nvidia-ctk --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ NVIDIA Container Toolkit is already installed" -ForegroundColor Green
        Write-Host ""
        Write-Host "Current version:"
        nvidia-ctk --version
        Write-Host ""
        
        # Test GPU access
        Write-Host "🧪 Testing GPU access in Docker..." -ForegroundColor Cyan
        docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ GPU access test successful!" -ForegroundColor Green
            Write-Host ""
            Write-Host "Your Docker containers can now access NVIDIA GPUs." -ForegroundColor Green
        } else {
            Write-Host "⚠️  GPU access test failed" -ForegroundColor Yellow
            Write-Host "   This may indicate a configuration issue"
        }
        exit 0
    }
} catch {
    # Not installed, continue
}

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "ℹ️  About NVIDIA Container Toolkit" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "The NVIDIA Container Toolkit allows Docker containers to access your GPU."
Write-Host "Without it, GPU-accelerated containers will run on CPU only, which is"
Write-Host "10-50x slower for AI workloads."
Write-Host ""
Write-Host "Benefits:"
Write-Host "  ✅ Ollama and other AI tools will automatically use your GPU"
Write-Host "  ✅ GPU acceleration works in all Docker containers"
Write-Host "  ✅ Better performance: 20-100+ tokens/sec (GPU) vs 2-5 tokens/sec (CPU)"
Write-Host "  ✅ Uses GPU VRAM instead of system RAM for models"
Write-Host ""
Write-Host "This is a one-time setup. After installation, Docker will automatically"
Write-Host "configure containers to use your NVIDIA GPU when available."
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

$continue = Read-Host "Continue with installation? (y/n)"
if ($continue -ne "y" -and $continue -ne "Y") {
    Write-Host "Installation cancelled."
    exit 0
}

Write-Host ""
Write-Host "⚠️  PowerShell version is limited - please use the bash script in WSL2" -ForegroundColor Yellow
Write-Host ""
Write-Host "For proper installation, run this in WSL2:" -ForegroundColor Cyan
Write-Host "  bash scripts/install-nvidia-container-toolkit.sh" -ForegroundColor White
Write-Host ""
Write-Host "The bash script supports:" -ForegroundColor Cyan
Write-Host "  • Automatic distribution detection (Ubuntu/Debian, Fedora, Arch, etc.)"
Write-Host "  • Package repository configuration"
Write-Host "  • Docker runtime configuration"
Write-Host "  • Automatic Docker service restart"
Write-Host "  • GPU access testing"
Write-Host ""
Write-Host "Manual installation guide:" -ForegroundColor Cyan
Write-Host "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" -ForegroundColor White
Write-Host ""

