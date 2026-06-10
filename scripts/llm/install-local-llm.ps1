# Install and setup local LLM (Ollama) natively on Windows
# Simplified version for Windows PowerShell

$ErrorActionPreference = "Stop"

$OLLAMA_PORT = 11434
$OLLAMA_URL = "http://localhost:$OLLAMA_PORT"

Write-Host "Installing Local LLM (Ollama) for Cyrex" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Gray
Write-Host ""

# Function to check if command exists
function Test-Command {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

# Function to find Ollama executable path
function Get-OllamaPath {
    # First try if it's in PATH
    if (Test-Command "ollama") {
        $cmd = Get-Command "ollama" -ErrorAction SilentlyContinue
        if ($cmd) {
            return $cmd.Source
        }
    }
    
    # Common installation paths
    $possiblePaths = @(
        "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
        "$env:ProgramFiles\Ollama\ollama.exe",
        "$env:ProgramFiles(x86)\Ollama\ollama.exe",
        "$env:USERPROFILE\AppData\Local\Programs\Ollama\ollama.exe"
    )
    
    foreach ($path in $possiblePaths) {
        if (Test-Path $path) {
            return $path
        }
    }
    
    return $null
}

# GPU Detection
Write-Host "GPU Detection" -ForegroundColor Yellow
Write-Host ("=" * 80) -ForegroundColor Gray
Write-Host ""

$HAS_NVIDIA_GPU = $false

# Check nvidia-smi
if (Test-Command "nvidia-smi") {
    try {
        $output = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>&1
        if (($LASTEXITCODE -eq 0) -and $output) {
            Write-Host "[OK] NVIDIA GPU detected" -ForegroundColor Green
            Write-Host $output
            Write-Host ""
            $HAS_NVIDIA_GPU = $true
        }
    } catch {
        # Continue to next method
    }
}

# Check WMI if nvidia-smi didn't work
if (-not $HAS_NVIDIA_GPU) {
    try {
        $gpus = Get-WmiObject Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "*NVIDIA*" }
        if ($gpus) {
            Write-Host "[OK] Found $($gpus.Count) NVIDIA GPU(s)" -ForegroundColor Green
            foreach ($gpu in $gpus) {
                Write-Host "   - $($gpu.Name)" -ForegroundColor Gray
            }
            Write-Host ""
            $HAS_NVIDIA_GPU = $true
        }
    } catch {
        # Continue
    }
}

if (-not $HAS_NVIDIA_GPU) {
    Write-Host "[WARN] No NVIDIA GPU detected - will use CPU" -ForegroundColor Yellow
    Write-Host ""
}

# Check if Ollama is already installed
Write-Host "Checking Ollama Installation" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Gray
Write-Host ""

$OLLAMA_INSTALLED = $false
$OLLAMA_PATH = Get-OllamaPath

if ($OLLAMA_PATH) {
    Write-Host "[OK] Ollama is already installed at: $OLLAMA_PATH" -ForegroundColor Green
    $OLLAMA_INSTALLED = $true
} else {
    Write-Host "[INFO] Ollama not found. Will install now..." -ForegroundColor Yellow
    Write-Host ""
}

# Install Ollama if needed
if (-not $OLLAMA_INSTALLED) {
    Write-Host "Installing Ollama" -ForegroundColor Cyan
    Write-Host ("=" * 80) -ForegroundColor Gray
    Write-Host ""
    
    # Try winget first (Windows Package Manager)
    if (Test-Command "winget") {
        Write-Host "[INFO] Installing Ollama using winget..." -ForegroundColor Cyan
        try {
            winget install Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[OK] Ollama installed successfully via winget" -ForegroundColor Green
                $OLLAMA_INSTALLED = $true
            } else {
                Write-Host "[WARN] winget installation failed, trying direct download..." -ForegroundColor Yellow
            }
        } catch {
            Write-Host "[WARN] winget installation failed, trying direct download..." -ForegroundColor Yellow
        }
    }
    
    # If winget failed or not available, try direct download
    if (-not $OLLAMA_INSTALLED) {
        Write-Host "[INFO] Attempting direct download from GitHub..." -ForegroundColor Cyan
        
        # Try to get the latest download URL (Ollama uses GitHub releases)
        $installerUrl = "https://github.com/ollama/ollama/releases/latest/download/OllamaSetup.exe"
        $tempPath = $env:TEMP
        $installerPath = Join-Path $tempPath "OllamaSetup.exe"
        
        try {
            Write-Host "[INFO] Downloading Ollama installer..." -ForegroundColor Gray
            Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing -ErrorAction Stop
            
            Write-Host "[INFO] Running installer (this may take a moment)..." -ForegroundColor Cyan
            Write-Host "[INFO] The installer will run silently" -ForegroundColor Gray
            
            # Run installer silently
            $null = Start-Process -FilePath $installerPath -ArgumentList "/S" -Wait -PassThru -NoNewWindow
            
            # Clean up installer
            Start-Sleep -Seconds 1
            Remove-Item $installerPath -ErrorAction SilentlyContinue
            
            # Refresh PATH
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
            
            # Wait a bit for installation to complete
            Start-Sleep -Seconds 5
            
            # Check if installation succeeded
            $OLLAMA_PATH = Get-OllamaPath
            if ($OLLAMA_PATH) {
                Write-Host "[OK] Ollama installed successfully at: $OLLAMA_PATH" -ForegroundColor Green
                $OLLAMA_INSTALLED = $true
            } else {
                Write-Host "[WARN] Installation may have completed but ollama command not found yet" -ForegroundColor Yellow
                Write-Host "[INFO] This usually means you need to restart your terminal" -ForegroundColor Yellow
                Write-Host "[INFO] Ollama is typically installed at: C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama" -ForegroundColor Gray
                Write-Host ""
                Write-Host "Please:" -ForegroundColor Cyan
                Write-Host "  1. Close this terminal" -ForegroundColor White
                Write-Host "  2. Open a new terminal" -ForegroundColor White
                Write-Host "  3. Run this script again" -ForegroundColor White
                Write-Host ""
                exit 0
            }
        } catch {
            Write-Host "[ERROR] Failed to download/install Ollama automatically: $_" -ForegroundColor Red
            Write-Host ""
            Write-Host "Please install Ollama manually:" -ForegroundColor Cyan
            Write-Host "  1. Download from: https://ollama.com/download/windows" -ForegroundColor White
            Write-Host "  2. Run the installer" -ForegroundColor White
            Write-Host "  3. Restart your terminal" -ForegroundColor White
            Write-Host "  4. Run this script again" -ForegroundColor White
            Write-Host ""
            exit 1
        }
    }
    Write-Host ""
}

# Start Ollama service
Write-Host "Starting Ollama Service" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Gray
Write-Host ""

# Ensure we have the Ollama path
if (-not $OLLAMA_PATH) {
    $OLLAMA_PATH = Get-OllamaPath
}

if (-not $OLLAMA_PATH) {
    Write-Host "[ERROR] Ollama executable not found. Please restart your terminal and run this script again." -ForegroundColor Red
    exit 1
}

# Check if Ollama is already running
try {
    $response = Invoke-WebRequest -Uri "$OLLAMA_URL/api/tags" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "[OK] Ollama is already running" -ForegroundColor Green
        Write-Host ""
    }
} catch {
    Write-Host "[INFO] Starting Ollama service..." -ForegroundColor Cyan
    
    # Start Ollama in the background
    try {
        Start-Process -FilePath $OLLAMA_PATH -ArgumentList "serve" -WindowStyle Hidden
        Write-Host "[INFO] Waiting for Ollama to start..." -ForegroundColor Gray
        
        # Wait for Ollama to be ready (max 30 seconds)
        $MAX_WAIT = 30
        $WAIT_TIME = 0
        $ready = $false
        
        while ($WAIT_TIME -lt $MAX_WAIT) {
            try {
                $response = Invoke-WebRequest -Uri "$OLLAMA_URL/api/tags" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
                if ($response.StatusCode -eq 200) {
                    $ready = $true
                    break
                }
            } catch {
                # Not ready yet
            }
            Start-Sleep -Seconds 1
            $WAIT_TIME += 1
            Write-Host "   Waiting... ($WAIT_TIME/$MAX_WAIT seconds)" -ForegroundColor Gray
        }
        
        if ($ready) {
            Write-Host "[OK] Ollama is ready!" -ForegroundColor Green
        } else {
            Write-Host "[WARN] Ollama may not be fully ready yet" -ForegroundColor Yellow
            Write-Host "[INFO] You can check status with: ollama list" -ForegroundColor Gray
        }
    } catch {
        Write-Host "[ERROR] Failed to start Ollama: $_" -ForegroundColor Red
        Write-Host "[INFO] Try running 'ollama serve' manually in a separate terminal" -ForegroundColor Yellow
    }
}
Write-Host ""

# Check Models
Write-Host "Checking Models" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Gray
Write-Host ""

# Ensure we have the Ollama path
if (-not $OLLAMA_PATH) {
    $OLLAMA_PATH = Get-OllamaPath
}

$MODELS = ""
if ($OLLAMA_PATH) {
    try {
        $MODELS = & $OLLAMA_PATH list 2>&1
        if ($LASTEXITCODE -ne 0) {
            $MODELS = ""
        }
    } catch {
        $MODELS = ""
    }
}

$MODEL_LINES = $MODELS -split "`n" | Where-Object { ($_ -notmatch "^NAME") -and ($_.Trim() -ne "") }
$MODEL_COUNT = ($MODEL_LINES | Measure-Object).Count

if ($MODEL_COUNT -eq 0) {
    Write-Host "[WARN] No models found" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "RECOMMENDED: llama3:8b (4.7GB) - Default for this project" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Available models:" -ForegroundColor Cyan
    Write-Host "  1) llama3:8b (4.7GB) - DEFAULT"
    Write-Host "  2) llama3.2:1b (1.3GB)"
    Write-Host "  3) llama3.2:3b (2.0GB)"
    Write-Host "  4) mistral:7b (4.1GB)"
    Write-Host "  5) gemma2:2b (1.4GB)"
    Write-Host "  6) phi3:mini (2.3GB)"
    Write-Host "  7) codellama:7b (3.8GB)"
    Write-Host "  8) Custom model name"
    Write-Host "  9) Skip"
    Write-Host ""
    
    $selection = Read-Host "Enter model number (or comma-separated for multiple)"
    
    $MODEL_MAP = @{
        "1" = "llama3:8b"
        "2" = "llama3.2:1b"
        "3" = "llama3.2:3b"
        "4" = "mistral:7b"
        "5" = "gemma2:2b"
        "6" = "phi3:mini"
        "7" = "codellama:7b"
    }
    
    $MODELS_TO_PULL = @()
    $SELECTED = $selection -split ',' | ForEach-Object { $_.Trim() }
    
    foreach ($num in $SELECTED) {
        if ($num -eq "8") {
            $customModel = Read-Host "Enter custom model name"
            if ($customModel) {
                $MODELS_TO_PULL += $customModel
            }
        } elseif ($num -eq "9") {
            Write-Host "[INFO] Skipping installation" -ForegroundColor Yellow
            break
        } elseif ($MODEL_MAP.ContainsKey($num)) {
            $MODELS_TO_PULL += $MODEL_MAP[$num]
        }
    }
    
    if ($MODELS_TO_PULL.Count -gt 0) {
        Write-Host ""
        Write-Host "[INFO] Pulling $($MODELS_TO_PULL.Count) model(s)..." -ForegroundColor Cyan
        Write-Host "[INFO] This may take several minutes depending on your internet connection..." -ForegroundColor Gray
        Write-Host ""
        
        foreach ($model in $MODELS_TO_PULL) {
            Write-Host "[INFO] Pulling: $model" -ForegroundColor Cyan
            if ($OLLAMA_PATH) {
                & $OLLAMA_PATH pull $model
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "[OK] Success: $model" -ForegroundColor Green
                } else {
                    Write-Host "[ERROR] Failed: $model" -ForegroundColor Red
                }
            } else {
                Write-Host "[ERROR] Ollama path not found" -ForegroundColor Red
            }
            Write-Host ""
        }
    }
} else {
    Write-Host "[OK] Found $MODEL_COUNT model(s):" -ForegroundColor Green
    Write-Host $MODELS
    Write-Host ""
    if ($OLLAMA_PATH) {
        Write-Host "[INFO] To add more: $OLLAMA_PATH pull [model-name]" -ForegroundColor Gray
    } else {
        Write-Host "[INFO] To add more: ollama pull [model-name]" -ForegroundColor Gray
    }
    Write-Host ""
}

# Summary
Write-Host ("=" * 80) -ForegroundColor Gray
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Gray
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "   URL: $OLLAMA_URL"
Write-Host "   API: $OLLAMA_URL/api"
Write-Host ""
Write-Host "Commands:" -ForegroundColor Cyan
if ($OLLAMA_PATH) {
    Write-Host "   List models: $OLLAMA_PATH list"
    Write-Host "   Pull model: $OLLAMA_PATH pull [model-name]"
    Write-Host "   Run model: $OLLAMA_PATH run [model-name]"
} else {
    Write-Host "   List models: ollama list"
    Write-Host "   Pull model: ollama pull [model-name]"
    Write-Host "   Run model: ollama run [model-name]"
}
Write-Host "   Stop service: Stop-Process -Name ollama -Force"
Write-Host ""
Write-Host "[INFO] Ollama runs as a background service. It will start automatically on boot." -ForegroundColor Gray
Write-Host "[INFO] To stop it, use: Stop-Process -Name ollama -Force" -ForegroundColor Gray
Write-Host ""
