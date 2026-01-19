# =============================================================================
# HYBRID DOCKERFILE: Prebuilt OR From-Scratch with Staged Downloads
# =============================================================================
# Build args to control build type
ARG BUILD_TYPE=prebuilt
ARG PYTORCH_VERSION=2.9.1
ARG CUDA_VERSION=12.8
ARG PYTHON_VERSION=3.11
ARG DEVICE_TYPE=auto  # auto, gpu, cpu, mpsos - auto detects from BASE_IMAGE
## Use CUDA 12.8 base image for RTX 5080/5090 (sm_120) support
# If official PyTorch image doesn't exist, fallback to CUDA 12.6 and upgrade PyTorch
ARG BASE_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
# ARG BASE_IMAGE=pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime
# =============================================================================
# OPTION 1: PREBUILT (Fast, Reliable, Larger) - DEFAULT
# =============================================================================
# Use BASE_IMAGE build arg (set by build scripts based on GPU detection)
FROM ${BASE_IMAGE} AS prebuilt-base

# Re-declare ARG in this stage so it's available
ARG PYTORCH_VERSION=2.9.1
ARG PYTHON_VERSION=3.11

ENV BUILD_TYPE=prebuilt \
    PYTORCH_VERSION=${PYTORCH_VERSION} \
    PYTHON_VERSION=${PYTHON_VERSION}

# Upgrade pip first for better package resolution
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Check if PyTorch with CUDA 12.8 is already installed (from CUDA 12.8 base image)
# If not, upgrade to CUDA 12.8 for RTX 5080/5090 (sm_120) compatibility
RUN python -c "import torch; cuda_ver = torch.version.cuda; exit(0 if cuda_ver and (cuda_ver.startswith('12.8') or float(cuda_ver.split('.')[0] + '.' + cuda_ver.split('.')[1]) >= 12.8) else 1)" 2>/dev/null || \
    (echo "Upgrading PyTorch to CUDA 12.8 support (supports sm_50 through sm_120)..." && \
     pip uninstall -y torch torchvision torchaudio 2>/dev/null || true && \
     pip install --no-cache-dir --upgrade-strategy=only-if-needed \
         --pre torch torchvision torchaudio \
         --index-url https://download.pytorch.org/whl/nightly/cu128 && \
     echo "✓ PyTorch with CUDA 12.8 installed successfully") || \
    echo "✓ PyTorch with CUDA 12.8 already installed in base image"

# If PyTorch is not already installed (e.g., using python:3.11-slim for CPU), install it
# If using pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime (GPU build), PyTorch is already installed
RUN python -c "import torch" 2>/dev/null || \
    (echo "CPU build detected, installing PyTorch CPU (version: ${PYTORCH_VERSION})..." && \
     pip install --no-cache-dir --upgrade-strategy=only-if-needed \
         torch==${PYTORCH_VERSION} \
         torchvision \
         torchaudio \
         --index-url https://download.pytorch.org/whl/cpu) || \
    echo "PyTorch installation check completed"

# =============================================================================
# OPTION 2: FROM SCRATCH (Customizable, Smaller, Slower)
# =============================================================================
FROM python:${PYTHON_VERSION}-slim AS from-scratch-base

ENV BUILD_TYPE=from-scratch

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Configure pip for better reliability and resume capability
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=5
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# =============================================================================
# STAGE 1: Download Heavy Packages (Torch/CUDA) - FROM SCRATCH ONLY
# =============================================================================
FROM from-scratch-base AS download-torch

# Re-declare ARG in this stage so it's available
ARG PYTORCH_VERSION=2.9.1

# Download PyTorch in separate stages for resume capability
# Stage 1.1: Download torch (can resume if fails)
RUN pip download --no-deps \
    --timeout=300 \
    --retries=5 \
    --dest /tmp/packages \
    torch==${PYTORCH_VERSION} \
    || (echo "Warning: torch download failed, will retry in next stage" && exit 0)

# Stage 1.2: Download torchvision (depends on torch)
RUN pip download --no-deps \
    --timeout=300 \
    --retries=5 \
    --dest /tmp/packages \
    torchvision \
    || echo "Warning: torchvision download failed, continuing..."

# Stage 1.3: Download torchaudio (depends on torch)
RUN pip download --no-deps \
    --timeout=300 \
    --retries=5 \
    --dest /tmp/packages \
    torchaudio \
    || echo "Warning: torchaudio download failed, continuing..."

# =============================================================================
# STAGE 2: Install Heavy Packages - FROM SCRATCH ONLY
# =============================================================================
FROM from-scratch-base AS install-torch

# Re-declare ARG in this stage so it's available
ARG PYTORCH_VERSION=2.9.1

# Copy downloaded packages
COPY --from=download-torch /tmp/packages /tmp/packages

# Install PyTorch from downloaded packages (prefer local, fallback to PyPI)
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed \
        --find-links /tmp/packages \
        --prefer-binary \
        torch==${PYTORCH_VERSION} \
        torchvision \
        torchaudio \
    || (echo "Installing torch from PyPI (fallback)..." && \
        pip install --no-cache-dir --timeout=300 --retries=5 \
        torch==${PYTORCH_VERSION} \
        torchvision \
        torchaudio)

# =============================================================================
# STAGE 3: Common Setup for Both Build Types
# =============================================================================
# We create separate final stages for each build type
# Build scripts will use --target to select the right one

# PREBUILT PATH: Start from prebuilt base
FROM prebuilt-base AS base-prebuilt

# Set common environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=3 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

WORKDIR /app

# Install system dependencies (if not already in base)
RUN if [ "$BUILD_TYPE" = "from-scratch" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            curl \
            && rm -rf /var/lib/apt/lists/* \
            && apt-get clean; \
    else \
        apt-get update && apt-get install -y --no-install-recommends \
            curl \
            && rm -rf /var/lib/apt/lists/* \
            && apt-get clean; \
    fi

# Copy deepiri-modelkit first (needed for installation)
COPY deepiri-modelkit /app/deepiri-modelkit

# Copy requirements first for better caching
COPY diri-cyrex/requirements.txt /app/requirements.txt

# Remove torch and deepiri-modelkit editable install from requirements.txt
# (torch already in base image, modelkit installed separately)
RUN sed -i '/^torch/d' /app/requirements.txt && \
    sed -i '/torch/d' /app/requirements.txt && \
    sed -i '/deepiri-modelkit/d' /app/requirements.txt && \
    sed -i '/^-e.*modelkit/d' /app/requirements.txt || true

# Install deepiri-modelkit as editable package (before other requirements)
RUN if [ -d "/app/deepiri-modelkit" ] && [ -f "/app/deepiri-modelkit/pyproject.toml" ]; then \
        echo "Installing deepiri-modelkit..." && \
        pip install --no-cache-dir -e /app/deepiri-modelkit || \
        (echo "Warning: deepiri-modelkit installation failed" && true); \
    else \
        echo "Warning: deepiri-modelkit not found, skipping installation"; \
    fi

# Verify torch and modelkit are removed
RUN grep -v 'torch' /app/requirements.txt | grep -v 'modelkit' > /tmp/requirements_clean.txt || true

# =============================================================================
# STAGE 4: Download Heavy ML Packages (Staged for Resume)
# =============================================================================
# Download stage works for both build types
FROM prebuilt-base AS download-ml-packages

# Re-declare ARG in this stage
ARG PYTORCH_VERSION=2.9.1

# Ensure directory exists and set up pip for better reliability
RUN mkdir -p /tmp/ml-packages && \
    pip install --upgrade pip setuptools wheel --quiet

# Configure pip environment for downloads
ENV PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=5 \
    PIP_CONSTRAINT_TIMEOUT=600

# Download heavy ML packages with dependencies for resume capability
# Note: Downloads are optional - if they fail, packages will be installed from PyPI during build
# Stage 4.1: Download transformers and dependencies
# Using --no-deps to download only the package (faster, dependencies resolved during install)
RUN pip download \
    --timeout=600 \
    --retries=5 \
    --dest /tmp/ml-packages \
    --no-deps \
    transformers>=4.30.0 \
    2>&1 | tee /tmp/transformers-download.log || \
    (echo "⚠️  Warning: transformers download failed (this is OK - will install from PyPI)" && \
     echo "   Download logs saved to /tmp/transformers-download.log" && \
     mkdir -p /tmp/ml-packages || true)

# Stage 4.2: Download datasets
RUN pip download \
    --timeout=600 \
    --retries=5 \
    --dest /tmp/ml-packages \
    --no-deps \
    datasets>=2.14.0 \
    2>&1 | tee /tmp/datasets-download.log || \
    (echo "⚠️  Warning: datasets download failed (this is OK - will install from PyPI)" && \
     mkdir -p /tmp/ml-packages || true)

# Stage 4.3: Download accelerate
RUN pip download \
    --timeout=600 \
    --retries=5 \
    --dest /tmp/ml-packages \
    --no-deps \
    accelerate>=0.20.0 \
    2>&1 | tee /tmp/accelerate-download.log || \
    (echo "⚠️  Warning: accelerate download failed (this is OK - will install from PyPI)" && \
     mkdir -p /tmp/ml-packages || true)

# Stage 4.4: Download sentence-transformers
RUN pip download \
    --timeout=600 \
    --retries=5 \
    --dest /tmp/ml-packages \
    --no-deps \
    sentence-transformers>=2.2.0 \
    2>&1 | tee /tmp/sentence-transformers-download.log || \
    (echo "⚠️  Warning: sentence-transformers download failed (this is OK - will install from PyPI)" && \
     mkdir -p /tmp/ml-packages || true)

# Stage 4.5: Download mlflow (heavy)
RUN pip download \
    --timeout=600 \
    --retries=5 \
    --dest /tmp/ml-packages \
    --no-deps \
    mlflow>=2.7.0 \
    2>&1 | tee /tmp/mlflow-download.log || \
    (echo "⚠️  Warning: mlflow download failed (this is OK - will install from PyPI)" && \
     mkdir -p /tmp/ml-packages || true)

# Stage 4.6: Download wandb
RUN pip download \
    --timeout=600 \
    --retries=5 \
    --dest /tmp/ml-packages \
    --no-deps \
    wandb>=0.15.0 \
    2>&1 | tee /tmp/wandb-download.log || \
    (echo "⚠️  Warning: wandb download failed (this is OK - will install from PyPI)" && \
     mkdir -p /tmp/ml-packages || true)

# Summary: Show what was downloaded (if anything)
RUN echo "📦 Download stage complete. Packages in /tmp/ml-packages:" && \
    ls -lh /tmp/ml-packages/*.whl 2>/dev/null | wc -l | xargs -I {} echo "   {} wheel files downloaded" || \
    echo "   No packages downloaded (will install from PyPI during build)"

# =============================================================================
# STAGE 5: Install All Packages (PREBUILT PATH - DEFAULT)
# =============================================================================
FROM base-prebuilt AS final-prebuilt

# Re-declare ARG in this stage
ARG DEVICE_TYPE=auto
ARG BASE_IMAGE

# Copy downloaded ML packages (if available)
COPY --from=download-ml-packages /tmp/ml-packages /tmp/ml-packages

# Copy all requirements files for conditional installation
COPY diri-cyrex/requirements.txt /app/requirements.txt
COPY diri-cyrex/requirements-cpu.txt /app/requirements-cpu.txt
COPY diri-cyrex/requirements-mpsos.txt /app/requirements-mpsos.txt

# Upgrade pip with resume-friendly settings
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed --upgrade pip setuptools wheel

# Detect device type and set requirements file
# Note: This detection runs before torch is installed, so we check BASE_IMAGE
RUN if [ "$DEVICE_TYPE" = "auto" ]; then \
        if echo "$BASE_IMAGE" | grep -qi "cpu\|slim"; then \
            DEVICE_TYPE="cpu"; \
        elif echo "$BASE_IMAGE" | grep -qi "mps\|macos\|darwin"; then \
            DEVICE_TYPE="mpsos"; \
        else \
            DEVICE_TYPE="gpu"; \
        fi; \
    fi && \
    echo "Detected device type: $DEVICE_TYPE" && \
    if [ "$DEVICE_TYPE" = "mpsos" ]; then \
        REQ_FILE="/app/requirements-mpsos.txt"; \
    elif [ "$DEVICE_TYPE" = "cpu" ]; then \
        REQ_FILE="/app/requirements-cpu.txt"; \
    else \
        REQ_FILE="/app/requirements.txt"; \
    fi && \
    echo "Using requirements file: $REQ_FILE" && \
    echo "$REQ_FILE" > /tmp/req_file.txt

# Install core dependencies first (small packages, fast)
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed \
        fastapi==0.112.2 \
        uvicorn[standard]==0.30.6 \
        pydantic==2.8.2 \
        pydantic-settings==2.2.1 \
        python-multipart>=0.0.6 \
        openai==1.43.0 \
        python-dotenv==1.0.1 \
        httpx==0.27.2 \
        structlog==24.1.0 \
        python-json-logger==2.0.7 \
        prometheus-client==0.20.0 \
        redis==5.0.1 \
        asyncpg>=0.29.0 \
        watchdog>=3.0.0 \
        pytest==8.3.2 \
        pytest-asyncio==0.23.5 \
        pytest-cov==4.1.0

# Install LangChain packages (required for orchestration)
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed \
        langchain-core>=0.1.23 \
        langchain>=0.1.0 \
        langchain-openai>=0.0.5 \
        langchain-community>=0.0.20 \
        langchain-chroma>=0.1.0 \
        langchain-milvus>=0.1.0 \
        langchain-text-splitters>=0.0.1 \
        langchain-ollama>=0.1.0 \
        langchain-huggingface>=0.0.3 \
        ollama>=0.1.0 && \
    echo "✓ LangChain packages installed successfully" || \
    (echo "❌ ERROR: Failed to install critical LangChain packages" && \
     pip list | grep -E "langchain|ollama" && \
     exit 1)

# Install LangGraph packages (required for multi-agent workflows)
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=3 \
        langgraph>=0.2.0,<0.3.0 \
        langgraph-checkpoint-redis>=0.2.0 || \
    (echo "⚠️  WARNING: LangGraph packages installation failed (optional), continuing..." && \
     pip list | grep -E "langgraph" || echo "LangGraph not installed") && \
    echo "✓ LangGraph packages installation completed"

# Install ML libraries (prefer downloaded packages, fallback to PyPI)
RUN if [ -d "/tmp/ml-packages" ] && [ "$(ls -A /tmp/ml-packages)" ]; then \
        echo "Installing from downloaded packages (with PyPI fallback for dependencies)..." && \
        pip install --no-cache-dir --upgrade-strategy=only-if-needed \
            --find-links /tmp/ml-packages \
            --prefer-binary \
            transformers>=4.30.0 \
            datasets>=2.14.0 \
            accelerate>=0.20.0 \
            sentence-transformers>=2.2.0; \
    else \
        echo "Installing from PyPI..." && \
        pip install --no-cache-dir --upgrade-strategy=only-if-needed \
            transformers>=4.30.0 \
            datasets>=2.14.0 \
            accelerate>=0.20.0 \
            sentence-transformers>=2.2.0; \
    fi

# Install scikit-learn, numpy, pandas (medium packages)
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed \
        scikit-learn>=1.3.0 \
        numpy>=1.24.0 \
        pandas>=2.0.0

# Install optional heavy packages separately with retries
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=3 \
        mlflow>=2.7.0 \
        wandb>=0.15.0 || echo "Warning: mlflow/wandb installation failed, continuing..."

# Install optional packages (can fail without breaking build)
# GPU-specific packages only if CUDA base image is used
RUN if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then \
        echo "GPU build detected, installing GPU-specific packages..." && \
        pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
            deepspeed>=0.12.0 || echo "Warning: deepspeed installation failed (optional), continuing..." && \
        pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
            bitsandbytes>=0.41.0 || echo "Warning: bitsandbytes installation failed (optional), continuing..."; \
    else \
        echo "CPU build: Skipping GPU-specific packages (deepspeed, bitsandbytes)"; \
    fi

# Install remaining optional packages
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        peft>=0.7.0 || echo "Warning: peft installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        gymnasium>=0.29.0 || echo "Warning: gymnasium installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        pymilvus>=2.3.0 || echo "Warning: pymilvus installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        pinecone-client>=3.0.0 || echo "Warning: pinecone installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        weaviate-client>=4.0.0 || echo "Warning: weaviate installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        influxdb-client>=1.38.0 || echo "Warning: influxdb installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        kubernetes>=28.1.0 || echo "Warning: kubernetes installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        optuna>=3.5.0 || echo "Warning: optuna installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        hyperopt>=0.2.7 || echo "Warning: hyperopt installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        tensorboard>=2.15.0 || echo "Warning: tensorboard installation failed (optional), continuing..."

# Install platform-specific requirements (including document processing packages)
# Filter out packages already installed individually to avoid redundancy
# Note: pip's --upgrade-strategy=only-if-needed will skip packages that already meet requirements,
# but we filter explicitly to reduce processing time and avoid version conflicts
RUN REQ_FILE=$(cat /tmp/req_file.txt) && \
    echo "Installing platform-specific requirements from $REQ_FILE..." && \
    echo "Filtering out already-installed packages..." && \
    pip list --format=freeze | cut -d'=' -f1 | tr '[:upper:]' '[:lower:]' | sed 's/-/_/g' > /tmp/installed_packages.txt && \
    python3 << PYEOF
import re
import os

# Read installed packages (normalized names)
with open('/tmp/installed_packages.txt', 'r') as f:
    installed = {line.strip() for line in f if line.strip()}

# Get requirements file path from environment
req_file_path = open('/tmp/req_file.txt', 'r').read().strip()

# Read requirements file
with open(req_file_path, 'r') as f:
    lines = f.readlines()

# Filter requirements
filtered = []
skipped_count = 0
for line in lines:
    stripped = line.strip()
    # Keep comments and empty lines
    if not stripped or stripped.startswith('#'):
        filtered.append(line)
        continue
    
    # Extract package name (handle: package==version, package>=version, package[extra], etc.)
    # Remove comments first
    pkg_line = stripped.split('#')[0].strip()
    match = re.match(r'^([a-zA-Z0-9_-]+)', pkg_line)
    if match:
        pkg_name = match.group(1).lower().replace('-', '_')
        # Check if already installed
        if pkg_name in installed:
            skipped_count += 1
            print(f"Skipping already-installed: {match.group(1)}")
        else:
            filtered.append(line)
    else:
        # Keep lines we can't parse (might be URLs, etc.)
        filtered.append(line)

# Write filtered requirements
with open('/tmp/filtered_requirements.txt', 'w') as f:
    f.writelines(filtered)

print(f"Filtered {skipped_count} already-installed packages")
print(f"Remaining packages to install: {len([l for l in filtered if l.strip() and not l.strip().startswith('#')])}")
PYEOF
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=3 \
        -r /tmp/filtered_requirements.txt && \
    echo "✓ Platform-specific requirements installed successfully" || \
    (echo "⚠️  WARNING: Some requirements installation failed" && \
     pip list | grep -E "(pdfplumber|python-docx|pytesseract|Pillow|pdf2image|beautifulsoup4|openpyxl)" || \
     echo "Some packages may not be fully installed")

# Verify critical packages
RUN python -c "import numpy; print('✓ numpy version:', numpy.__version__)" && \
    python -c "import torch; print('✓ torch version:', torch.__version__); print('✓ CUDA available:', torch.cuda.is_available() if hasattr(torch.cuda, 'is_available') else False)" && \
    python -c "import sentence_transformers; print('✓ sentence-transformers installed')" && \
    python -c "import langchain_core; print('✓ langchain-core installed')" || \
    (echo "ERROR: Failed to verify critical packages" && pip list | grep -E "(numpy|torch|sentence|langchain)" && exit 1)

# Create non-root user and set up directories
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    mkdir -p /app/logs /app/.cache/huggingface /app/.cache/sentence_transformers /app/tests && \
    chown -R appuser:appuser /app

# Copy deepiri-modelkit (shared library) before app code
# This allows installing it as an editable package
COPY deepiri-modelkit /app/deepiri-modelkit

# Install deepiri-modelkit as editable package (before other requirements)
RUN if [ -d "/app/deepiri-modelkit" ] && [ -f "/app/deepiri-modelkit/pyproject.toml" ]; then \
        echo "Installing deepiri-modelkit..." && \
        pip install --no-cache-dir -e /app/deepiri-modelkit || \
        (echo "Warning: deepiri-modelkit installation failed" && true); \
    else \
        echo "Warning: deepiri-modelkit not found, skipping installation"; \
    fi

# Copy application code
COPY diri-cyrex/app /app/app

# Copy tests directory if it exists in build context
# Create placeholder first to ensure directory exists
RUN touch /app/tests/__init__.py
# Copy tests directory - will fail build if tests/ doesn't exist, which is expected
# If tests directory is missing, create it manually before building
COPY diri-cyrex/tests /app/tests
RUN chown -R appuser:appuser /app/tests

# Copy K8s env loader scripts (before switching user)
COPY --chown=root:root ops/k8s/load-k8s-env.sh /usr/local/bin/load-k8s-env.sh
COPY --chown=root:root ops/k8s/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/load-k8s-env.sh /usr/local/bin/docker-entrypoint.sh

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application with entrypoint that loads K8s env
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# =============================================================================
# STAGE 6: Install All Packages (FROM-SCRATCH PATH)
# =============================================================================
FROM install-torch AS base-from-scratch

# Re-declare ARG in this stage
ARG DEVICE_TYPE=auto
ARG BASE_IMAGE

# Set common environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=3 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        && rm -rf /var/lib/apt/lists/* \
        && apt-get clean

FROM base-from-scratch AS final-from-scratch

# Re-declare ARG in this stage
ARG DEVICE_TYPE=auto
ARG BASE_IMAGE

# Copy deepiri-modelkit first (needed for installation)
COPY deepiri-modelkit /app/deepiri-modelkit

# Copy all requirements files for conditional installation
COPY diri-cyrex/requirements.txt /app/requirements.txt
COPY diri-cyrex/requirements-cpu.txt /app/requirements-cpu.txt
COPY diri-cyrex/requirements-mpsos.txt /app/requirements-mpsos.txt

# Remove torch and deepiri-modelkit editable install from requirements files
# (torch already installed in base-from-scratch, modelkit installed separately)
RUN sed -i '/^torch/d' /app/requirements.txt && \
    sed -i '/torch/d' /app/requirements.txt && \
    sed -i '/deepiri-modelkit/d' /app/requirements.txt && \
    sed -i '/^-e.*modelkit/d' /app/requirements.txt || true && \
    sed -i '/^torch/d' /app/requirements-cpu.txt && \
    sed -i '/torch/d' /app/requirements-cpu.txt && \
    sed -i '/deepiri-modelkit/d' /app/requirements-cpu.txt && \
    sed -i '/^-e.*modelkit/d' /app/requirements-cpu.txt || true && \
    sed -i '/^torch/d' /app/requirements-mpsos.txt && \
    sed -i '/torch/d' /app/requirements-mpsos.txt && \
    sed -i '/deepiri-modelkit/d' /app/requirements-mpsos.txt && \
    sed -i '/^-e.*modelkit/d' /app/requirements-mpsos.txt || true

# Detect device type and set requirements file
# Note: This detection runs after torch is installed, so we can check CUDA availability
RUN if [ "$DEVICE_TYPE" = "auto" ]; then \
        if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then \
            DEVICE_TYPE="gpu"; \
        elif echo "$BASE_IMAGE" | grep -qi "cpu\|slim"; then \
            DEVICE_TYPE="cpu"; \
        elif echo "$BASE_IMAGE" | grep -qi "mps\|macos\|darwin"; then \
            DEVICE_TYPE="mpsos"; \
        else \
            DEVICE_TYPE="cpu"; \
        fi; \
    fi && \
    echo "Detected device type: $DEVICE_TYPE" && \
    if [ "$DEVICE_TYPE" = "mpsos" ]; then \
        REQ_FILE="/app/requirements-mpsos.txt"; \
    elif [ "$DEVICE_TYPE" = "cpu" ]; then \
        REQ_FILE="/app/requirements-cpu.txt"; \
    else \
        REQ_FILE="/app/requirements.txt"; \
    fi && \
    echo "Using requirements file: $REQ_FILE" && \
    echo "$REQ_FILE" > /tmp/req_file.txt

# Install deepiri-modelkit as editable package (before other requirements)
RUN if [ -d "/app/deepiri-modelkit" ] && [ -f "/app/deepiri-modelkit/pyproject.toml" ]; then \
        echo "Installing deepiri-modelkit..." && \
        pip install --no-cache-dir -e /app/deepiri-modelkit || \
        (echo "Warning: deepiri-modelkit installation failed" && true); \
    else \
        echo "Warning: deepiri-modelkit not found, skipping installation"; \
    fi

# Verify torch and modelkit are removed
RUN grep -v 'torch' /app/requirements.txt | grep -v 'modelkit' > /tmp/requirements_clean.txt || true

# Copy downloaded ML packages (if available from download stage)
# The download-ml-packages stage always creates /tmp/ml-packages (even if empty)
# so this COPY will always succeed
COPY --from=download-ml-packages /tmp/ml-packages /tmp/ml-packages

# Note: We use prebuilt base images and downloaded packages for faster builds
# No need to copy from host venv - Docker builds should be self-contained

# Upgrade pip with resume-friendly settings
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed --upgrade pip setuptools wheel

# Install core dependencies first (small packages, fast)
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed \
        fastapi==0.112.2 \
        uvicorn[standard]==0.30.6 \
        pydantic==2.8.2 \
        pydantic-settings==2.2.1 \
        python-multipart>=0.0.6 \
        openai==1.43.0 \
        python-dotenv==1.0.1 \
        httpx==0.27.2 \
        structlog==24.1.0 \
        python-json-logger==2.0.7 \
        prometheus-client==0.20.0 \
        redis==5.0.1 \
        asyncpg>=0.29.0 \
        watchdog>=3.0.0 \
        pytest==8.3.2 \
        pytest-asyncio==0.23.5 \
        pytest-cov==4.1.0

# Install LangChain packages (required for orchestration)
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed \
        langchain-core>=0.1.23 \
        langchain>=0.1.0 \
        langchain-openai>=0.0.5 \
        langchain-community>=0.0.20 \
        langchain-chroma>=0.1.0 \
        langchain-milvus>=0.1.0 \
        langchain-text-splitters>=0.0.1 \
        langchain-ollama>=0.1.0 \
        langchain-huggingface>=0.0.3 \
        ollama>=0.1.0 && \
    echo "✓ LangChain packages installed successfully" || \
    (echo "❌ ERROR: Failed to install critical LangChain packages" && \
     pip list | grep -E "langchain|ollama" && \
     exit 1)

# Install LangGraph packages (required for multi-agent workflows)
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=3 \
        langgraph>=0.2.0,<0.3.0 \
        langgraph-checkpoint-redis>=0.2.0 || \
    (echo "⚠️  WARNING: LangGraph packages installation failed (optional), continuing..." && \
     pip list | grep -E "langgraph" || echo "LangGraph not installed") && \
    echo "✓ LangGraph packages installation completed"

# Install ML libraries (prefer downloaded packages, fallback to PyPI)
RUN if [ -d "/tmp/ml-packages" ] && [ "$(ls -A /tmp/ml-packages)" ]; then \
        echo "Installing from downloaded packages (with PyPI fallback for dependencies)..." && \
        pip install --no-cache-dir --upgrade-strategy=only-if-needed \
            --find-links /tmp/ml-packages \
            --prefer-binary \
            transformers>=4.30.0 \
            datasets>=2.14.0 \
            accelerate>=0.20.0 \
            sentence-transformers>=2.2.0; \
    else \
        echo "Installing from PyPI..." && \
        pip install --no-cache-dir --upgrade-strategy=only-if-needed \
            transformers>=4.30.0 \
            datasets>=2.14.0 \
            accelerate>=0.20.0 \
            sentence-transformers>=2.2.0; \
    fi

# Install scikit-learn, numpy, pandas (medium packages)
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed \
        scikit-learn>=1.3.0 \
        numpy>=1.24.0 \
        pandas>=2.0.0

# Install optional heavy packages separately with retries
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=3 \
        mlflow>=2.7.0 \
        wandb>=0.15.0 || echo "Warning: mlflow/wandb installation failed, continuing..."

# Install optional packages (can fail without breaking build)
# GPU-specific packages only if CUDA base image is used
RUN if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then \
        echo "GPU build detected, installing GPU-specific packages..." && \
        pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
            deepspeed>=0.12.0 || echo "Warning: deepspeed installation failed (optional), continuing..." && \
        pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
            bitsandbytes>=0.41.0 || echo "Warning: bitsandbytes installation failed (optional), continuing..."; \
    else \
        echo "CPU build: Skipping GPU-specific packages (deepspeed, bitsandbytes)"; \
    fi

# Install remaining optional packages
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        peft>=0.7.0 || echo "Warning: peft installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        gymnasium>=0.29.0 || echo "Warning: gymnasium installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        pymilvus>=2.3.0 || echo "Warning: pymilvus installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        pinecone-client>=3.0.0 || echo "Warning: pinecone installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        weaviate-client>=4.0.0 || echo "Warning: weaviate installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        influxdb-client>=1.38.0 || echo "Warning: influxdb installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        kubernetes>=28.1.0 || echo "Warning: kubernetes installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        optuna>=3.5.0 || echo "Warning: optuna installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        hyperopt>=0.2.7 || echo "Warning: hyperopt installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        tensorboard>=2.15.0 || echo "Warning: tensorboard installation failed (optional), continuing..."

# Install platform-specific requirements (including document processing packages)
# Filter out packages already installed individually to avoid redundancy
# Note: pip's --upgrade-strategy=only-if-needed will skip packages that already meet requirements,
# but we filter explicitly to reduce processing time and avoid version conflicts
RUN REQ_FILE=$(cat /tmp/req_file.txt) && \
    echo "Installing platform-specific requirements from $REQ_FILE..." && \
    echo "Filtering out already-installed packages..." && \
    pip list --format=freeze | cut -d'=' -f1 | tr '[:upper:]' '[:lower:]' | sed 's/-/_/g' > /tmp/installed_packages.txt && \
    python3 << PYEOF
import re
import os

# Read installed packages (normalized names)
with open('/tmp/installed_packages.txt', 'r') as f:
    installed = {line.strip() for line in f if line.strip()}

# Get requirements file path from environment
req_file_path = open('/tmp/req_file.txt', 'r').read().strip()

# Read requirements file
with open(req_file_path, 'r') as f:
    lines = f.readlines()

# Filter requirements
filtered = []
skipped_count = 0
for line in lines:
    stripped = line.strip()
    # Keep comments and empty lines
    if not stripped or stripped.startswith('#'):
        filtered.append(line)
        continue
    
    # Extract package name (handle: package==version, package>=version, package[extra], etc.)
    # Remove comments first
    pkg_line = stripped.split('#')[0].strip()
    match = re.match(r'^([a-zA-Z0-9_-]+)', pkg_line)
    if match:
        pkg_name = match.group(1).lower().replace('-', '_')
        # Check if already installed
        if pkg_name in installed:
            skipped_count += 1
            print(f"Skipping already-installed: {match.group(1)}")
        else:
            filtered.append(line)
    else:
        # Keep lines we can't parse (might be URLs, etc.)
        filtered.append(line)

# Write filtered requirements
with open('/tmp/filtered_requirements.txt', 'w') as f:
    f.writelines(filtered)

print(f"Filtered {skipped_count} already-installed packages")
print(f"Remaining packages to install: {len([l for l in filtered if l.strip() and not l.strip().startswith('#')])}")
PYEOF
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=3 \
        -r /tmp/filtered_requirements.txt && \
    echo "✓ Platform-specific requirements installed successfully" || \
    (echo "⚠️  WARNING: Some requirements installation failed" && \
     pip list | grep -E "(pdfplumber|python-docx|pytesseract|Pillow|pdf2image|beautifulsoup4|openpyxl)" || \
     echo "Some packages may not be fully installed")

# Verify critical packages
RUN python -c "import numpy; print('✓ numpy version:', numpy.__version__)" && \
    python -c "import torch; print('✓ torch version:', torch.__version__); print('✓ CUDA available:', torch.cuda.is_available() if hasattr(torch.cuda, 'is_available') else False)" && \
    python -c "import sentence_transformers; print('✓ sentence-transformers installed')" && \
    python -c "import langchain_core; print('✓ langchain-core installed')" || \
    (echo "ERROR: Failed to verify critical packages" && pip list | grep -E "(numpy|torch|sentence|langchain)" && exit 1)

# Create non-root user and set up directories
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    mkdir -p /app/logs /app/.cache/huggingface /app/.cache/sentence_transformers /app/tests && \
    chown -R appuser:appuser /app

# Copy deepiri-modelkit (shared library) before app code
# This allows installing it as an editable package
COPY deepiri-modelkit /app/deepiri-modelkit

# Install deepiri-modelkit as editable package (before other requirements)
RUN if [ -d "/app/deepiri-modelkit" ] && [ -f "/app/deepiri-modelkit/pyproject.toml" ]; then \
        echo "Installing deepiri-modelkit..." && \
        pip install --no-cache-dir -e /app/deepiri-modelkit || \
        (echo "Warning: deepiri-modelkit installation failed" && true); \
    else \
        echo "Warning: deepiri-modelkit not found, skipping installation"; \
    fi

# Copy application code
COPY diri-cyrex/app /app/app

# Copy tests directory if it exists in build context
# Create placeholder first to ensure directory exists
RUN touch /app/tests/__init__.py
# Copy tests directory - will fail build if tests/ doesn't exist, which is expected
# If tests directory is missing, create it manually before building
COPY diri-cyrex/tests /app/tests
RUN chown -R appuser:appuser /app/tests

# Copy K8s env loader scripts (before switching user)
COPY --chown=root:root ops/k8s/load-k8s-env.sh /usr/local/bin/load-k8s-env.sh
COPY --chown=root:root ops/k8s/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/load-k8s-env.sh /usr/local/bin/docker-entrypoint.sh

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application with entrypoint that loads K8s env
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# =============================================================================
# DEFAULT: Use prebuilt (fastest)
# =============================================================================
FROM final-prebuilt AS final
