# =============================================================================
# Cyrex Dockerfile — Poetry-based dependency management
# =============================================================================
# Build context: deepiri-platform monorepo root (see docker-compose.yml)
# =============================================================================
ARG BUILD_TYPE=prebuilt
ARG PYTORCH_VERSION=2.9.1
ARG CUDA_VERSION=12.8
ARG PYTHON_VERSION=3.11
ARG DEVICE_TYPE=auto
ARG BASE_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
ARG POETRY_VERSION=1.8.5
ARG POETRY_EXTRAS=gpu

FROM ${BASE_IMAGE} AS base

ARG PYTORCH_VERSION=2.9.1
ARG POETRY_VERSION=1.8.5

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry-cache \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

WORKDIR /app

COPY diri-cyrex/setup.sh /tmp/setup.sh
RUN chmod +x /tmp/setup.sh && /tmp/setup.sh

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    (python -c "import torch; cuda_ver = torch.version.cuda; exit(0 if cuda_ver and (cuda_ver.startswith('12.8') or float(cuda_ver.split('.')[0] + '.' + cuda_ver.split('.')[1]) >= 12.8) else 1)" 2>/dev/null || \
    (echo "Upgrading PyTorch to CUDA 12.8 support..." && \
     pip uninstall -y torch torchvision torchaudio 2>/dev/null || true && \
     pip install --no-cache-dir --upgrade-strategy=only-if-needed \
         --pre torch torchvision torchaudio \
         --index-url https://download.pytorch.org/whl/nightly/cu128)) || true

RUN python -c "import torch" 2>/dev/null || \
    (echo "Installing PyTorch CPU (version: ${PYTORCH_VERSION})..." && \
     pip install --no-cache-dir --upgrade-strategy=only-if-needed \
         torch==${PYTORCH_VERSION} torchvision torchaudio \
         --index-url https://download.pytorch.org/whl/cpu)

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

COPY diri-cyrex/pyproject.toml diri-cyrex/poetry.lock /app/
COPY deepiri-modelkit /deepiri-modelkit

ARG POETRY_EXTRAS=gpu
RUN ln -sf /deepiri-modelkit ../deepiri-modelkit && \
    bash -ec '\
      extras="${POETRY_EXTRAS}"; \
      args=(install --no-root --no-ansi); \
      IFS=, read -ra xs <<< "$extras"; \
      for x in "${xs[@]}"; do x="${x// /}"; [ -n "$x" ] && args+=(--extras "$x"); done; \
      poetry "${args[@]}"; \
    '

RUN python -c "import numpy; import fastapi; import redis; print('✓ core deps OK')" && \
    python -c "import torch; print('✓ torch', torch.__version__)" && \
    (python -c "import langchain_core; print('✓ langchain-core OK')" 2>/dev/null || \
     echo "⚠ langchain import check skipped")

RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    mkdir -p /app/logs /app/.cache/huggingface /app/.cache/sentence_transformers /app/tests && \
    chown -R appuser:appuser /app

COPY diri-cyrex/app /app/app

RUN touch /app/tests/__init__.py
COPY diri-cyrex/tests /app/tests
RUN chown -R appuser:appuser /app/tests

COPY --chown=root:root ops/k8s/load-k8s-env.sh /usr/local/bin/load-k8s-env.sh
COPY --chown=root:root ops/k8s/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/load-k8s-env.sh /usr/local/bin/docker-entrypoint.sh

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM base AS final-prebuilt
FROM base AS final-from-scratch
FROM base AS final
