#!/bin/bash
# GPU Detection for Cyrex Docker build — delegates to deepiri-gpu-utils.
# Contract: stdout is exactly one line, the Docker BASE_IMAGE.
#
# Resolution order:
#   1. deepiri-gpu on PATH
#   2. poetry run deepiri-gpu (full Cyrex env)
#   3. pip install deepiri-gpu-utils tag v0.1.1 into a temp/venv (standalone)
#
# See: requirements/README.md and docs/operations/GPU_UTILS_HOST.md

set -e

CYREX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_UTILS_PIP_SPEC='deepiri-gpu-utils[torch] @ git+https://github.com/Team-Deepiri/deepiri-gpu-utils.git@v0.1.1'

_run_cli() {
  deepiri-gpu build-args --base-image-only "$@"
}

if command -v deepiri-gpu >/dev/null 2>&1; then
  exec _run_cli "$@"
fi

if [ -f "$CYREX_ROOT/pyproject.toml" ] && command -v poetry >/dev/null 2>&1; then
  exec bash -c "cd \"$CYREX_ROOT\" && poetry run deepiri-gpu build-args --base-image-only \"\$@\"" bash "$@"
fi

if command -v pip >/dev/null 2>&1; then
  echo "detect_gpu.sh: installing standalone deepiri-gpu-utils for this invocation..." >&2
  pip install -q "$GPU_UTILS_PIP_SPEC"
  exec _run_cli "$@"
fi

echo "detect_gpu.sh: need deepiri-gpu on PATH, Poetry in diri-cyrex, or pip." >&2
echo "  cd diri-cyrex && poetry install --extras cpu   # or gpu, mps, rocm" >&2
echo "  ./setup.sh && INSTALL_PYTHON_DEPS=1 ./setup.sh" >&2
exit 127
