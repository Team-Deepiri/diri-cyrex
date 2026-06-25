#!/bin/bash
# GPU Detection for Cyrex Docker build — delegates to deepiri-gpu-utils.
# Contract: stdout is exactly one line, the Docker BASE_IMAGE (same as before).
#
# Prefer: poetry install in diri-cyrex (declares deepiri-gpu-utils by tag).
# See: docs/operations/GPU_UTILS_HOST.md

set -e

_run_deepiri_gpu() {
  if command -v deepiri-gpu >/dev/null 2>&1; then
    deepiri-gpu build-args --base-image-only "$@"
    return
  fi
  local root
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  if [ -f "$root/pyproject.toml" ] && command -v poetry >/dev/null 2>&1; then
    (cd "$root" && poetry run deepiri-gpu build-args --base-image-only "$@")
    return
  fi
  echo "detect_gpu.sh: 'deepiri-gpu' not on PATH. Install via Poetry in diri-cyrex:" >&2
  echo "  cd diri-cyrex && poetry install --with dev" >&2
  echo "Docs: diri-cyrex/docs/operations/GPU_UTILS_HOST.md" >&2
  exit 127
}

_run_deepiri_gpu "$@"
