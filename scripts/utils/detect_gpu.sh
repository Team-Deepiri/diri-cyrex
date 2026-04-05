#!/bin/bash
# GPU Detection for Cyrex Docker build — delegates to deepiri-gpu-utils.
# Contract: stdout is exactly one line, the Docker BASE_IMAGE (same as before).
#
# Install (monorepo layout: Deepiri/deepiri-platform + Deepiri/deepiri-gpu-utils):
#   pip install -e "../../deepiri-gpu-utils"
# See: docs/operations/GPU_UTILS_HOST.md

set -e

if ! command -v deepiri-gpu >/dev/null 2>&1; then
  echo "detect_gpu.sh: 'deepiri-gpu' not on PATH. Install deepiri-gpu-utils:" >&2
  echo "  pip install -e \"\$(git rev-parse --show-toplevel)/../deepiri-gpu-utils\"" >&2
  echo "  (or from Deepiri/: pip install -e deepiri-gpu-utils)" >&2
  echo "Docs: diri-cyrex/docs/operations/GPU_UTILS_HOST.md" >&2
  exit 127
fi

exec deepiri-gpu build-args --base-image-only "$@"
