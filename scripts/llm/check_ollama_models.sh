#!/bin/bash
# Compatibility wrapper for callers that use snake_case naming.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/check-ollama-models.sh"

if [ ! -f "${TARGET_SCRIPT}" ]; then
    echo "Error: expected script not found: ${TARGET_SCRIPT}" >&2
    exit 1
fi

exec bash "${TARGET_SCRIPT}" "$@"
