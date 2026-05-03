#!/usr/bin/env bash
# Compatibility wrapper for callers that still use snake_case naming.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/check-ollama-models.sh"

echo "⚠️  Deprecated script name: check_ollama_models.sh"
echo "   Using canonical script: check-ollama-models.sh"
echo ""
if [ ! -f "${TARGET_SCRIPT}" ]; then
    echo "Error: expected script not found: ${TARGET_SCRIPT}" >&2
    exit 1
fi

exec bash "${TARGET_SCRIPT}" "$@"
