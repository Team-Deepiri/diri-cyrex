#!/usr/bin/env bash
# Compatibility wrapper: keep snake_case entrypoint working.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/check-ollama-models.sh"

echo "⚠️  Deprecated script name: check_ollama_models.sh"
echo "   Using canonical script: check-ollama-models.sh"
echo ""

exec "${TARGET_SCRIPT}" "$@"
