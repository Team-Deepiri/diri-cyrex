#!/usr/bin/env bash
set -euo pipefail

if [ -x /usr/local/bin/load-k8s-env.sh ]; then
  # shellcheck source=/dev/null
  source /usr/local/bin/load-k8s-env.sh || true
fi

exec "$@"
