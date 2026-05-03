#!/bin/bash
# Record Cyrex + Helox GPU-related outputs for before/after integration comparison.
# Usage: from deepiri-platform repo root:
#   bash diri-cyrex/scripts/record_gpu_integration_baseline.sh
# Output: /tmp/cyrex-helox-gpu-baseline-YYYYMMDD-HHMMSS/

set -euo pipefail

OUT="${OUT_DIR:-/tmp/cyrex-helox-gpu-baseline-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"

PLATFORM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CYREX="$PLATFORM_ROOT/diri-cyrex"
HELOX="$PLATFORM_ROOT/diri-helox"

{
  echo "=== date ==="
  date -Iseconds 2>/dev/null || date
  echo "=== platform git ==="
  (cd "$PLATFORM_ROOT" && git rev-parse HEAD 2>/dev/null) || echo "n/a"
  echo "=== uname ==="
  uname -a 2>/dev/null || true
  echo "=== nvidia-smi ==="
  nvidia-smi 2>&1 || echo "(nvidia-smi not available)"
  echo "=== detect_gpu.sh (stdout = base image) ==="
  bash "$CYREX/scripts/utils/detect_gpu.sh" 2>&1 || echo "(detect_gpu failed — install deepiri-gpu-utils)"
  echo "=== deepiri-gpu build-args ==="
  deepiri-gpu build-args 2>&1 || echo "(deepiri-gpu not installed)"
  echo "=== deepiri-gpu validate --json (first 8k chars) ==="
  deepiri-gpu validate --json 2>&1 | head -c 8000 || true
} >"$OUT/host_snapshot.txt"

if [[ -d "$HELOX" ]]; then
  (
    cd "$HELOX"
    PYTHONPATH="$HELOX" python3 -c "
import sys
sys.path.insert(0, '.')
from core.device_manager import DeviceManager
dm = DeviceManager()
print('device:', dm.get_device())
print('device_info:', dm.get_device_info())
"
  ) >"$OUT/helox_device_manager.txt" 2>&1 || echo "(Helox DeviceManager failed)" >>"$OUT/helox_device_manager.txt"
fi

echo "Wrote: $OUT"
echo "Compare this folder after integration on the same machine class."
