# GPU Detection for Cyrex Docker build — delegates to deepiri-gpu-utils.
# Contract: stdout is exactly one line, the Docker BASE_IMAGE.
#
# Install: pip install -e <path-to-deepiri-gpu-utils>
# See: docs/operations/GPU_UTILS_HOST.md

$deepiriGpu = Get-Command deepiri-gpu -ErrorAction SilentlyContinue
if (-not $deepiriGpu) {
    Write-Host "detect_gpu.ps1: 'deepiri-gpu' not on PATH. Install deepiri-gpu-utils (pip install -e ...)." -ForegroundColor Red
    Write-Host "Docs: diri-cyrex/docs/operations/GPU_UTILS_HOST.md" -ForegroundColor Yellow
    exit 127
}

& deepiri-gpu build-args --base-image-only @args
exit $LASTEXITCODE
