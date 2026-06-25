"""Host and runtime helpers backed by **deepiri-gpu-utils**."""

from __future__ import annotations

from typing import Optional

import torch

from app.utils.device_detection import get_device, get_torch_device


def detect_torch_device(force: Optional[str] = None) -> torch.device:
    """Resolve ``torch.device``; optional ``force`` maps to gpu-utils policy."""
    if force is None:
        return get_torch_device()
    try:
        from deepiri_gpu_utils.torch_device import resolve_torch_device

        policy = force.lower()  # type: ignore[arg-type]
        return torch.device(resolve_torch_device(policy).device)
    except Exception:
        key = force.lower()
        if key == "cpu":
            return torch.device("cpu")
        if key in ("cuda", "rocm") and torch.cuda.is_available():
            return torch.device("cuda")
        if key == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return get_torch_device()


def is_gpu_available() -> bool:
    """True when resolved device is not CPU."""
    return get_device() != "cpu"


def docker_base_image() -> str:
    """Docker BASE_IMAGE line for Cyrex hybrid builds (same contract as detect_gpu.sh)."""
    from deepiri_gpu_utils.build_args import build_args_from_detection

    return build_args_from_detection().base_image


__all__ = [
    "detect_torch_device",
    "docker_base_image",
    "get_device",
    "get_torch_device",
    "is_gpu_available",
]
