"""Cyrex device policy: **deepiri-gpu-utils** first, modelkit fallback."""

from __future__ import annotations

import torch
from deepiri_modelkit.utils.device import get_device as _modelkit_get_device
from deepiri_modelkit.utils.device import get_torch_device as _modelkit_get_torch_device


def get_device() -> str:
    """Return device string (``cuda``, ``mps``, or ``cpu``) for PyTorch workloads."""
    try:
        from deepiri_gpu_utils.torch_device import resolve_torch_device

        return resolve_torch_device("auto").device
    except Exception:
        return _modelkit_get_device()


def get_torch_device() -> torch.device:
    """Return ``torch.device`` using shared Deepiri GPU policy when available."""
    try:
        from deepiri_gpu_utils.torch_device import resolve_torch_device

        return torch.device(resolve_torch_device("auto").device)
    except Exception:
        return _modelkit_get_torch_device()


__all__ = ["get_device", "get_torch_device"]
