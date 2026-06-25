"""Contract tests for deepiri-gpu-utils integration."""

from __future__ import annotations

import pytest

from app.utils.device_detection import get_device, get_torch_device
from app.utils.gpu_utils_bridge import docker_base_image, is_gpu_available


def test_device_detection_returns_known_device() -> None:
    assert get_device() in {"cpu", "cuda", "mps"}
    assert get_torch_device().type in {"cpu", "cuda", "mps"}


def test_docker_base_image_is_non_empty_string() -> None:
    image = docker_base_image()
    assert isinstance(image, str)
    assert len(image) > 0
    assert ":" in image


def test_is_gpu_available_matches_device_string() -> None:
    assert is_gpu_available() == (get_device() != "cpu")


@pytest.mark.parametrize("force", ["cpu", "auto"])
def test_detect_torch_device_force_policies(force: str) -> None:
    from app.utils.gpu_utils_bridge import detect_torch_device

    device = detect_torch_device(force=force)
    assert device.type in {"cpu", "cuda", "mps"}
