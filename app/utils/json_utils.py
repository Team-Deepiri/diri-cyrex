"""Shared helpers for values returned from PostgreSQL JSON columns."""

from __future__ import annotations

import json
from typing import Any


def _json_value(value: Any, default: Any) -> Any:
    """Return a decoded JSON value while preserving already-decoded values."""
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return value
