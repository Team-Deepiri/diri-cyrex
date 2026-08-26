"""Artifact Engine bootstrap — mode selection + schema ensure."""

from __future__ import annotations

import os

from app.database.agi_schema import ensure_agi_schema
from app.database.cyrex_migrations import maybe_apply_platform_migrations
from app.logging_config import get_logger

logger = get_logger("cyrex.pipeline.bootstrap")


def pipeline_mode() -> str:
    """``CYREX_PIPELINE_MODE``: postgres (default) | fake | sqlite."""
    return os.environ.get("CYREX_PIPELINE_MODE", "postgres").strip().lower()


async def bootstrap_artifact_engine() -> None:
    """Idempotent startup: AGI DDL + log active pipeline mode."""
    mode = pipeline_mode()
    logger.info("artifact engine bootstrap", extra={"mode": mode})
    if mode in ("postgres", "default", ""):
        await maybe_apply_platform_migrations()
        await ensure_agi_schema()
