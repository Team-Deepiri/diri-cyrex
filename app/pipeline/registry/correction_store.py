"""Correction store — Postgres only (postgres-cyrex).

Import ``PostgresCorrectionStore`` from here or from
``app.pipeline.registry.postgres_correction_store``.
"""

from app.pipeline.registry.postgres_correction_store import PostgresCorrectionStore

__all__ = ["PostgresCorrectionStore"]
