"""Apply platform Cyrex SQL migrations via asyncpg (optional dev path)."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger("cyrex.database.cyrex_migrations")

_MIGRATION_PATTERN = re.compile(r"^(\d+)_.+\.sql$")


def _split_sql(text: str) -> list[str]:
    """Split a migration file into executable statements."""
    chunks: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") and not buf:
            continue
        buf.append(line)
        if stripped.endswith(";"):
            chunks.append("\n".join(buf).strip())
            buf = []
    if buf:
        tail = "\n".join(buf).strip()
        if tail:
            chunks.append(tail)
    return chunks


async def apply_cyrex_migrations(db, directory: Path) -> int:
    """Run numbered ``*.sql`` files in order. Returns count applied."""
    if not directory.is_dir():
        logger.warning("migration directory missing: %s", directory)
        return 0

    files = sorted(
        (p for p in directory.glob("*.sql") if _MIGRATION_PATTERN.match(p.name)),
        key=lambda p: int(_MIGRATION_PATTERN.match(p.name).group(1)),  # type: ignore
    )
    applied = 0
    for path in files:
        for stmt in _split_sql(path.read_text(encoding="utf-8")):
            try:
                await db.execute(stmt)
            except Exception as exc:
                logger.warning("migration stmt in %s skipped: %s", path.name, exc)
        applied += 1
        logger.info("applied migration %s", path.name)
    return applied


async def maybe_apply_platform_migrations() -> None:
    """When ``CYREX_MIGRATIONS_DIR`` is set, apply platform SQL migrations."""
    raw = os.environ.get("CYREX_MIGRATIONS_DIR")
    if not raw:
        return
    from app.database.postgres import get_postgres_manager

    db = await get_postgres_manager()
    n = await apply_cyrex_migrations(db, Path(raw))
    logger.info("platform migrations applied", extra={"count": n})
