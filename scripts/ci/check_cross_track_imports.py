#!/usr/bin/env python3
"""Reject imports between Cyrex AGI implementation tracks.

The tracks may share contracts, standard-library code, and their own
implementation.  Cross-track dependencies must be expressed through the
protocols in ``app.pipeline.contracts`` instead of concrete implementations.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Keep this map aligned with the AGI implementation plan.  Missing directories
# are intentional: the gate protects future Track C/D packages before they are
# created.
TRACK_MODULE_PREFIXES = {
    "A": ("app.pipeline.registry", "app.pipeline.projectors"),
    "B": ("app.pipeline.stages", "app.pipeline.processors"),
    "C": ("app.pipeline.voice", "app.pipeline.corrections", "app.routes.artifacts"),
    "D": (
        "app.pipeline.pressure",
        "app.routes.pressure",
        "app.routes.reckoning",
        "app.mcp",
    ),
}


def module_name(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    return ".".join(relative.parts)


def owner_for(module: str) -> str | None:
    owners = [
        owner
        for owner, prefixes in TRACK_MODULE_PREFIXES.items()
        if any(module == prefix or module.startswith(prefix + ".") for prefix in prefixes)
    ]
    if len(owners) == 1:
        return owners[0]
    return None


def resolve_import(current_module: str, node: ast.ImportFrom) -> str:
    imported = node.module or ""
    if node.level == 0:
        return imported

    package = current_module.split(".")[:-1]
    keep = len(package) - (node.level - 1)
    base = package[: max(keep, 0)]
    return ".".join([*base, imported]).strip(".")


def iter_python_files() -> list[Path]:
    files: list[Path] = []
    for relative_root in ("app/pipeline", "app/routes", "app/mcp"):
        root = ROOT / relative_root
        if root.exists():
            files.extend(root.rglob("*.py"))
    return sorted(files)


def main() -> int:
    violations: list[str] = []

    for path in iter_python_files():
        source_module = module_name(path)
        source_owner = owner_for(source_module)
        if source_owner is None:
            continue

        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            print(f"{path}:{exc.lineno}: syntax error: {exc.msg}", file=sys.stderr)
            return 2

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported_modules = [resolve_import(source_module, node)]
            else:
                continue

            for imported_module in imported_modules:
                target_owner = owner_for(imported_module)
                if target_owner and target_owner != source_owner:
                    violations.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: "
                        f"Track {source_owner} must not import Track {target_owner}: "
                        f"{imported_module}"
                    )

    if violations:
        print("Cross-track import violations detected:", file=sys.stderr)
        print("\n".join(violations), file=sys.stderr)
        print(
            "Use app.pipeline.contracts ports/models instead of importing another "
            "track's implementation.",
            file=sys.stderr,
        )
        return 1

    print("Cross-track import gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
