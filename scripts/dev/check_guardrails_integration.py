#!/usr/bin/env python3
"""
Smoke-test diri-agent-guardrails the same way diri-cyrex uses it.

Run from the diri-cyrex project root (recommended):

    python scripts/dev/check_guardrails_integration.py

Optional: verify Cyrex can import modules that depend on guardrails:

    python scripts/dev/check_guardrails_integration.py --with-cyrex

Exit 0 if all checks pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
from pathlib import Path

# diri-cyrex root: scripts/dev -> scripts -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def check_package_imports() -> bool:
    """Same symbols Cyrex imports from diri_agent_guardrails."""
    print("Checking diri_agent_guardrails imports (Cyrex surface)...")
    try:
        import diri_agent_guardrails as pkg

        from diri_agent_guardrails import (
            GuardrailAction,
            SafetyGuardrails,
            get_advanced_guardrails,
            get_enhanced_guardrails,
            get_guardrails,
            reset_enhanced_guardrails,
        )

        ver = getattr(pkg, "__version__", "unknown")
        print(f"  OK — package version: {ver}")
        del SafetyGuardrails, GuardrailAction  # used only to verify import
        del get_guardrails, get_advanced_guardrails, get_enhanced_guardrails, reset_enhanced_guardrails
        return True
    except ImportError as e:
        print(f"  FAIL — {e}")
        print("  Fix: pip install -r requirements.txt")
        print("  Or: pip install -e /path/to/diri-agent-guardrails")
        return False


def check_safety_guardrails() -> bool:
    """Orchestrator / execution_engine path: get_guardrails() + check_prompt."""
    print("Checking SafetyGuardrails (sync)...")
    try:
        from diri_agent_guardrails import get_guardrails

        g = get_guardrails()
        result = g.check_prompt("hello world — benign test")
        print(f"  OK — level={result.level.value}, score={result.score:.2f}")
        return True
    except Exception as e:
        print(f"  FAIL — {e}")
        return False


async def check_advanced_guardrails() -> bool:
    """Agent playground path: get_advanced_guardrails + check_input."""
    print("Checking AdvancedGuardrails (async)...")
    try:
        from diri_agent_guardrails import get_advanced_guardrails

        adv = await get_advanced_guardrails(force_reload=True)
        res = await adv.check_input("hello world — benign test")
        print(f"  OK — passed={res.passed}, action={res.action.value}")
        return True
    except Exception as e:
        print(f"  FAIL — {e}")
        return False


async def check_enhanced_guardrails() -> bool:
    """Base agent / system_initializer path: get_enhanced_guardrails (no Postgres)."""
    print("Checking EnhancedGuardrails without Postgres (in-memory rules)...")
    try:
        from diri_agent_guardrails import get_enhanced_guardrails, reset_enhanced_guardrails

        reset_enhanced_guardrails()
        g = await get_enhanced_guardrails(postgres=None, schema="cyrex")
        out = await g.check("hello world — benign test")
        reset_enhanced_guardrails()
        safe = out.get("safe")
        print(f"  OK — safe={safe}, action={out.get('action')}")
        return True
    except Exception as e:
        print(f"  FAIL — {e}")
        return False


def check_cyrex_modules() -> bool:
    """Import Cyrex modules that reference diri_agent_guardrails (heavy)."""
    print("Checking Cyrex imports (--with-cyrex)...")
    root = PROJECT_ROOT.resolve()
    if not (root / "app").is_dir():
        print(f"  FAIL — expected app/ under {root}")
        return False

    prev_cwd = os.getcwd()
    prev_path = list(sys.path)
    try:
        os.chdir(root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        importlib.import_module("app.core.execution_engine")
        from app.core.execution_engine import TaskExecutionEngine

        engine = TaskExecutionEngine()
        if engine.guardrails is None:
            print("  FAIL — TaskExecutionEngine.guardrails is None")
            return False
        r = engine.guardrails.check_prompt("smoke test")
        print(f"  OK — app.core.execution_engine + guardrails (level={r.level.value})")

        importlib.import_module("app.routes.agent_playground_api")
        print("  OK — app.routes.agent_playground_api imports")

        return True
    except Exception as e:
        print(f"  FAIL — {e}")
        return False
    finally:
        os.chdir(prev_cwd)
        sys.path[:] = prev_path


async def run_async_checks() -> bool:
    ok = True
    ok = await check_advanced_guardrails() and ok
    ok = await check_enhanced_guardrails() and ok
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-cyrex",
        action="store_true",
        help="Also import Cyrex modules (requires full app dependencies and cwd side effects).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DEEPIRI AGENT GUARDRAILS — CYREX SMOKE TEST")
    print("=" * 60)

    results: list[tuple[str, bool]] = []
    results.append(("Package imports", check_package_imports()))
    results.append(("Safety guardrails", check_safety_guardrails()))
    results.append(("Advanced + enhanced (async)", asyncio.run(run_async_checks())))
    if args.with_cyrex:
        results.append(("Cyrex module imports", check_cyrex_modules()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        mark = "OK" if passed else "FAIL"
        print(f"  [{mark}] {name}")

    if all(p for _, p in results):
        print("\nAll checks passed.")
        return 0
    print("\nOne or more checks failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
