"""Unit tests for the Cyrex ToolGate host adapter (isolated from full app boot)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("diri_agent_guardrails")

from diri_agent_guardrails import ReasonCode, Verdict  # noqa: E402

_CYREX_ROOT = Path(__file__).resolve().parents[2]
_TOOL_GATE_PATH = _CYREX_ROOT / "app" / "agents" / "tools" / "tool_gate.py"


def _load_tool_gate_module():
    # Ensure `app` is not required; load the adapter file directly.
    name = "cyrex_tool_gate_under_test"
    spec = importlib.util.spec_from_file_location(name, _TOOL_GATE_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tg = _load_tool_gate_module()


@pytest.fixture
def gate(tmp_path: Path):
    names = [
        "calculate",
        "http_get",
        "http_post",
        "file_read",
        "file_write",
        "db_query",
        "search_documents",
    ]
    return tg.make_cyrex_tool_gate(names), tmp_path


def test_policy_covers_registered_and_bans_shell():
    policy = tg.build_cyrex_tool_policy(["calculate", "file_read", "search_documents"])
    assert policy.get("calculate") is not None
    assert policy.get("search_documents") is not None
    assert policy.get("execute_shell") is not None
    assert policy.get("execute_shell").denied is True


def test_calculate_allowed(gate):
    g, root = gate
    result = tg.authorize_cyrex_tool(
        g, tool_name="calculate", parameters={"expression": "1+1"}, sandbox_root=root
    )
    assert result.verdict == Verdict.ALLOW


def test_unknown_tool_denied(gate):
    g, root = gate
    result = tg.authorize_cyrex_tool(
        g, tool_name="not_registered", parameters={}, sandbox_root=root
    )
    assert result.verdict == Verdict.BLOCK
    assert result.reason_code == ReasonCode.TOOL_UNKNOWN
    assert "TOOL_UNKNOWN" in tg.denial_message(result)


def test_execute_shell_denied_even_if_forced(gate):
    g, root = gate
    result = tg.authorize_cyrex_tool(
        g,
        tool_name="execute_shell",
        parameters={"cmd": "rm -rf /"},
        sandbox_root=root,
        approved=True,
    )
    assert result.reason_code == ReasonCode.TOOL_DENIED


def test_path_escape_on_file_read(gate):
    g, root = gate
    result = tg.authorize_cyrex_tool(
        g,
        tool_name="file_read",
        parameters={"path": "../../etc/passwd"},
        sandbox_root=root,
    )
    assert result.passed is False
    assert result.verdict == Verdict.BLOCK


def test_ssrf_metadata_blocked(gate):
    g, root = gate
    result = tg.authorize_cyrex_tool(
        g,
        tool_name="http_get",
        parameters={"url": "http://169.254.169.254/latest/meta-data/"},
        sandbox_root=root,
    )
    assert result.reason_code == ReasonCode.TOOL_DANGEROUS_ARGUMENT
