"""Cyrex host adapter for ``diri_agent_guardrails.ToolGate``.

Authorize tool calls *after* kwargs are finalized and *before* the
implementation runs (see ``ComprehensiveAPITools.execute``).

Until a human-approval loop exists, registered suite tools pass
``approved=True`` so ``requires_approval`` entries do not soft-deadlock
agents. Forbidden names, unknown tools, sandbox escape, and dangerous
argument patterns still block.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from diri_agent_guardrails import (
    CheckResult,
    ToolCallRequest,
    ToolGate,
    ToolPermission,
    ToolPolicy,
    ToolRisk,
    Verdict,
    default_tool_policy,
)

# Scopes Cyrex agents hold for the registered toolbox suite.
CYREX_AGENT_SCOPES: frozenset[str] = frozenset(
    {
        "files:read",
        "files:write",
        "http",
        "db:read",
        "db:write",
        "calendar:read",
        "calendar:write",
        "crm",
        "data",
    }
)

_FILE_TOOLS = frozenset(
    {
        "file_read",
        "file_write",
        "file_list_dir",
        "file_stat",
        "file_delete",
        "file_copy",
        "file_move",
        "file_read_binary",
    }
)
_WRITE_FILE_TOOLS = frozenset(
    {"file_write", "file_delete", "file_copy", "file_move"}
)
_HTTP_TOOLS = frozenset({"http_get", "http_post", "http_request", "call_external_api"})
_DB_WRITE = frozenset({"db_execute"})
_DB_READ = frozenset({"db_query", "db_get_tables"})


def build_cyrex_tool_policy(registered_names: Iterable[str]) -> ToolPolicy:
    """Deny-by-default policy covering Cyrex's registered suite + ban list."""
    base = default_tool_policy()
    permissions: list[ToolPermission] = []
    seen: set[str] = set()

    for name in base.names:
        perm = base.get(name)
        if perm is None:
            continue
        if perm.denied:
            permissions.append(perm)
            seen.add(name)

    for name in registered_names:
        if name in seen:
            continue
        existing = base.get(name)
        if existing is not None and existing.denied:
            continue
        permissions.append(_permission_for_cyrex_tool(name, existing))
        seen.add(name)

    return ToolPolicy(permissions, default_deny=True)


def _permission_for_cyrex_tool(
    name: str,
    existing: ToolPermission | None,
) -> ToolPermission:
    """Map a registered Cyrex tool to a permission.

    ``allowed_param_keys`` is left open (``None``): Cyrex kwargs (e.g. ``params``
    vs ``query_params``, ``path`` vs ``relative_path``) do not always match the
    toolbox catalog allowlist. Param safety still comes from dangerous-argument
    and path-escape checks.

    ``requires_approval`` is False until Cyrex has a HITL loop; Joe's wire-up
    still blocks shell/eval aliases, unknown tools, and sandbox escapes.
    """
    if existing is not None:
        return ToolPermission(
            name=name,
            risk=existing.risk,
            denied=False,
            requires_sandbox=existing.requires_sandbox or name in _FILE_TOOLS,
            requires_approval=False,
            scopes=existing.scopes or _scopes_for(name),
            allowed_param_keys=None,
        )

    if name in _WRITE_FILE_TOOLS:
        risk, scopes = ToolRisk.HIGH, frozenset({"files:write"})
        requires_sandbox = True
    elif name in _FILE_TOOLS:
        risk, scopes = ToolRisk.MEDIUM, frozenset({"files:read"})
        requires_sandbox = True
    elif name in _HTTP_TOOLS:
        risk, scopes = ToolRisk.HIGH, frozenset({"http"})
        requires_sandbox = False
    elif name in _DB_WRITE:
        risk, scopes = ToolRisk.CRITICAL, frozenset({"db:write"})
        requires_sandbox = False
    elif name in _DB_READ:
        risk, scopes = ToolRisk.HIGH, frozenset({"db:read"})
        requires_sandbox = False
    else:
        risk, scopes = ToolRisk.LOW, frozenset()
        requires_sandbox = False

    return ToolPermission(
        name=name,
        risk=risk,
        requires_sandbox=requires_sandbox,
        requires_approval=False,
        scopes=scopes,
        allowed_param_keys=None,
    )


def _scopes_for(name: str) -> frozenset[str]:
    if name in _WRITE_FILE_TOOLS:
        return frozenset({"files:write"})
    if name in _FILE_TOOLS:
        return frozenset({"files:read"})
    if name in _HTTP_TOOLS:
        return frozenset({"http"})
    if name in _DB_WRITE:
        return frozenset({"db:write"})
    if name in _DB_READ:
        return frozenset({"db:read"})
    return frozenset()


def make_cyrex_tool_gate(registered_names: Iterable[str]) -> ToolGate:
    return ToolGate(build_cyrex_tool_policy(registered_names))


def authorize_cyrex_tool(
    gate: ToolGate,
    *,
    tool_name: str,
    parameters: Mapping[str, Any],
    sandbox_root: str | Path,
    agent_id: Optional[str] = None,
    approved: bool = True,
    scopes: frozenset[str] = CYREX_AGENT_SCOPES,
) -> CheckResult:
    """Authorize a finalized tool call. Host should refuse unless ``Verdict.ALLOW``."""
    perm = gate.policy.get(tool_name)
    needs_sandbox = bool(perm and perm.requires_sandbox) or tool_name in _FILE_TOOLS
    return gate.authorize(
        ToolCallRequest(
            tool_name=tool_name,
            parameters=dict(parameters),
            agent_id=agent_id,
            scopes=scopes,
            sandboxed=needs_sandbox,
            sandbox_root=sandbox_root if needs_sandbox else None,
            # Suite registration is the interim approval signal until HITL exists.
            approved=approved,
        )
    )


def denial_message(result: CheckResult) -> str:
    code = result.reason_code.value if result.reason_code else result.verdict.value
    return f"tool gate denied ({code}): {result.message}"


__all__ = [
    "CYREX_AGENT_SCOPES",
    "Verdict",
    "authorize_cyrex_tool",
    "build_cyrex_tool_policy",
    "denial_message",
    "make_cyrex_tool_gate",
]
