"""Fake DuelRunnerPort for track-local tests."""

from __future__ import annotations

from typing import Any

from app.pipeline.contracts.models import DuelState
from app.pipeline.contracts.ports import DuelRunnerPort


def _default_duel_state(document_id: str) -> DuelState:
    return DuelState(
        document_id=document_id,
        agent_a_id="agent-a",
        agent_b_id="agent-b",
    )


class NoOpDuelRunner(DuelRunnerPort):
    """Returns a minimal valid DuelState without running a duel."""

    async def run(
        self,
        parsed_doc: Any,
        document_id: str,
        source_doc_hash: str,
    ) -> DuelState:
        return _default_duel_state(document_id)


class FixedDuelRunner(DuelRunnerPort):
    """Returns a fixed duel state for tests."""

    def __init__(self, state: DuelState) -> None:
        self._state = state

    async def run(
        self,
        parsed_doc: Any,
        document_id: str,
        source_doc_hash: str,
    ) -> DuelState:
        return self._state


# Alias used by contract compliance tests
FakeDuelRunner = NoOpDuelRunner
