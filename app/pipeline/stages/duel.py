"""Adversarial duel stage — two independent extractors compared for disagreement."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.pipeline.contracts.models import (
    CitedField,
    DuelResolutionStatus,
    DuelState,
    FieldDiscrepancy,
    Provenance,
    SynthesisResult,
)
from app.pipeline.contracts.ports import DuelRunnerPort, ExtractPort

logger = logging.getLogger(__name__)

DEFAULT_AGENT_A_ID = "agent_a"
DEFAULT_AGENT_B_ID = "agent_b"


def _empty_synthesis_result(document_id: str, source_doc_hash: str) -> SynthesisResult:
    return SynthesisResult(
        document_id=document_id,
        source_doc_hash=source_doc_hash,
        final_fields=[],
        confidence=0.0,
        provenance=Provenance(source_doc_hash=source_doc_hash, document_id=document_id),
    )


async def _run_agent_safely(
    agent: ExtractPort,
    agent_label: str,
    parsed_doc: Any,
    document_id: str,
    source_doc_hash: str,
) -> SynthesisResult:
    """Run a single agent, degrading to an empty result on failure.

    Mirrors extract.py's per-pass error handling: one agent failing should
    not crash the whole duel, it should just contribute no fields.
    """
    try:
        return await agent.run(parsed_doc, document_id, source_doc_hash)
    except Exception as exc:
        logger.error("Duel %s failed for document %s: %s", agent_label, document_id, exc)
        return _empty_synthesis_result(document_id, source_doc_hash)


def _values_equal(a: Any, b: Any) -> bool:
    if a == b:
        return True
    return str(a).strip().lower() == str(b).strip().lower()


def _confidence_delta(
    field_a: Optional[CitedField],
    field_b: Optional[CitedField],
) -> Optional[float]:
    if field_a is None or field_b is None:
        return None
    return abs(field_a.confidence - field_b.confidence)


def _disagreement_reason(
    field_name: str,
    field_a: Optional[CitedField],
    field_b: Optional[CitedField],
) -> str:
    if field_a is None:
        return f"Agent A did not extract '{field_name}'; Agent B did."
    if field_b is None:
        return f"Agent B did not extract '{field_name}'; Agent A did."
    return f"Agent A and Agent B disagree on '{field_name}'."


def _compare_agent_fields(
    agent_a_fields: List[CitedField],
    agent_b_fields: List[CitedField],
) -> List[FieldDiscrepancy]:
    fields_a: Dict[str, CitedField] = {f.field_name: f for f in agent_a_fields}
    fields_b: Dict[str, CitedField] = {f.field_name: f for f in agent_b_fields}
    field_names = sorted(set(fields_a) | set(fields_b))

    disagreements: List[FieldDiscrepancy] = []
    for field_name in field_names:
        field_a = fields_a.get(field_name)
        field_b = fields_b.get(field_name)

        if field_a is not None and field_b is not None:
            if _values_equal(field_a.value, field_b.value):
                continue
        # Either values differ, or one agent is missing the field entirely.

        disagreements.append(
            FieldDiscrepancy(
                field_name=field_name,
                agent_a_value=field_a.value if field_a else None,
                agent_b_value=field_b.value if field_b else None,
                agent_a_confidence=field_a.confidence if field_a else None,
                agent_b_confidence=field_b.confidence if field_b else None,
                confidence_delta=_confidence_delta(field_a, field_b),
                reason=_disagreement_reason(field_name, field_a, field_b),
            )
        )

    return disagreements


def to_arena_rows(state: DuelState) -> List[Dict[str, Any]]:
    """Map a DuelState to DuelArenaResponse.fields[]-shaped rows for VIZ-03/04."""
    fields_a: Dict[str, CitedField] = {f.field_name: f for f in state.agent_a_fields}
    fields_b: Dict[str, CitedField] = {f.field_name: f for f in state.agent_b_fields}
    disagreement_names = {d.field_name for d in state.disagreements}
    field_names = sorted(set(fields_a) | set(fields_b))

    rows: List[Dict[str, Any]] = []
    for field_name in field_names:
        field_a = fields_a.get(field_name)
        field_b = fields_b.get(field_name)
        rows.append(
            {
                "field_name": field_name,
                "agent_a_value": field_a.value if field_a else None,
                "agent_b_value": field_b.value if field_b else None,
                "agent_a_confidence": field_a.confidence if field_a else None,
                "agent_b_confidence": field_b.confidence if field_b else None,
                "is_disagreement": field_name in disagreement_names,
            }
        )
    return rows


class DuelStage(DuelRunnerPort):
    """Runs two independent ExtractPort agents and returns their DuelState.

    v1 runs agents sequentially (simplicity over throughput; parallelizing
    with ``asyncio.gather`` is a documented v1.1 micro-opt) and compares
    only ``final_fields`` — real per-pass provenance from each agent's
    SynthesisResult is discarded here; downstream reckoning/pressure
    consumes ``DuelState`` only. Each agent call is isolated: if one agent
    raises, it degrades to an empty result rather than failing the duel.
    """

    def __init__(
        self,
        agent_a: ExtractPort,
        agent_b: ExtractPort,
        agent_a_id: str = DEFAULT_AGENT_A_ID,
        agent_b_id: str = DEFAULT_AGENT_B_ID,
    ) -> None:
        self._agent_a = agent_a
        self._agent_b = agent_b
        self._agent_a_id = agent_a_id
        self._agent_b_id = agent_b_id

    async def run(
        self,
        parsed_doc: Any,
        document_id: str,
        source_doc_hash: str,
    ) -> DuelState:
        result_a = await _run_agent_safely(
            self._agent_a, self._agent_a_id, parsed_doc, document_id, source_doc_hash
        )
        result_b = await _run_agent_safely(
            self._agent_b, self._agent_b_id, parsed_doc, document_id, source_doc_hash
        )

        disagreements = _compare_agent_fields(
            result_a.final_fields, result_b.final_fields
        )

        return DuelState(
            document_id=document_id,
            agent_a_id=self._agent_a_id,
            agent_b_id=self._agent_b_id,
            agent_a_fields=result_a.final_fields,
            agent_b_fields=result_b.final_fields,
            disagreements=disagreements,
            resolution_status=DuelResolutionStatus.UNRESOLVED,
        )
