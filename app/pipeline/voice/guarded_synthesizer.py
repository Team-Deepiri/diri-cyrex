"""Guarded voice synthesis — guardrails + synthesizer + dataset export hook."""

from __future__ import annotations

from typing import Any, List, Optional

from diri_agent_guardrails.agi.voice import build_voice_guardrail_engine
from diri_agent_guardrails.core.interfaces import GuardrailEngine

from app.pipeline.contracts.models import PersonaScope
from app.pipeline.contracts.ports import ArtifactStorePort
from app.pipeline.voice.synthesizer import (
    ConfessionGap,
    VoiceQueryResult,
    VoiceSynthesizer,
    WitnessSpan,
)


class GuardedVoiceSynthesizer:
    """Voice synthesizer with diri-agent-guardrails enforcement."""

    def __init__(
        self,
        store: ArtifactStorePort,
        engine: Optional[GuardrailEngine] = None,
    ) -> None:
        self._inner = VoiceSynthesizer(store)
        self._engine = engine or build_voice_guardrail_engine()

    def _get_scorer(self):
        from app.pipeline.voice.witness_scorer import get_witness_scorer

        return get_witness_scorer()

    async def query(
        self,
        document_id: str,
        question: str,
        persona_scope: Optional[PersonaScope] = None,
    ) -> VoiceQueryResult:
        scope = persona_scope or PersonaScope()
        pre = self._engine.check(
            question,
            document_id=document_id,
            corpus_filter=scope.corpus_filter,
            witness_set_only=scope.witness_set_only,
        )
        if not pre.passed:
            return VoiceQueryResult(
                confessed=True,
                spans=[],
                gaps=[
                    ConfessionGap(
                        claim_attempted=question,
                        reason=pre.reason_code.value if pre.reason_code else "guardrail_block",
                    )
                ],
            )

        result = await VoiceSynthesizer(
            store, scorer=self._get_scorer()
        ).query(document_id, question, scope)

        span_dicts = [s.model_dump() for s in result.spans]
        post = self._engine.check(
            " ".join(s.quote for s in result.spans) if result.spans else "",
            question=question,
            spans=span_dicts,
            confessed=result.confessed,
            hard_citation_gate=scope.hard_citation_gate,
        )
        if not post.passed and not result.confessed:
            return VoiceQueryResult(
                confessed=True,
                spans=[],
                gaps=[
                    ConfessionGap(
                        claim_attempted=question,
                        reason=post.reason_code.value if post.reason_code else "citation_gate",
                    )
                ],
            )

        if result.spans and len(result.spans) > 1:
            ranked = self._get_scorer().rank(
                question, [s.quote for s in result.spans], threshold=0.0
            )
            order = {row["quote"]: i for i, row in enumerate(ranked)}
            result.spans.sort(key=lambda s: order.get(s.quote, 999))

        return result

    async def to_training_records(
        self, result: VoiceQueryResult, *, document_id: str
    ) -> List[dict[str, Any]]:
        """Export witness spans as structured training rows."""
        rows: List[dict[str, Any]] = []
        for span in result.spans:
            rows.append(
                {
                    "instruction": f"Witness for document {document_id}",
                    "input": "",
                    "output": span.quote,
                    "text": span.quote,
                    "category": "voice_witness",
                    "quality_score": 0.95,
                    "producer": "cyrex.voice",
                    "metadata": {
                        "document_id": document_id,
                        "citation_id": span.citation_id,
                        "char_start": span.char_start,
                        "char_end": span.char_end,
                    },
                }
            )
        return rows
