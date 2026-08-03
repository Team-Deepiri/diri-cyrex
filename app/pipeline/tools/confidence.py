"""Confidence scoring for the artifact pipeline.

Separated from the orchestrator so the scoring strategy is a small,
independently testable component that can be swapped or evolved
(e.g. per-stage weighted aggregation) without touching pipeline wiring.
"""

from __future__ import annotations

from typing import Optional

from app.pipeline.contracts.models import ReflectionResult, SynthesisResult
from app.pipeline.stages.parse import ParseResult


class ConfidenceCalculator:
    """Computes an overall artifact confidence from stage results.

    Strategy: start from the synthesis result's confidence, then reduce it
    for reflection issues with severity ``error`` (5% per error, floored at
    0.10). Warnings are surfaced in the payload but do not reduce the score.
    """

    ERROR_PENALTY = 0.05
    FLOOR = 0.10
    DEFAULT_CONFIDENCE = 0.5

    def compute(
        self,
        synthesis_result: Optional[SynthesisResult],
        reflection_result: Optional[ReflectionResult],
        parse_result: ParseResult,
    ) -> float:
        """Return a 0–1 confidence score for the artifact bundle."""
        confidence = self.DEFAULT_CONFIDENCE

        if synthesis_result is not None:
            confidence = float(synthesis_result.confidence)

        if reflection_result is not None and reflection_result.issues:
            error_count = sum(
                1
                for issue in reflection_result.issues
                if getattr(issue, "severity", None) == "error"
            )
            confidence -= error_count * self.ERROR_PENALTY
            confidence = max(confidence, self.FLOOR)

        return round(confidence, 4)
