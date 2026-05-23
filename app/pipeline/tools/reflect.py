"""Shared reflection/validation tool for Tracks B and C.

Both tracks call this module to validate extraction results.
This prevents Track B and Track C from racing on two different
implementations of the same validation logic.

Track B uses this on post-extraction fields.
Track C uses this on answer-time / Voice-of-the-Document paths.

This is the **skeleton** — real tracks may extend it, but the
core shape (ReflectionResult) is frozen in contracts/models.py.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from app.pipeline.contracts.models import (
    CitedField,
    ReflectionIssue,
    ReflectionResult,
)

# Default confidence floor used when no explicit value is provided.
_DEFAULT_CONFIDENCE_FLOOR = 0.60


class ReflectTool:
    """Validates extracted fields for citation integrity, confidence,
    and basic type/date sanity checks.
    """

    def __init__(self, confidence_floor: float = _DEFAULT_CONFIDENCE_FLOOR):
        self.confidence_floor = confidence_floor

    def reflect_fields(
        self,
        fields: Iterable[CitedField],
        source_text: str,
    ) -> ReflectionResult:
        """Run reflection checks on a list of extracted fields.

        Checks performed:
        1. **Low confidence** — field confidence below floor
        2. **Missing citation** — field has zero citations
        3. **Unverifiable quote** — citation quote not found verbatim in source

        Args:
            fields: Extracted fields to validate.
            source_text: Full raw text of the source document for
                         verbatim quote verification.

        Returns:
            ReflectionResult with issues, low-confidence fields,
            missing-citation fields, and unverifiable citations.
        """
        issues: List[ReflectionIssue] = []
        low_confidence_fields: List[str] = []
        missing_citation_fields: List[str] = []
        unverifiable_citations: List[str] = []

        for field in fields:
            # 1. Check confidence floor
            if field.confidence < self.confidence_floor:
                low_confidence_fields.append(field.field_name)
                issues.append(
                    ReflectionIssue(
                        code="low_confidence",
                        severity="warning",
                        field_name=field.field_name,
                        message=f"Field confidence ({field.confidence:.2f}) is below the floor "
                                f"({self.confidence_floor}).",
                    )
                )

            # 2. Check for missing citations
            if not field.citations:
                missing_citation_fields.append(field.field_name)
                issues.append(
                    ReflectionIssue(
                        code="missing_citation",
                        severity="error",
                        field_name=field.field_name,
                        message="Field has no supporting citation.",
                    )
                )
                # Skip quote checks if there are no citations
                continue

            # 3. Verify each citation's quote appears verbatim in source
            for citation in field.citations:
                if citation.quote and citation.quote not in source_text:
                    unverifiable_citations.append(citation.citation_id)
                    issues.append(
                        ReflectionIssue(
                            code="quote_not_found",
                            severity="error",
                            field_name=field.field_name,
                            citation_id=citation.citation_id,
                            message=(
                                f"Citation quote ({citation.quote!r}) does not appear "
                                "verbatim in source text."
                            ),
                        )
                    )

        return ReflectionResult(
            passed=not any(issue.severity == "error" for issue in issues),
            issues=issues,
            low_confidence_fields=low_confidence_fields,
            missing_citation_fields=missing_citation_fields,
            unverifiable_citations=unverifiable_citations,
            confidence_floor=self.confidence_floor,
        )
