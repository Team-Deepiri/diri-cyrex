"""Multi-pass extract stage — REGEX → LLM → CROSS_REF synthesis."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

from diri_agent_toolbox.data import text_extract

from app.pipeline.contracts.models import (
    Citation,
    CitationLocator,
    CitedField,
    ExtractionMethod,
    FieldDiscrepancy,
    Provenance,
    ProvenancePass,
    SynthesisResult,
)
from app.pipeline.contracts.ports import ExtractPort
from app.pipeline.tools.reflect import ReflectTool

logger = logging.getLogger(__name__)

REGEX_PASS = 1
LLM_PASS = 2

REGEX_CONFIDENCE = 0.85
LLM_CONFIDENCE = 0.90

DOCUMENT_CLASS_FIELDS: Dict[str, List[str]] = {
    "lease": [
        "base_rent",
        "lease_start",
        "security_deposit",
        "notice_period",
    ],
}


@dataclass(frozen=True)
class _PassField:
    field_name: str
    value: Any
    value_type: str
    quote: Optional[str]
    char_start: Optional[int]
    char_end: Optional[int]
    confidence: float
    pass_number: int


class LlmExtractBackend(Protocol):
    """Injectable LLM extraction backend — real impl gated for integration tests."""

    async def extract_fields(
        self,
        raw_text: str,
        fields: list[str],
        document_class: str,
    ) -> dict[str, Any]: ...


class NoOpLlmExtract:
    """Default LLM backend for CI — returns no fields."""

    async def extract_fields(
        self,
        raw_text: str,
        fields: list[str],
        document_class: str,
    ) -> dict[str, Any]:
        del raw_text, fields, document_class
        return {}


def _coerce_source_text(parsed_doc: Any) -> str:
    if parsed_doc is None:
        return ""
    if isinstance(parsed_doc, str):
        return parsed_doc
    if isinstance(parsed_doc, dict):
        raw = parsed_doc.get("raw_text")
        return raw if isinstance(raw, str) else str(parsed_doc)
    raw_text = getattr(parsed_doc, "raw_text", None)
    if isinstance(raw_text, str):
        return raw_text
    return str(parsed_doc)


def _coerce_document_class(parsed_doc: Any) -> str:
    if parsed_doc is None:
        return "lease"
    if isinstance(parsed_doc, dict):
        metadata = parsed_doc.get("metadata") or {}
        if isinstance(metadata, dict) and metadata.get("document_class"):
            return str(metadata["document_class"])
        doc_type = parsed_doc.get("document_type")
        return str(doc_type) if doc_type else "lease"
    metadata = getattr(parsed_doc, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("document_class"):
        return str(metadata["document_class"])
    document_type = getattr(parsed_doc, "document_type", None)
    return str(document_type) if document_type else "lease"


def _resolve_fields(
    parsed_doc: Any,
    field_templates: Dict[str, List[str]],
) -> List[str]:
    document_class = _coerce_document_class(parsed_doc)
    return list(field_templates.get(document_class, []))


def _find_verbatim_span(raw_text: str, value: str) -> Optional[Tuple[str, int, int]]:
    if not value:
        return None
    match = re.search(re.escape(value), raw_text, re.IGNORECASE)
    if match is None:
        return None
    start, end = match.start(), match.end()
    return raw_text[start:end], start, end


def _infer_value_type(field_name: str, raw_value: str) -> Tuple[Any, str]:
    if field_name in {"base_rent", "security_deposit"}:
        cleaned = raw_value.replace(",", "").replace("$", "").strip()
        try:
            numeric = float(cleaned)
            if numeric.is_integer():
                return int(numeric), "currency"
            return numeric, "currency"
        except ValueError:
            return raw_value, "string"
    if field_name in {"notice_period"}:
        digits = re.search(r"\d+", raw_value)
        if digits:
            return int(digits.group()), "integer"
    if field_name.endswith("_date") or field_name.endswith("_start"):
        return raw_value.strip(), "date"
    return raw_value.strip(), "string"


def _pass_field_from_raw(
    field_name: str,
    raw_value: Any,
    source_text: str,
    pass_number: int,
    confidence: float,
) -> Optional[_PassField]:
    if raw_value is None:
        return None
    raw_str = str(raw_value).strip()
    if not raw_str:
        return None

    value, value_type = _infer_value_type(field_name, raw_str)
    span = _find_verbatim_span(source_text, raw_str)
    if span is None:
        return _PassField(
            field_name=field_name,
            value=value,
            value_type=value_type,
            quote=None,
            char_start=None,
            char_end=None,
            confidence=confidence,
            pass_number=pass_number,
        )

    quote, char_start, char_end = span
    if len(quote) > 500:
        quote = quote[:500]
        char_end = char_start + 500
    return _PassField(
        field_name=field_name,
        value=value,
        value_type=value_type,
        quote=quote,
        char_start=char_start,
        char_end=char_end,
        confidence=confidence,
        pass_number=pass_number,
    )


async def _run_regex_pass(
    source_text: str,
    fields: List[str],
) -> Tuple[Dict[str, _PassField], int]:
    if not source_text or not fields:
        return {}, 0

    started = time.perf_counter()
    tool_result = await text_extract(source_text, fields)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if not tool_result.success or not isinstance(tool_result.result, dict):
        return {}, elapsed_ms

    extracted: Dict[str, _PassField] = {}
    for field_name, raw_value in tool_result.result.items():
        pass_field = _pass_field_from_raw(
            field_name,
            raw_value,
            source_text,
            REGEX_PASS,
            REGEX_CONFIDENCE,
        )
        if pass_field is not None:
            extracted[field_name] = pass_field
    return extracted, elapsed_ms


async def _run_llm_pass(
    llm_backend: LlmExtractBackend,
    source_text: str,
    fields: List[str],
    document_class: str,
) -> Tuple[Dict[str, _PassField], int]:
    if not source_text or not fields:
        return {}, 0

    started = time.perf_counter()
    llm_values = await llm_backend.extract_fields(source_text, fields, document_class)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    extracted: Dict[str, _PassField] = {}
    for field_name, raw_value in llm_values.items():
        pass_field = _pass_field_from_raw(
            field_name,
            raw_value,
            source_text,
            LLM_PASS,
            LLM_CONFIDENCE,
        )
        if pass_field is not None:
            extracted[field_name] = pass_field
    return extracted, elapsed_ms


def _values_equal(a: Any, b: Any) -> bool:
    if a == b:
        return True
    return str(a).strip().lower() == str(b).strip().lower()


def _synthesize(
    regex_fields: Dict[str, _PassField],
    llm_fields: Dict[str, _PassField],
    field_names: List[str],
    document_id: str,
    source_doc_hash: str,
) -> Tuple[List[CitedField], List[FieldDiscrepancy]]:
    final_fields: List[CitedField] = []
    discrepancies: List[FieldDiscrepancy] = []

    for field_name in field_names:
        regex_field = regex_fields.get(field_name)
        llm_field = llm_fields.get(field_name)

        if regex_field is None and llm_field is None:
            continue

        if regex_field is not None and llm_field is not None:
            if not _values_equal(regex_field.value, llm_field.value):
                winner = (
                    llm_field
                    if llm_field.confidence >= regex_field.confidence
                    else regex_field
                )
                discrepancies.append(
                    FieldDiscrepancy(
                        field_name=field_name,
                        pass_a_value=regex_field.value,
                        pass_b_value=llm_field.value,
                        confidence_delta=abs(
                            llm_field.confidence - regex_field.confidence
                        ),
                        reason=(
                            "Regex and LLM passes disagree; "
                            f"selected pass {winner.pass_number} value"
                        ),
                    )
                )
                chosen = winner
            else:
                chosen = llm_field if llm_field.quote else regex_field
                if chosen.quote is None and regex_field.quote:
                    chosen = regex_field
        else:
            chosen = llm_field or regex_field

        assert chosen is not None
        citations: List[Citation] = []
        if chosen.quote is not None:
            citations.append(
                Citation(
                    document_id=document_id,
                    source_doc_hash=source_doc_hash,
                    locator=CitationLocator(
                        locator_type="char_range",
                        char_start=chosen.char_start,
                        char_end=chosen.char_end,
                    ),
                    quote=chosen.quote,
                    confidence=chosen.confidence,
                    extraction_pass=chosen.pass_number,
                )
            )

        final_fields.append(
            CitedField(
                field_name=field_name,
                value=chosen.value,
                value_type=chosen.value_type,
                citations=citations,
                confidence=chosen.confidence,
            )
        )

    return final_fields, discrepancies


def _build_synthesis_result(
    document_id: str,
    source_doc_hash: str,
    final_fields: List[CitedField],
    discrepancies: List[FieldDiscrepancy],
    regex_fields: Dict[str, _PassField],
    llm_fields: Dict[str, _PassField],
    regex_elapsed_ms: int,
    llm_elapsed_ms: int,
) -> SynthesisResult:
    passes: List[ProvenancePass] = []
    if regex_fields:
        passes.append(
            ProvenancePass(
                pass_number=REGEX_PASS,
                method=ExtractionMethod.REGEX,
                fields_extracted=sorted(regex_fields.keys()),
                extraction_time_ms=regex_elapsed_ms,
            )
        )
    if llm_fields:
        passes.append(
            ProvenancePass(
                pass_number=LLM_PASS,
                method=ExtractionMethod.LLM,
                fields_extracted=sorted(llm_fields.keys()),
                extraction_time_ms=llm_elapsed_ms,
            )
        )

    all_citations = [
        citation for field in final_fields for citation in field.citations
    ]
    confidence = (
        sum(field.confidence for field in final_fields) / len(final_fields)
        if final_fields
        else 0.0
    )

    provenance = Provenance(
        source_doc_hash=source_doc_hash,
        document_id=document_id,
        passes=passes,
    )

    return SynthesisResult(
        document_id=document_id,
        source_doc_hash=source_doc_hash,
        final_fields=final_fields,
        all_citations=all_citations,
        confidence=confidence,
        passes=passes,
        provenance=provenance,
        discrepancies=discrepancies,
    )


class ExtractStage(ExtractPort):
    """Multi-pass extraction stage returning a synthesized field bundle."""

    def __init__(
        self,
        reflect_tool: ReflectTool | None = None,
        llm_backend: LlmExtractBackend | None = None,
        field_templates: Dict[str, List[str]] | None = None,
    ) -> None:
        self._reflect_tool = reflect_tool or ReflectTool()
        self._llm_backend = llm_backend or NoOpLlmExtract()
        self._field_templates = field_templates or DOCUMENT_CLASS_FIELDS

    async def run(
        self,
        parsed_doc: Any,
        document_id: str,
        source_doc_hash: str,
    ) -> SynthesisResult:
        source_text = _coerce_source_text(parsed_doc)
        document_class = _coerce_document_class(parsed_doc)
        field_names = _resolve_fields(parsed_doc, self._field_templates)

        regex_fields, regex_elapsed_ms = await _run_regex_pass(
            source_text, field_names
        )
        llm_fields, llm_elapsed_ms = await _run_llm_pass(
            self._llm_backend,
            source_text,
            field_names,
            document_class,
        )

        final_fields, discrepancies = _synthesize(
            regex_fields,
            llm_fields,
            field_names,
            document_id,
            source_doc_hash,
        )

        result = _build_synthesis_result(
            document_id=document_id,
            source_doc_hash=source_doc_hash,
            final_fields=final_fields,
            discrepancies=discrepancies,
            regex_fields=regex_fields,
            llm_fields=llm_fields,
            regex_elapsed_ms=regex_elapsed_ms,
            llm_elapsed_ms=llm_elapsed_ms,
        )

        reflection = self._reflect_tool.reflect_fields(
            result.final_fields, source_text
        )
        if reflection.issues:
            logger.debug(
                "Extract reflection issues for %s: %s",
                document_id,
                [issue.code for issue in reflection.issues],
            )

        return result
