"""ParseStage — wraps the existing DocumentParserService into a pipeline stage.

This module provides a lightweight ``ParseStage`` class that normalises
the output of ``DocumentParserService.parse_document()`` into a
consistent ``ParseResult`` dataclass consumed by downstream pipeline
stages (anticipate, extract, duel, etc.).

Usage::

    from app.pipeline.stages.parse import ParseStage

    stage = ParseStage()
    result = await stage.parse(b"hello world", "test.txt")
    assert result.raw_text == "hello world"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.pipeline.contracts.ports import DocumentParserPort


class ParseError(Exception):
    """Raised when document parsing fails."""


@dataclass
class ParseResult:
    """Normalised output of the parse stage."""

    raw_text: str
    document_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_count: Optional[int] = None


class ParseStage:
    """Pipeline stage that parses a raw document into structured text."""

    def __init__(self, parser: Optional[DocumentParserPort] = None) -> None:
        """
        Args:
            parser: Optional ``DocumentParserPort``. If ``None``, the real
                ``DocumentParserService`` is imported lazily on first ``parse()``.
        """
        self._parser = parser
        self._parser_imported = parser is not None

    async def parse(self, file_content: bytes, filename: str) -> ParseResult:
        if not file_content:
            raise ParseError("Empty file content — nothing to parse")

        if not self._parser_imported:
            from app.services.document_parser_service import DocumentParserService

            self._parser = DocumentParserService()
            self._parser_imported = True

        assert self._parser is not None
        try:
            parsed = await self._parser.parse_document(
                file_content=file_content,
                filename=filename,
                use_ocr=False,
                extract_tables=False,
            )
        except Exception as exc:
            raise ParseError(f"Parser failed: {exc}") from exc

        raw_text = getattr(parsed, "raw_text", "") or ""
        doc_type = getattr(parsed, "document_type", None)
        metadata = getattr(parsed, "metadata", {}) or {}

        if not raw_text.strip():
            raise ParseError("Parser returned empty text — document may be unreadable")

        return ParseResult(
            raw_text=raw_text,
            document_type=doc_type.value if hasattr(doc_type, "value") else (doc_type or "unknown"),
            metadata=dict(metadata),
            page_count=metadata.get("page_count"),
        )
