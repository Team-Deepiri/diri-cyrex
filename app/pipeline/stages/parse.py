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

class ParseError(Exception):
    """Raised when document parsing fails."""


@dataclass
class ParseResult:
    """Normalised output of the parse stage.

    Attributes:
        raw_text: Full extracted text content of the document.
        document_type: String identifying the document type (e.g. \"txt\", \"pdf\").
        metadata: Arbitrary key-value metadata from the parser.
        page_count: Number of pages, if the parser provides it.
    """
    raw_text: str
    document_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_count: Optional[int] = None


class ParseStage:
    """Pipeline stage that parses a raw document into structured text.

    Wraps ``DocumentParserService.parse_document()`` and normalises
    its ``ParsedDocument`` output into a ``ParseResult`` dataclass that
    downstream stages can consume without depending on the parser's
    internal shape.
    """

    def __init__(self, parser: Any = None) -> None:
        """
        Args:
            parser: Optional parser instance. If ``None``, the real
                ``DocumentParserService`` is imported lazily on the
                first call to ``parse()`` to avoid triggering the
                ``app.services`` eager import chain (which requires
                ``openai`` and other heavy dependencies).
        """
        self._parser = parser
        self._parser_imported = parser is not None

    async def parse(self, file_content: bytes, filename: str) -> ParseResult:
        """Parse a document and return a normalised result.

        Args:
            file_content: Raw bytes of the document.
            filename: Original filename (used for type detection).

        Returns:
            A ``ParseResult`` with extracted text and metadata.

        Raises:
            ParseError: If the document is empty, unparseable, or the
                underlying parser raises an exception.
        """
        if not file_content:
            raise ParseError("Empty file content — nothing to parse")

        # Lazily import the real parser on first use to avoid
        # triggering ``app.services.__init__`` eager import chain.
        if not self._parser_imported:
            from app.services.document_parser_service import DocumentParserService
            self._parser = DocumentParserService()
            self._parser_imported = True

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
            document_type=doc_type.value if doc_type else "unknown",
            metadata=dict(metadata),
            page_count=metadata.get("page_count"),
        )
