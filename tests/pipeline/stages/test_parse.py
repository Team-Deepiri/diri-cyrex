"""Tests for ParseStage.

These tests verify that ``ParseStage`` correctly wraps
``DocumentParserService`` and normalises output into ``ParseResult``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.pipeline.stages.parse import ParseError, ParseResult, ParseStage


# ---------------------------------------------------------------------------
# Fake parser
# ---------------------------------------------------------------------------


class FakeDocumentParser:
    """A minimal fake that mimics ``DocumentParserService.parse_document``
    without triggering the ``app.services`` eager import chain."""

    async def parse_document(self, file_content: bytes, filename: str, **kwargs) -> Any:
        from types import SimpleNamespace
        return SimpleNamespace(
            raw_text=file_content.decode("utf-8", errors="replace"),
            document_type=SimpleNamespace(value="txt"),
            metadata={"filename": filename},
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stage() -> ParseStage:
    """A ParseStage with a fake parser (avoids openai import chain)."""
    return ParseStage(parser=FakeDocumentParser())


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_parse_txt(stage: ParseStage):
    """A plain text file returns raw_text matching the input."""
    content = b"Hello, this is a test document."
    result = await stage.parse(content, "test.txt")
    assert isinstance(result, ParseResult)
    assert result.raw_text == "Hello, this is a test document."
    assert result.document_type == "txt"


@pytest.mark.asyncio()
async def test_parse_markdown(stage: ParseStage):
    """Markdown bytes parse; FakeDocumentParser returns type from extension-agnostic txt."""
    content = b"# Heading\n\nSome *markdown* content."
    result = await stage.parse(content, "readme.md")
    assert isinstance(result, ParseResult)
    assert result.raw_text
    # FakeDocumentParser always yields document_type=txt (not a full mime sniffer).
    assert result.document_type == "txt"


@pytest.mark.asyncio()
async def test_parse_metadata_populated(stage: ParseStage):
    """Metadata dict includes filename and other parser outputs."""
    result = await stage.parse(b"metadata test", "data.txt")
    assert isinstance(result.metadata, dict)
    assert "filename" in result.metadata


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_parse_empty_bytes_raises(stage: ParseStage):
    """Empty file content raises ParseError."""
    with pytest.raises(ParseError, match="Empty file content"):
        await stage.parse(b"", "empty.txt")


@pytest.mark.asyncio()
async def test_parse_unknown_extension(stage: ParseStage):
    """Files with unknown extensions are still parsed as text."""
    content = b"some content"
    result = await stage.parse(content, "file.xyz")
    assert isinstance(result, ParseResult)
    assert result.raw_text


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_parse_none_content(stage: ParseStage):
    """None-like content raises ParseError (via empty check)."""
    with pytest.raises(ParseError, match="Empty file content"):
        await stage.parse(b"", "empty.txt")
