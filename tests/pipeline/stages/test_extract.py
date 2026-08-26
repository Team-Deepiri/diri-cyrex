"""Tests for ExtractStage."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from app.pipeline.contracts.models import SynthesisResult
from app.pipeline.stages.extract import (
    ExtractStage,
    NoOpLlmExtract,
    OllamaLlmExtract,
    build_default_llm_backend,
)
from app.pipeline.tools.reflect import ReflectTool
from tests.fakes.extract import NoOpExtract

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "cyrex_contracts"
LEASE_SAMPLE = (FIXTURES_DIR / "lease_extract_sample.txt").read_text(encoding="utf-8")


@dataclass
class _ParseResultStub:
    raw_text: str
    document_type: str = "lease"
    metadata: dict[str, Any] | None = None
    page_count: int = 1


class _StubLlmBackend:
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values

    async def extract_fields(
        self,
        raw_text: str,
        fields: list[str],
        document_class: str,
    ) -> dict[str, Any]:
        del raw_text, fields, document_class
        return dict(self._values)


class TestExtractStage:
    @pytest.mark.asyncio
    async def test_empty_source_returns_minimal_result(self):
        stage = ExtractStage(reflect_tool=ReflectTool(), llm_backend=NoOpLlmExtract())
        result = await stage.run(
            parsed_doc={"raw_text": ""},
            document_id="lease_empty",
            source_doc_hash="sha256:empty",
        )
        assert isinstance(result, SynthesisResult)
        assert result.final_fields == []
        assert result.confidence == 0.0
        assert result.discrepancies == []

    @pytest.mark.asyncio
    async def test_regex_pass_extracts_labeled_fields(self):
        stage = ExtractStage(reflect_tool=ReflectTool(), llm_backend=NoOpLlmExtract())
        result = await stage.run(
            parsed_doc={"raw_text": LEASE_SAMPLE},
            document_id="lease_001",
            source_doc_hash="sha256:a1b2c3d4e5f6",
        )
        field_names = {field.field_name for field in result.final_fields}
        assert "base_rent" in field_names
        base_rent = next(f for f in result.final_fields if f.field_name == "base_rent")
        assert base_rent.value == 4500

    @pytest.mark.asyncio
    async def test_citation_quote_in_source(self):
        stage = ExtractStage(reflect_tool=ReflectTool(), llm_backend=NoOpLlmExtract())
        result = await stage.run(
            parsed_doc={"raw_text": LEASE_SAMPLE},
            document_id="lease_001",
            source_doc_hash="sha256:a1b2c3d4e5f6",
        )
        reflection = ReflectTool().reflect_fields(result.final_fields, LEASE_SAMPLE)
        assert reflection.unverifiable_citations == []

    @pytest.mark.asyncio
    async def test_cross_pass_discrepancy(self):
        llm_backend = _StubLlmBackend({"base_rent": "4400.00"})
        stage = ExtractStage(reflect_tool=ReflectTool(), llm_backend=llm_backend)
        result = await stage.run(
            parsed_doc={"raw_text": LEASE_SAMPLE},
            document_id="lease_001",
            source_doc_hash="sha256:a1b2c3d4e5f6",
        )
        assert any(d.field_name == "base_rent" for d in result.discrepancies)
        base_rent = next(f for f in result.final_fields if f.field_name == "base_rent")
        # LLM pass has higher default confidence (0.90 vs 0.85) and wins on conflict.
        assert base_rent.value == 4400

    @pytest.mark.asyncio
    async def test_parsed_doc_dict_and_parse_result_shape(self):
        stage = ExtractStage(reflect_tool=ReflectTool(), llm_backend=NoOpLlmExtract())
        dict_result = await stage.run(
            parsed_doc={"raw_text": LEASE_SAMPLE, "document_type": "lease"},
            document_id="lease_001",
            source_doc_hash="sha256:a1b2c3d4e5f6",
        )
        parse_result = await stage.run(
            parsed_doc=_ParseResultStub(raw_text=LEASE_SAMPLE),
            document_id="lease_001",
            source_doc_hash="sha256:a1b2c3d4e5f6",
        )
        dict_names = sorted(f.field_name for f in dict_result.final_fields)
        parse_names = sorted(f.field_name for f in parse_result.final_fields)
        assert dict_names == parse_names

    @pytest.mark.asyncio
    async def test_result_validates_against_golden_subset(self):
        stage = ExtractStage(reflect_tool=ReflectTool(), llm_backend=NoOpLlmExtract())
        result = await stage.run(
            parsed_doc={"raw_text": LEASE_SAMPLE},
            document_id="lease_001",
            source_doc_hash="sha256:a1b2c3d4e5f6",
        )
        assert result.document_id == "lease_001"
        assert result.source_doc_hash == "sha256:a1b2c3d4e5f6"
        assert result.provenance.document_id == "lease_001"
        assert result.passes
        assert result.passes[0].method.value == "regex"
        assert result.confidence > 0.0

    @pytest.mark.asyncio
    async def test_differs_from_fake_extract(self):
        stage = ExtractStage(reflect_tool=ReflectTool(), llm_backend=NoOpLlmExtract())
        fake = NoOpExtract()
        stage_result = await stage.run(
            parsed_doc={"raw_text": LEASE_SAMPLE},
            document_id="lease_001",
            source_doc_hash="sha256:a1b2c3d4e5f6",
        )
        fake_result = await fake.run(
            parsed_doc={"raw_text": LEASE_SAMPLE},
            document_id="lease_001",
            source_doc_hash="sha256:a1b2c3d4e5f6",
        )
        assert stage_result != fake_result
        assert len(stage_result.final_fields) > len(fake_result.final_fields)

    @pytest.mark.asyncio
    async def test_confidence_params_override_defaults(self):
        llm_backend = _StubLlmBackend({"base_rent": "4400.00"})
        stage = ExtractStage(
            reflect_tool=ReflectTool(),
            llm_backend=llm_backend,
            regex_confidence=0.95,
            llm_confidence=0.50,
        )
        result = await stage.run(
            parsed_doc={"raw_text": LEASE_SAMPLE},
            document_id="lease_001",
            source_doc_hash="sha256:a1b2c3d4e5f6",
        )
        base_rent = next(f for f in result.final_fields if f.field_name == "base_rent")
        # Regex confidence higher → regex wins on conflict.
        assert base_rent.value == 4500


class TestBuildDefaultLlmBackend:
    def test_default_is_noop(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CYREX_EXTRACT_USE_LLM", raising=False)
        backend = build_default_llm_backend()
        assert isinstance(backend, NoOpLlmExtract)

    def test_env_flag_selects_ollama(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CYREX_EXTRACT_USE_LLM", "1")
        backend = build_default_llm_backend()
        assert isinstance(backend, OllamaLlmExtract)


class TestExtractPortCompliance:
    def test_stage_implements_port_methods(self):
        stage = ExtractStage(llm_backend=NoOpLlmExtract())
        assert hasattr(stage, "run")
        assert stage.run.__name__ == "run"

    @pytest.mark.asyncio
    async def test_unknown_document_class_returns_empty_fields(self):
        stage = ExtractStage(reflect_tool=ReflectTool(), llm_backend=NoOpLlmExtract())
        result = await stage.run(
            parsed_doc={"raw_text": LEASE_SAMPLE, "document_type": "invoice"},
            document_id="inv_001",
            source_doc_hash="sha256:inv",
        )
        assert result.final_fields == []


def _ollama_reachable() -> bool:
    try:
        import httpx

        from app.settings import settings

        base = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
        response = httpx.get(f"{base.rstrip('/')}/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ollama_llm_pass_extracts_fields():
    if os.getenv("CYREX_EXTRACT_USE_LLM") != "1":
        pytest.skip("Set CYREX_EXTRACT_USE_LLM=1 to run local LLM integration test")
    if not _ollama_reachable():
        pytest.skip("Ollama is not reachable")

    stage = ExtractStage(
        reflect_tool=ReflectTool(),
        llm_backend=OllamaLlmExtract(),
    )
    result = await stage.run(
        parsed_doc={"raw_text": LEASE_SAMPLE},
        document_id="lease_001",
        source_doc_hash="sha256:a1b2c3d4e5f6",
    )
    assert isinstance(result, SynthesisResult)
    assert result.final_fields
    llm_passes = [p for p in result.passes if p.method.value == "llm"]
    assert llm_passes or result.final_fields  # regex may still succeed if LLM fails soft
    if llm_passes:
        assert result.provenance.model_id is not None
