"""Fake PipelineRunnerPort for track-local API route tests."""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.pipeline.contracts.models import ArtifactBundle
from app.pipeline.contracts.ports import PipelineRunnerPort


class FakePipelineRunner(PipelineRunnerPort):
    """Returns a golden ArtifactBundle without running the real pipeline."""

    def __init__(self, golden_bundle: ArtifactBundle) -> None:
        self._golden = golden_bundle
        self.call_count: int = 0

    async def run_document(
        self,
        file_content: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactBundle:
        self.call_count += 1
        return self._golden
