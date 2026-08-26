"""Cyrex Artifact Engine — voice package."""

from app.pipeline.contracts.models import ConfessionGap, WitnessSpan
from app.pipeline.voice.synthesizer import (
    VoiceQueryResult,
    VoiceSynthesizer,
    collect_witness_citations,
)

# Back-compat alias used by older Track C call sites / docs.
VoiceQueryResponse = VoiceQueryResult

__all__ = [
    "ConfessionGap",
    "VoiceQueryResponse",
    "VoiceQueryResult",
    "VoiceSynthesizer",
    "WitnessSpan",
    "collect_witness_citations",
]
