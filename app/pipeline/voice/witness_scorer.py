"""Cyrex witness scorer — wires sentence-transformers into toolbox RAG scoring."""

from __future__ import annotations

from typing import Sequence

from diri_agent_toolbox.agi.witness import rank_witness_quotes, score_witness_match


class SentenceTransformerWitnessScorer:
    """Embedding-backed witness relevance for real-time voice/RAG inference."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from app.integrations.embeddings_wrapper import get_robust_embeddings

            model = get_robust_embeddings(self._model_name)

            class _Adapter:
                def embed(self, texts: Sequence[str]):
                    return model.embed_documents(list(texts))

            self._embedder = _Adapter()
        return self._embedder

    def score(self, question: str, quote: str) -> float:
        return score_witness_match(question, quote, embedder=self._get_embedder())

    def rank(self, question: str, quotes: Sequence[str], *, threshold: float = 0.35):
        return rank_witness_quotes(
            question,
            quotes,
            embedder=self._get_embedder(),
            threshold=threshold,
        )


_default_scorer: SentenceTransformerWitnessScorer | None = None


def get_witness_scorer() -> SentenceTransformerWitnessScorer:
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = SentenceTransformerWitnessScorer()
    return _default_scorer
