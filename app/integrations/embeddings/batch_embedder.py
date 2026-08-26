"""Batch embedding service — gpu-utils policy + toolbox batch chunking."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, List, Optional, TypeVar

from app.logging_config import get_logger

logger = get_logger("cyrex.integrations.batch_embedder")

T = TypeVar("T")
R = TypeVar("R")


class BatchEmbedder:
    """Embed texts in GPU-aware batches."""

    def __init__(
        self,
        embed_fn: Callable[[List[str]], Coroutine[Any, Any, List[Any]]],
        *,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        self._embed_fn = embed_fn
        self._batch_size = batch_size
        self._device = device

    def _resolve_policy(self) -> tuple[int, str]:
        try:
            from deepiri_gpu_utils.batch_embed import recommend_embed_batch

            policy = recommend_embed_batch()
            batch = self._batch_size or policy.batch_size
            device = self._device or policy.device
            return batch, device
        except ImportError:
            return self._batch_size or 32, self._device or "cpu"

    async def embed_all(self, texts: List[str]) -> List[Any]:
        if not texts:
            return []
        batch_size, device = self._resolve_policy()
        logger.debug(
            "batch embed",
            extra={"count": len(texts), "batch_size": batch_size, "device": device},
        )

        try:
            from diri_agent_toolbox.agi import batch_embed_items

            chunks = batch_embed_items(texts, batch_size=batch_size)
        except ImportError:
            chunks = [
                texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
            ]

        results: List[Any] = []
        for chunk in chunks:
            results.extend(await self._embed_fn(chunk))
        return results

    async def map_batches(
        self,
        items: List[T],
        fn: Callable[[T], Coroutine[Any, Any, R]],
        *,
        batch_size: Optional[int] = None,
        max_concurrent: int = 4,
    ) -> List[R]:
        """Async map over items with bounded concurrency (primary path)."""
        if not items:
            return []
        bs = batch_size or self._resolve_policy()[0]
        # Prefer direct gather with a semaphore — AsyncBatchProcessor returns
        # BatchProcessingResult (status object), not mapped values.
        sem = asyncio.Semaphore(max(1, min(max_concurrent, bs)))

        async def _one(item: T) -> R:
            async with sem:
                return await fn(item)

        return list(await asyncio.gather(*[_one(item) for item in items]))
