"""Batch embedding service — gpu-utils policy + toolbox batch processor."""

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
            from diri_agent_toolbox.processing import AsyncBatchProcessor, BatchProcessingConfig

            chunks = batch_embed_items(texts, batch_size=batch_size)
            processor = AsyncBatchProcessor(
                BatchProcessingConfig(batch_size=1, max_concurrent_batches=4)
            )

            async def _one(batch: List[str]) -> List[Any]:
                return await self._embed_fn(batch)

            results: List[Any] = []
            for chunk in chunks:
                vecs = await _one(chunk)
                results.extend(vecs)
            return results
        except ImportError:
            out: List[Any] = []
            for i in range(0, len(texts), batch_size):
                out.extend(await self._embed_fn(texts[i : i + batch_size]))
            return out

    async def map_batches(
        self,
        items: List[T],
        fn: Callable[[T], Coroutine[Any, Any, R]],
        *,
        batch_size: Optional[int] = None,
    ) -> List[R]:
        """Generic async batch map using toolbox processor when available."""
        bs = batch_size or self._resolve_policy()[0]
        try:
            from diri_agent_toolbox.processing import AsyncBatchProcessor, BatchProcessingConfig

            processor = AsyncBatchProcessor(BatchProcessingConfig(batch_size=bs))
            result = await processor.process_batch(items, fn)
            if result.failed_items:
                logger.warning("batch map failures", extra={"failed": result.failed_items})
            # process_batch returns BatchProcessingResult not values — fallback
        except ImportError:
            pass
        return await asyncio.gather(*[fn(item) for item in items])
