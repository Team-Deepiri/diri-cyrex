"""Elkedel client — Cyrex → episodic visual memory / sensory cortex.

Elkedel is a separate service (not folded into ``cyrex.*`` MCP tools).
Call it over HTTP using the namespaced tool contract documented in
deepiri-elkedel ``docs/MCP.md`` (``elkedel.remember``, ``elkedel.stats``, …).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("cyrex.elkedel")


def _settings_value(name: str, default: Any) -> Any:
    try:
        from ...settings import settings

        return getattr(settings, name, default)
    except Exception:
        return os.environ.get(name, default)


class ElkedelClient:
    """HTTP client for the Elkedel runtime (``:8765``)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        raw_base = base_url or _settings_value("ELKEDEL_BASE_URL", "http://elkedel:8765")
        self.base_url = str(raw_base).rstrip("/")
        self.api_key = (
            api_key
            if api_key is not None
            else _settings_value("ELKEDEL_API_KEY", None)
        )
        raw_timeout = timeout if timeout is not None else _settings_value(
            "ELKEDEL_TIMEOUT_SEC", 30.0
        )
        self.timeout = float(raw_timeout)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers: Dict[str, str] = {}
            if self.api_key:
                headers["x-api-key"] = str(self.api_key)
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def health(self) -> Dict[str, Any]:
        client = await self._get_client()
        r = await client.get("/health")
        r.raise_for_status()
        return r.json()

    async def ready(self) -> Dict[str, Any]:
        client = await self._get_client()
        r = await client.get("/ready")
        r.raise_for_status()
        return r.json()

    async def list_tools(self) -> Dict[str, Any]:
        """Discovery — namespaced ``elkedel.*`` tool map."""
        client = await self._get_client()
        r = await client.get("/v1/tools")
        r.raise_for_status()
        return r.json()

    async def stats(self) -> Dict[str, Any]:
        """``elkedel.stats``"""
        client = await self._get_client()
        r = await client.get("/memory/stats")
        r.raise_for_status()
        return r.json()

    async def remember(self, jpeg_bytes: bytes) -> Dict[str, Any]:
        """``elkedel.remember`` — ingest one JPEG frame."""
        client = await self._get_client()
        r = await client.post(
            "/memory/remember",
            content=jpeg_bytes,
            headers={"Content-Type": "image/jpeg"},
        )
        r.raise_for_status()
        return r.json()

    async def where(self, query: str = "", top_k: int = 5) -> Dict[str, Any]:
        """``elkedel.where``"""
        client = await self._get_client()
        r = await client.get(
            "/memory/recall",
            params={"q": query, "top_k": top_k},
        )
        r.raise_for_status()
        return r.json()

    async def what_changed(self, since_ms: int = 0) -> Dict[str, Any]:
        """``elkedel.what_changed``"""
        client = await self._get_client()
        r = await client.get("/memory/changed", params={"since_ms": since_ms})
        r.raise_for_status()
        return r.json()

    async def episode(self, identity_id: str) -> Dict[str, Any]:
        """``elkedel.episode``"""
        client = await self._get_client()
        r = await client.get(f"/memory/episode/{identity_id}")
        r.raise_for_status()
        return r.json()

    async def forget(self, identity_id: str) -> Dict[str, Any]:
        """``elkedel.forget``"""
        client = await self._get_client()
        r = await client.post(f"/memory/forget/{identity_id}")
        r.raise_for_status()
        return r.json()

    async def sensory_stats(self) -> Dict[str, Any]:
        """``elkedel.sensory_stats``"""
        client = await self._get_client()
        r = await client.get("/sensory/stats")
        r.raise_for_status()
        return r.json()

    async def perceive(
        self, jpeg_bytes: bytes, *, persist: bool = False
    ) -> Dict[str, Any]:
        """``elkedel.perceive`` — Physics Tensor from one frame."""
        client = await self._get_client()
        params = {"persist": "1"} if persist else None
        r = await client.post(
            "/sensory/perceive",
            content=jpeg_bytes,
            headers={"Content-Type": "image/jpeg"},
            params=params,
        )
        r.raise_for_status()
        return r.json()

    async def replay(
        self, limit: int = 100, min_novelty: float = 0.0
    ) -> Dict[str, Any]:
        """``elkedel.replay``"""
        client = await self._get_client()
        r = await client.get(
            "/sensory/replay",
            params={"limit": limit, "min_novelty": min_novelty},
        )
        r.raise_for_status()
        return r.json()

    # --- Eyes (continuous vision pipeline) ---------------------------------

    async def eyes_status(self) -> Dict[str, Any]:
        """``elkedel.eyes_status``"""
        client = await self._get_client()
        r = await client.get("/v1/eyes/status")
        r.raise_for_status()
        return r.json()

    async def eyes_start(self) -> Dict[str, Any]:
        """``elkedel.eyes_start``"""
        client = await self._get_client()
        r = await client.post("/v1/eyes/start")
        r.raise_for_status()
        return r.json()

    async def eyes_stop(self) -> Dict[str, Any]:
        """``elkedel.eyes_stop``"""
        client = await self._get_client()
        r = await client.post("/v1/eyes/stop")
        r.raise_for_status()
        return r.json()

    async def eyes_scene(self, top_k: int = 20) -> Dict[str, Any]:
        """``elkedel.eyes_scene``"""
        client = await self._get_client()
        r = await client.get("/v1/eyes/scene", params={"top_k": top_k})
        r.raise_for_status()
        return r.json()

    async def eyes_events(
        self, since_ms: int = 0, limit: int = 50
    ) -> Dict[str, Any]:
        """``elkedel.eyes_events``"""
        client = await self._get_client()
        r = await client.get(
            "/v1/eyes/events",
            params={"since_ms": since_ms, "limit": limit},
        )
        r.raise_for_status()
        return r.json()

    async def eyes_where(self, query: str = "", top_k: int = 5) -> Dict[str, Any]:
        """``elkedel.eyes_where`` — text query against live scene memory."""
        client = await self._get_client()
        r = await client.get(
            "/v1/eyes/where",
            params={"q": query, "top_k": top_k},
        )
        r.raise_for_status()
        return r.json()

    async def eyes_changed(self, since_ms: int = 0) -> Dict[str, Any]:
        """``elkedel.what_changed`` via eyes pipeline memory."""
        client = await self._get_client()
        r = await client.get("/v1/eyes/changed", params={"since_ms": since_ms})
        r.raise_for_status()
        return r.json()


_default_client: Optional[ElkedelClient] = None


def get_elkedel_client() -> ElkedelClient:
    global _default_client
    if _default_client is None:
        _default_client = ElkedelClient()
    return _default_client


def visual_artifact_from_trace(trace: Dict[str, Any], *, query: str | None = None) -> Dict[str, Any]:
    """Map an Elkedel trace dict into a Cyrex VisualObservation artifact."""
    return {
        "artifact_type": "VisualObservation",
        "modality": "vision",
        "source": "elkedel",
        "identity_id": trace.get("trace_id") or trace.get("identity_id"),
        "label": trace.get("label"),
        "strength": trace.get("strength"),
        "n_observations": trace.get("n_observations"),
        "history": trace.get("history") or [],
        "citation": {
            "kind": "frame_timestamp",
            "ts_ms": trace.get("last_seen_ms") or trace.get("first_seen_ms"),
            "query": query,
        },
    }
