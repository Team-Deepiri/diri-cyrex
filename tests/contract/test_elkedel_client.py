"""Unit tests for ElkedelClient (httpx MockTransport — no live Elkedel)."""

from __future__ import annotations

import json

import httpx
import pytest

from app.integrations.elkedel.client import ElkedelClient


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/health":
        return httpx.Response(200, json={"ok": True, "service": "elkedel-runtime"})
    if path == "/memory/stats":
        return httpx.Response(200, json={"traces": 2, "observations": 5})
    if path == "/v1/tools":
        return httpx.Response(
            200, json={"prefix": "elkedel", "tools": [{"name": "elkedel.stats"}]}
        )
    if path == "/memory/remember":
        return httpx.Response(
            200, json={"assigned": 0, "spawned": 1, "total_traces": 1}
        )
    if path == "/v1/eyes/status":
        return httpx.Response(
            200, json={"running": True, "frames": 3, "active_identities": 1}
        )
    if path == "/v1/eyes/scene":
        return httpx.Response(200, json={"count": 1, "identities": []})
    if path == "/v1/eyes/start":
        return httpx.Response(200, json={"running": True})
    return httpx.Response(404, json={"error": "not found"})


@pytest.mark.asyncio
async def test_elkedel_stats_and_tools():
    transport = httpx.MockTransport(_handler)
    client = ElkedelClient(base_url="http://elkedel-test:8765", api_key="k", timeout=5.0)
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        headers={"x-api-key": "k"},
        timeout=5.0,
        transport=transport,
    )
    try:
        health = await client.health()
        assert health["ok"] is True
        stats = await client.stats()
        assert stats["traces"] == 2
        tools = await client.list_tools()
        assert tools["prefix"] == "elkedel"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_elkedel_remember():
    transport = httpx.MockTransport(_handler)
    client = ElkedelClient(base_url="http://elkedel-test:8765")
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        timeout=5.0,
        transport=transport,
    )
    try:
        out = await client.remember(b"\xff\xd8fakejpeg")
        assert out["spawned"] == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_elkedel_eyes():
    transport = httpx.MockTransport(_handler)
    client = ElkedelClient(base_url="http://elkedel-test:8765")
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        timeout=5.0,
        transport=transport,
    )
    try:
        st = await client.eyes_status()
        assert st["running"] is True
        scene = await client.eyes_scene()
        assert scene["count"] == 1
        started = await client.eyes_start()
        assert started["running"] is True
    finally:
        await client.close()
