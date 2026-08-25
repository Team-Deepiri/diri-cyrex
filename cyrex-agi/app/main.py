"""
cyrex-agi V1 pressure observer — consumes pipeline.pressure.events.

Prefer Sugar Glider HTTP `/v1/read` + `/v1/ack` when configured; fall back to
direct Redis Streams consumer groups (same substrate Sugar Glider writes to).
Producers publish through Sugar Glider; this observer does not import the
Cyrex app package (avoids brittle sys.path hacks).
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI

app = FastAPI(
    title="Cyrex-AGI",
    description="Autonomous AI system with platform awareness (V1 pressure observer)",
    version="0.2.1",
)

PRESSURE_STREAM = "pipeline.pressure.events"
INVALIDATION_STREAM = "pipeline.artifact.invalidation"
CONSUMER_GROUP = "cyrex-agi-pressure"
CONSUMER_NAME = os.getenv("CYREX_AGI_CONSUMER_NAME", "cyrex-agi-1")

_state: Dict[str, Any] = {
    "status": "starting",
    "pressure_events_seen": 0,
    "invalidation_events_seen": 0,
    "last_pressure": None,
    "last_invalidation": None,
    "errors": 0,
    "transport": "starting",
}
_tasks: List[asyncio.Task] = []


def _sugar_glider_url() -> Optional[str]:
    transport = (os.getenv("SYNAPSE_TRANSPORT", "sidecar") or "sidecar").strip().lower()
    if transport in {"redis", "direct"}:
        return None
    url = (
        os.getenv("SYNAPSE_SUGAR_GLIDER_URL")
        or os.getenv("SYNAPSE_SIDECAR_URL")
        or ""
    ).rstrip("/")
    return url or None


def _parse_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, str) and value.startswith(("{", "[")):
            try:
                out[key] = json.loads(value)
                continue
            except json.JSONDecodeError:
                _state["errors"] += 1
        out[key] = value
    inner = out.get("payload")
    if isinstance(inner, str) and inner.startswith(("{", "[")):
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                return {**out, **parsed}
        except json.JSONDecodeError:
            _state["errors"] += 1
    if isinstance(inner, dict):
        return {**out, **inner}
    return out


async def _sugar_glider_ready(client: httpx.AsyncClient, base_url: str) -> bool:
    for path in ("/readyz", "/healthz", "/health"):
        try:
            resp = await client.get(f"{base_url}{path}", timeout=2.0)
            if 200 <= resp.status_code < 300:
                return True
        except asyncio.CancelledError:
            raise
        except Exception:
            continue
    return False


async def _consume_via_sugar_glider(
    stream: str, counter_key: str, last_key: str, base_url: str
) -> None:
    async with httpx.AsyncClient() as client:
        while True:
            try:
                if not await _sugar_glider_ready(client, base_url):
                    _state["status"] = "sugar_glider_not_ready"
                    await asyncio.sleep(2.0)
                    continue

                read_resp = await client.post(
                    f"{base_url}/v1/read",
                    json={
                        "stream": stream,
                        "consumer_group": CONSUMER_GROUP,
                        "consumer_name": CONSUMER_NAME,
                        "count": 32,
                        "block_ms": 2000,
                    },
                    timeout=5.0,
                )
                read_resp.raise_for_status()
                events = read_resp.json().get("events") or []
                entry_ids: List[str] = []
                for event in events:
                    entry_id = event.get("entry_id") or event.get("entry")
                    fields = event.get("fields") or {}
                    payload = _parse_fields(fields)
                    _state[counter_key] = int(_state[counter_key]) + 1
                    _state[last_key] = {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "entry_id": entry_id,
                        "payload": payload,
                    }
                    print(
                        f"[cyrex-agi] {stream} event={payload.get('event')} "
                        f"doc={payload.get('document_id')}"
                    )
                    if entry_id:
                        entry_ids.append(str(entry_id))

                if entry_ids:
                    ack_resp = await client.post(
                        f"{base_url}/v1/ack",
                        json={
                            "stream": stream,
                            "consumer_group": CONSUMER_GROUP,
                            "entry_ids": entry_ids,
                        },
                        timeout=5.0,
                    )
                    ack_resp.raise_for_status()
                _state["status"] = "running"
                _state["transport"] = "sugar-glider-http"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _state["errors"] = int(_state["errors"]) + 1
                _state["status"] = f"sugar_glider_error:{exc}"
                await asyncio.sleep(2.0)


async def _consume_via_redis(stream: str, counter_key: str, last_key: str) -> None:
    import redis.asyncio as redis

    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    client = redis.from_url(redis_url, decode_responses=True)
    try:
        await client.xgroup_create(stream, CONSUMER_GROUP, id="0", mkstream=True)
    except Exception as exc:
        if "BUSYGROUP" in str(exc):
            pass
        else:
            raise

    _state["transport"] = "redis-streams"
    while True:
        try:
            rows = await client.xreadgroup(
                CONSUMER_GROUP,
                CONSUMER_NAME,
                streams={stream: ">"},
                count=32,
                block=2000,
            )
            if not rows:
                continue
            for _stream_name, messages in rows:
                for entry_id, fields in messages:
                    payload = _parse_fields(fields)
                    _state[counter_key] = int(_state[counter_key]) + 1
                    _state[last_key] = {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "entry_id": entry_id,
                        "payload": payload,
                    }
                    print(
                        f"[cyrex-agi] {stream} event={payload.get('event')} "
                        f"doc={payload.get('document_id')}"
                    )
                    await client.xack(stream, CONSUMER_GROUP, entry_id)
            _state["status"] = "running"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _state["errors"] = int(_state["errors"]) + 1
            _state["status"] = f"redis_error:{exc}"
            await asyncio.sleep(2.0)


async def _consume(stream: str, counter_key: str, last_key: str) -> None:
    base_url = _sugar_glider_url()
    if base_url:
        await _consume_via_sugar_glider(stream, counter_key, last_key, base_url)
        return
    await _consume_via_redis(stream, counter_key, last_key)


@app.on_event("startup")
async def startup() -> None:
    _state["status"] = "running"
    _state["transport"] = (
        "sugar-glider-http" if _sugar_glider_url() else "redis-streams"
    )
    _tasks.append(
        asyncio.create_task(
            _consume(PRESSURE_STREAM, "pressure_events_seen", "last_pressure")
        )
    )
    _tasks.append(
        asyncio.create_task(
            _consume(
                INVALIDATION_STREAM, "invalidation_events_seen", "last_invalidation"
            )
        )
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    for task in _tasks:
        task.cancel()
    _state["status"] = "stopped"


@app.get("/health")
async def health():
    return {
        "status": _state["status"],
        "service": "cyrex-agi",
        "phase": "v1-pressure-observer",
        "consumer_group": CONSUMER_GROUP,
        "streams": [PRESSURE_STREAM, INVALIDATION_STREAM],
        "transport": _state.get("transport"),
        "pressure_events_seen": _state["pressure_events_seen"],
        "invalidation_events_seen": _state["invalidation_events_seen"],
        "last_pressure": _state["last_pressure"],
        "last_invalidation": _state["last_invalidation"],
        "errors": _state["errors"],
    }
