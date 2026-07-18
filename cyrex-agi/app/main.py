"""
cyrex-agi V1 pressure observer — consumes pipeline.pressure.events.

Subscribes via Sugar Glider when available; falls back to Redis XREADGROUP.
Logs pressure events and exposes last-seen state for /health and hooks.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI

app = FastAPI(
    title="Cyrex-AGI",
    description="Autonomous AI system with platform awareness (V1 pressure observer)",
    version="0.2.0",
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
}
_tasks: List[asyncio.Task] = []


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
                pass
        out[key] = value
    return out


async def _consume_via_sidecar(stream: str, counter_key: str, last_key: str) -> None:
    # Prefer in-tree Cyrex sugar-glider client when cyrex-agi is co-located.
    try:
        import sys
        from pathlib import Path

        cyrex_root = Path(__file__).resolve().parents[2]
        if str(cyrex_root) not in sys.path:
            sys.path.insert(0, str(cyrex_root))
        from app.integrations.streaming.synapse_sugar_glider_client import (
            SynapseSidecarClient,
        )
    except Exception as exc:
        raise RuntimeError(f"sidecar client unavailable: {exc}") from exc

    url = (
        os.getenv("SYNAPSE_SUGAR_GLIDER_URL")
        or os.getenv("SYNAPSE_SIDECAR_URL")
        or "http://synapse-sugar-glider:8081"
    )
    client = SynapseSidecarClient(base_url=url, default_sender="cyrex-agi")
    while True:
        try:
            events = await client.read(
                stream=stream,
                consumer_group=CONSUMER_GROUP,
                consumer_name=CONSUMER_NAME,
                count=32,
                block_ms=2000,
            )
            for event in events:
                payload = _parse_fields(event.fields)
                _state[counter_key] = int(_state[counter_key]) + 1
                _state[last_key] = {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "entry_id": event.entry_id,
                    "payload": payload,
                }
                print(
                    f"[cyrex-agi] {stream} event={payload.get('event')} "
                    f"doc={payload.get('document_id')}"
                )
                try:
                    await client.ack(stream, CONSUMER_GROUP, [event.entry_id])
                except Exception:
                    pass
        except Exception as exc:
            _state["errors"] = int(_state["errors"]) + 1
            _state["status"] = f"sidecar_error:{exc}"
            await asyncio.sleep(2.0)


async def _consume_via_redis(stream: str, counter_key: str, last_key: str) -> None:
    import redis.asyncio as redis

    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    client = redis.from_url(redis_url, decode_responses=True)
    try:
        await client.xgroup_create(stream, CONSUMER_GROUP, id="0", mkstream=True)
    except Exception:
        pass

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
        except Exception as exc:
            _state["errors"] = int(_state["errors"]) + 1
            _state["status"] = f"redis_error:{exc}"
            await asyncio.sleep(2.0)


async def _run_consumer(stream: str, counter_key: str, last_key: str) -> None:
    transport = (os.getenv("SYNAPSE_TRANSPORT", "sidecar") or "sidecar").strip().lower()
    if transport == "sidecar":
        try:
            await _consume_via_sidecar(stream, counter_key, last_key)
            return
        except Exception as exc:
            _state["status"] = f"sidecar_init_failed:{exc};falling_back_redis"
    await _consume_via_redis(stream, counter_key, last_key)


@app.on_event("startup")
async def startup() -> None:
    _state["status"] = "running"
    _tasks.append(
        asyncio.create_task(
            _run_consumer(PRESSURE_STREAM, "pressure_events_seen", "last_pressure")
        )
    )
    _tasks.append(
        asyncio.create_task(
            _run_consumer(
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
        "pressure_events_seen": _state["pressure_events_seen"],
        "invalidation_events_seen": _state["invalidation_events_seen"],
        "last_pressure": _state["last_pressure"],
        "last_invalidation": _state["last_invalidation"],
        "errors": _state["errors"],
    }
