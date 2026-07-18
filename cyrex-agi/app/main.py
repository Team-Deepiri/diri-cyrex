"""
cyrex-agi V1 pressure observer — consumes pipeline.pressure.events.

Consumes via Redis Streams consumer groups (same substrate Sugar Glider
writes to). Producers publish through Sugar Glider; this observer does not
import the Cyrex app package (avoids brittle sys.path hacks).
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

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
    # Sugar Glider often wraps JSON under "payload"
    inner = out.get("payload")
    if isinstance(inner, str) and inner.startswith(("{", "[")):
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                return {**out, **parsed}
        except json.JSONDecodeError:
            pass
    if isinstance(inner, dict):
        return {**out, **inner}
    return out


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


@app.on_event("startup")
async def startup() -> None:
    _state["status"] = "running"
    _tasks.append(
        asyncio.create_task(
            _consume_via_redis(PRESSURE_STREAM, "pressure_events_seen", "last_pressure")
        )
    )
    _tasks.append(
        asyncio.create_task(
            _consume_via_redis(
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
        "transport": "redis-streams (Sugar Glider publish substrate)",
        "pressure_events_seen": _state["pressure_events_seen"],
        "invalidation_events_seen": _state["invalidation_events_seen"],
        "last_pressure": _state["last_pressure"],
        "last_invalidation": _state["last_invalidation"],
        "errors": _state["errors"],
    }
