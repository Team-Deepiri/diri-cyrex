"""Eyes HTTP routes — Cyrex reads Elkedel's continuous vision pipeline."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Query

from app.integrations.elkedel import get_elkedel_client

router = APIRouter(prefix="/api/v1/eyes", tags=["eyes"])


@router.get("/status")
async def eyes_status() -> Dict[str, Any]:
    """``elkedel.eyes_status`` — pipeline running state + camera."""
    return await get_elkedel_client().eyes_status()


@router.post("/start")
async def eyes_start() -> Dict[str, Any]:
    """``elkedel.eyes_start`` — start camera → YOLO → memory loop."""
    return await get_elkedel_client().eyes_start()


@router.post("/stop")
async def eyes_stop() -> Dict[str, Any]:
    return await get_elkedel_client().eyes_stop()


@router.get("/scene")
async def eyes_scene(top_k: int = Query(20, ge=1, le=100)) -> Dict[str, Any]:
    """Active identities the eyes currently see."""
    return await get_elkedel_client().eyes_scene(top_k=top_k)


@router.get("/events")
async def eyes_events(
    since_ms: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
) -> Dict[str, Any]:
    return await get_elkedel_client().eyes_events(since_ms=since_ms, limit=limit)


@router.get("/where")
async def eyes_where(
    q: str = Query(""),
    top_k: int = Query(5, ge=1, le=50),
) -> Dict[str, Any]:
    return await get_elkedel_client().eyes_where(query=q, top_k=top_k)
