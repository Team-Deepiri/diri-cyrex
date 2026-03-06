"""
Monitoring Routes
Performance and health monitoring endpoints
"""
from fastapi import APIRouter, Request, HTTPException, Query
from typing import Optional
from ..services.performance_monitor import get_performance_monitor
from ..agents.metrics import get_agent_metrics_collector
from ..logging_config import get_logger

router = APIRouter()
logger = get_logger("cyrex.monitoring")


@router.get("/monitoring/stats")
async def get_performance_stats(request: Request):
    """Get performance statistics."""
    monitor = get_performance_monitor()
    stats = monitor.get_stats()

    return {
        'success': True,
        'data': stats
    }


@router.get("/monitoring/alerts")
async def get_alerts(request: Request):
    """Get performance alerts."""
    monitor = get_performance_monitor()
    alerts = monitor.check_alerts()

    return {
        'success': True,
        'data': {'alerts': alerts}
    }


# ---------------------------------------------------------------------------
# Agent metrics endpoints
# ---------------------------------------------------------------------------

@router.get("/monitoring/agents")
async def get_all_agent_metrics(request: Request):
    """
    Get aggregated performance metrics for all agents.

    Returns per-agent summary: response time percentiles, success/error rates,
    tool usage counts, and confidence distribution.
    """
    collector = get_agent_metrics_collector()
    summaries = collector.get_all_summaries()
    return {
        "success": True,
        "data": {
            "agent_count": len(summaries),
            "agents": [s.to_dict() for s in summaries],
        },
    }


@router.get("/monitoring/agents/{agent_id}")
async def get_agent_metrics(agent_id: str, request: Request):
    """
    Get aggregated performance metrics for a specific agent.
    """
    collector = get_agent_metrics_collector()
    summary = collector.get_summary(agent_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"No metrics found for agent_id={agent_id}")
    return {
        "success": True,
        "data": summary.to_dict(),
    }


@router.get("/monitoring/agents/{agent_id}/history")
async def get_agent_invoke_history(
    agent_id: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
):
    """
    Get recent raw invoke records for a specific agent.
    Useful for charting latency/confidence over time.
    """
    collector = get_agent_metrics_collector()
    records = collector.get_recent_records(agent_id, limit=limit)
    if not records:
        raise HTTPException(status_code=404, detail=f"No history found for agent_id={agent_id}")
    return {
        "success": True,
        "data": {
            "agent_id": agent_id,
            "record_count": len(records),
            "records": records,
        },
    }


@router.post("/monitoring/agents/flush")
async def flush_agent_metrics(request: Request):
    """
    Persist in-memory agent metrics summary to PostgreSQL.
    Can be called manually or scheduled externally.
    """
    collector = get_agent_metrics_collector()
    await collector.flush_to_db()
    return {"success": True, "message": "Agent metrics flushed to database"}


