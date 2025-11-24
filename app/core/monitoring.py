"""
System Monitor
Tracks cost, latency, drift, safety scores, and behavior analytics
Production-grade observability for AI systems
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
import statistics
from ..logging_config import get_logger

logger = get_logger("cyrex.monitoring")


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    request_id: str
    user_id: Optional[str]
    timestamp: datetime
    duration_ms: float
    tokens_used: int
    model: str
    cost: float = 0.0
    safety_score: float = 0.0
    error: Optional[str] = None


@dataclass
class ModelMetrics:
    """Aggregated metrics for a model"""
    model_name: str
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    safety_scores: List[float] = field(default_factory=list)


class SystemMonitor:
    """
    Comprehensive monitoring system for AI operations
    Tracks cost, latency, drift, safety, and behavior
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.request_history: deque = deque(maxlen=window_size)
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.error_log: deque = deque(maxlen=window_size)
        self.cost_tracking: Dict[str, float] = defaultdict(float)
        self.latency_tracking: Dict[str, List[float]] = defaultdict(list)
        self.safety_tracking: List[float] = []
        self.logger = logger
    
    def record_request(
        self,
        request_id: str,
        user_id: Optional[str] = None,
        duration_ms: float = 0.0,
        tokens_used: int = 0,
        model: str = "unknown",
        cost: float = 0.0,
        safety_score: float = 0.0,
        error: Optional[str] = None,
    ):
        """Record a request with metrics"""
        metrics = RequestMetrics(
            request_id=request_id,
            user_id=user_id,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            model=model,
            cost=cost,
            safety_score=safety_score,
            error=error,
        )
        
        self.request_history.append(metrics)
        
        # Update model metrics
        if model not in self.model_metrics:
            self.model_metrics[model] = ModelMetrics(model_name=model)
        
        model_metric = self.model_metrics[model]
        model_metric.total_requests += 1
        model_metric.total_tokens += tokens_used
        model_metric.total_cost += cost
        model_metric.safety_scores.append(safety_score)
        
        # Update latency tracking
        self.latency_tracking[model].append(duration_ms)
        if len(self.latency_tracking[model]) > self.window_size:
            self.latency_tracking[model] = self.latency_tracking[model][-self.window_size:]
        
        # Update average latency
        model_metric.avg_latency_ms = statistics.mean(self.latency_tracking[model])
        
        # Update error rate
        if error:
            model_metric.error_rate = (
                (model_metric.error_rate * (model_metric.total_requests - 1) + 1.0) /
                model_metric.total_requests
            )
            self.error_log.append({
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "error": error,
                "model": model,
            })
        else:
            model_metric.error_rate = (
                (model_metric.error_rate * (model_metric.total_requests - 1)) /
                model_metric.total_requests
            )
        
        # Track safety scores
        self.safety_tracking.append(safety_score)
        if len(self.safety_tracking) > self.window_size:
            self.safety_tracking = self.safety_tracking[-self.window_size:]
        
        # Track costs
        self.cost_tracking[model] += cost
    
    def record_error(self, request_id: str, error: str, model: str = "unknown"):
        """Record an error"""
        self.error_log.append({
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "model": model,
        })
        self.logger.error(f"Error recorded: {error} (request: {request_id})")
    
    def get_stats(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a model or all models"""
        if model:
            if model not in self.model_metrics:
                return {"error": f"Model {model} not found"}
            
            metric = self.model_metrics[model]
            latencies = self.latency_tracking.get(model, [])
            
            return {
                "model": model,
                "total_requests": metric.total_requests,
                "total_tokens": metric.total_tokens,
                "total_cost": metric.total_cost,
                "avg_latency_ms": metric.avg_latency_ms,
                "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
                "p95_latency_ms": self._percentile(latencies, 95) if latencies else 0.0,
                "p99_latency_ms": self._percentile(latencies, 99) if latencies else 0.0,
                "error_rate": metric.error_rate,
                "avg_safety_score": statistics.mean(metric.safety_scores) if metric.safety_scores else 0.0,
            }
        else:
            # All models
            return {
                "models": {
                    model: self.get_stats(model)
                    for model in self.model_metrics.keys()
                },
                "total_requests": len(self.request_history),
                "recent_errors": list(self.error_log)[-10:],
            }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def detect_drift(
        self,
        model: str,
        baseline_window: int = 100,
        current_window: int = 100,
    ) -> Dict[str, Any]:
        """
        Detect performance drift by comparing recent vs baseline metrics
        
        Returns:
            Drift detection results
        """
        if model not in self.model_metrics:
            return {"error": f"Model {model} not found"}
        
        # Get recent requests
        recent_requests = [
            r for r in self.request_history
            if r.model == model
        ][-current_window:]
        
        if len(recent_requests) < current_window:
            return {"error": "Insufficient data for drift detection"}
        
        # Calculate recent metrics
        recent_latencies = [r.duration_ms for r in recent_requests]
        recent_safety = [r.safety_score for r in recent_requests]
        recent_errors = sum(1 for r in recent_requests if r.error)
        
        # Get baseline (older requests)
        baseline_requests = [
            r for r in self.request_history
            if r.model == model
        ][-baseline_window - current_window:-current_window]
        
        if len(baseline_requests) < baseline_window:
            return {"error": "Insufficient baseline data"}
        
        baseline_latencies = [r.duration_ms for r in baseline_requests]
        baseline_safety = [r.safety_score for r in baseline_requests]
        baseline_errors = sum(1 for r in baseline_requests if r.error)
        
        # Calculate drift
        latency_drift = (
            (statistics.mean(recent_latencies) - statistics.mean(baseline_latencies)) /
            statistics.mean(baseline_latencies) * 100
        )
        
        safety_drift = (
            statistics.mean(recent_safety) - statistics.mean(baseline_safety)
        )
        
        error_drift = (
            (recent_errors / len(recent_requests)) -
            (baseline_errors / len(baseline_requests))
        ) * 100
        
        return {
            "model": model,
            "latency_drift_pct": latency_drift,
            "safety_drift": safety_drift,
            "error_drift_pct": error_drift,
            "drift_detected": abs(latency_drift) > 20 or abs(safety_drift) > 0.1 or abs(error_drift) > 5,
        }
    
    def get_cost_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get cost summary for a time period"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
        
        requests = [
            r for r in self.request_history
            if start_date <= r.timestamp <= end_date
        ]
        
        total_cost = sum(r.cost for r in requests)
        cost_by_model = defaultdict(float)
        
        for r in requests:
            cost_by_model[r.model] += r.cost
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "total_cost": total_cost,
            "total_requests": len(requests),
            "cost_by_model": dict(cost_by_model),
            "avg_cost_per_request": total_cost / len(requests) if requests else 0.0,
        }
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Get safety score report"""
        if not self.safety_tracking:
            return {"error": "No safety data available"}
        
        return {
            "avg_safety_score": statistics.mean(self.safety_tracking),
            "min_safety_score": min(self.safety_tracking),
            "max_safety_score": max(self.safety_tracking),
            "p95_safety_score": self._percentile(self.safety_tracking, 95),
            "unsafe_requests": sum(1 for s in self.safety_tracking if s > 0.6),
            "total_requests": len(self.safety_tracking),
        }


def get_monitor() -> SystemMonitor:
    """Get global monitor instance"""
    return SystemMonitor()

