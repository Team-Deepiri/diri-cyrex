"""
Universal Real-Time Training Data Storage
==========================================

This is the UNIVERSAL place where all real-time data gets stored for training later.

Data flows:
1. Real-time events → Synapse (Redis Streams) → Training Data Store (CSV/JSONL)
2. Agent events, tasks, tool executions → Training Data Store (CSV/JSONL)
3. Everything here is for training purposes

NOT stored here:
- Conversation messages (goes to PostgreSQL for persistence)
- Agent configs (in-memory or config files)
- Agent instances (in-memory only)
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json
import csv
import asyncio
try:
    from ..integrations.synapse_broker import get_synapse_broker
except ImportError:
    # Fallback if Synapse not available
    async def get_synapse_broker():
        return None
from ..logging_config import get_logger

logger = get_logger("cyrex.training_data_store")


class TrainingDataStore:
    """
    Universal storage for all real-time training data.
    
    Stores:
    - Agent events (for training event prediction models)
    - Agent tasks (for training task completion models)
    - Tool executions (for training tool selection models)
    - Workflow data (for training workflow orchestration)
    - Any other real-time data needed for training
    
    Storage format: CSV/JSONL files organized by data type and date
    Also streams to Synapse for real-time processing
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path("data/training")
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        
        # Create subdirectories
        (self.base_dir / "events").mkdir(exist_ok=True)
        (self.base_dir / "tasks").mkdir(exist_ok=True)
        (self.base_dir / "tool_executions").mkdir(exist_ok=True)
        (self.base_dir / "workflows").mkdir(exist_ok=True)
    
    async def store_agent_event(
        self,
        event_type: str,
        agent_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        session_id: Optional[str] = None,
        source: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        severity: str = "info",
    ) -> str:
        """
        Store agent event to training data store.
        Also publishes to Synapse for real-time streaming.
        """
        event_id = f"evt-{int(datetime.utcnow().timestamp())}-{id(self)}"
        
        event_data = {
            "event_id": event_id,
            "event_type": event_type,
            "agent_id": agent_id,
            "workflow_id": workflow_id,
            "session_id": session_id,
            "source": source or "cyrex",
            "payload": payload or {},
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Publish to Synapse for real-time streaming
        try:
            broker = await get_synapse_broker()
            if broker:
                await broker.publish(
                    channel="training-data.agent_events",
                    payload=event_data,
                    sender=source or "system",
                    headers={"data_type": "agent_event"},
                )
        except Exception as e:
            self.logger.warning(f"Failed to publish to Synapse: {e}")
        
        # Store to CSV for training
        await self._append_to_csv(
            "events",
            event_data,
            ["event_id", "event_type", "agent_id", "workflow_id", "session_id", 
             "source", "severity", "timestamp", "payload"]
        )
        
        self.logger.debug(f"Stored agent event: {event_type}")
        return event_id
    
    async def store_agent_task(
        self,
        task_id: str,
        agent_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        task_type: str = "unknown",
        priority: str = "normal",
        status: str = "pending",
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        retry_count: int = 0,
        max_retries: int = 3,
        timeout_seconds: int = 300,
        execution_time_ms: Optional[float] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> None:
        """Store agent task to training data store"""
        task_data = {
            "task_id": task_id,
            "agent_id": agent_id,
            "instance_id": instance_id,
            "task_type": task_type,
            "priority": priority,
            "status": status,
            "input_data": input_data or {},
            "output_data": output_data,
            "error": error,
            "retry_count": retry_count,
            "execution_time_ms": execution_time_ms,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Publish to Synapse
        try:
            broker = await get_synapse_broker()
            if broker:
                await broker.publish(
                    channel="training-data.agent_tasks",
                    payload=task_data,
                    sender=agent_id or "system",
                    headers={"data_type": "agent_task"},
                )
        except Exception as e:
            self.logger.warning(f"Failed to publish task to Synapse: {e}")
        
        # Store to CSV
        await self._append_to_csv(
            "tasks",
            task_data,
            ["task_id", "agent_id", "instance_id", "task_type", "priority", 
             "status", "retry_count", "execution_time_ms", "timestamp", 
             "input_data", "output_data", "error"]
        )
        
        self.logger.debug(f"Stored agent task: {task_id}")
    
    async def store_tool_execution(
        self,
        execution_id: str,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        tool_name: str = "unknown",
        tool_category: Optional[str] = None,
        input_params: Optional[Dict[str, Any]] = None,
        output_result: Optional[Dict[str, Any]] = None,
        status: str = "completed",
        error: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        completed_at: Optional[datetime] = None,
    ) -> None:
        """Store tool execution to training data store"""
        execution_data = {
            "execution_id": execution_id,
            "agent_id": agent_id,
            "task_id": task_id,
            "tool_name": tool_name,
            "tool_category": tool_category,
            "input_params": input_params or {},
            "output_result": output_result,
            "status": status,
            "error": error,
            "execution_time_ms": execution_time_ms,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Publish to Synapse
        try:
            broker = await get_synapse_broker()
            if broker:
                await broker.publish(
                    channel="training-data.tool_executions",
                    payload=execution_data,
                    sender=agent_id or "system",
                    headers={"data_type": "tool_execution"},
                )
        except Exception as e:
            self.logger.warning(f"Failed to publish tool execution to Synapse: {e}")
        
        # Store to CSV
        await self._append_to_csv(
            "tool_executions",
            execution_data,
            ["execution_id", "agent_id", "task_id", "tool_name", "tool_category",
             "status", "execution_time_ms", "timestamp", "input_params", 
             "output_result", "error"]
        )
        
        self.logger.debug(f"Stored tool execution: {tool_name}")
    
    async def store_workflow_data(
        self,
        workflow_id: str,
        workflow_type: Optional[str] = None,
        phase: Optional[str] = None,
        state_data: Optional[Dict[str, Any]] = None,
        step_results: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        current_step: Optional[str] = None,
        total_steps: int = 0,
        completed_steps: int = 0,
        assigned_agents: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Store workflow data to training data store"""
        workflow_data = {
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "phase": phase,
            "state_data": state_data or {},
            "step_results": step_results or {},
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Publish to Synapse
        try:
            broker = await get_synapse_broker()
            if broker:
                await broker.publish(
                    channel="training-data.workflows",
                    payload=workflow_data,
                    sender="workflow-system",
                    headers={"data_type": "workflow"},
                )
        except Exception as e:
            self.logger.warning(f"Failed to publish workflow to Synapse: {e}")
        
        # Store to CSV
        await self._append_to_csv(
            "workflows",
            workflow_data,
            ["workflow_id", "workflow_type", "phase", "timestamp", 
             "state_data", "step_results", "error"]
        )
        
        self.logger.debug(f"Stored workflow data: {workflow_id}")
    
    async def _append_to_csv(
        self,
        data_type: str,
        data: Dict[str, Any],
        columns: List[str],
    ) -> None:
        """Append data to CSV file (creates file if doesn't exist)"""
        # Use date-based file naming
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        csv_file = self.base_dir / data_type / f"{data_type}_{date_str}.csv"
        
        # Check if file exists to determine if we need to write headers
        file_exists = csv_file.exists()
        
        # Write to CSV
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            # Convert dict values to JSON strings for CSV
            csv_row = {}
            for key in columns:
                value = data.get(key)
                if isinstance(value, dict):
                    csv_row[key] = json.dumps(value)
                else:
                    csv_row[key] = value
            writer.writerow(csv_row)
    
    def export_for_training(
        self,
        data_type: str,
        output_path: Optional[Path] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Path:
        """
        Export training data in JSONL format (standard for ML training).
        Returns path to exported file.
        """
        if output_path is None:
            output_path = self.base_dir / f"{data_type}_training.jsonl"
        
        data_dir = self.base_dir / data_type
        if not data_dir.exists():
            self.logger.warning(f"No data directory found: {data_dir}")
            return output_path
        
        # Collect all CSV files in date range
        csv_files = sorted(data_dir.glob("*.csv"))
        
        if start_date:
            csv_files = [f for f in csv_files if f.stem.split("_")[-1] >= start_date]
        if end_date:
            csv_files = [f for f in csv_files if f.stem.split("_")[-1] <= end_date]
        
        # Convert CSV to JSONL
        with open(output_path, "w", encoding="utf-8") as f:
            for csv_file in csv_files:
                with open(csv_file, "r", encoding="utf-8") as infile:
                    reader = csv.DictReader(infile)
                    for row in reader:
                        # Parse JSON fields
                        for key, value in row.items():
                            if value and (key.endswith("_data") or key in ["payload", "input_params", "output_result", "input_data", "output_data", "state_data", "step_results"]):
                                try:
                                    row[key] = json.loads(value)
                                except:
                                    pass
                        f.write(json.dumps(row) + "\n")
        
        self.logger.info(f"Exported {data_type} training data to {output_path}")
        return output_path


# Singleton instance
_training_data_store: Optional[TrainingDataStore] = None


def get_training_data_store() -> TrainingDataStore:
    """Get or create training data store singleton"""
    global _training_data_store
    if _training_data_store is None:
        _training_data_store = TrainingDataStore()
    return _training_data_store

