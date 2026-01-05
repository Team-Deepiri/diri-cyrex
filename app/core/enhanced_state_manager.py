"""
Enhanced State Management for Agent Workflows
Advanced state tracking with checkpoints, rollback, and distributed coordination
"""
from typing import Dict, Any, Optional, List, Callable, Type, TypeVar
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
import json
import asyncio
import hashlib
import pickle
from pathlib import Path
import redis.asyncio as aioredis
from pydantic import BaseModel, Field
from ..logging_config import get_logger
from ..settings import settings

logger = get_logger("cyrex.enhanced_state_manager")

T = TypeVar('T')


class WorkflowPhase(str, Enum):
    """Workflow execution phases"""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    VALIDATING = "validating"
    COMPLETING = "completing"
    ROLLING_BACK = "rolling_back"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StateChangeType(str, Enum):
    """Types of state changes"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    CHECKPOINT = "checkpoint"
    ROLLBACK = "rollback"
    TRANSITION = "transition"


@dataclass
class StateCheckpoint:
    """Checkpoint for state recovery"""
    checkpoint_id: str
    workflow_id: str
    phase: WorkflowPhase
    step_name: str
    state_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "workflow_id": self.workflow_id,
            "phase": self.phase.value,
            "step_name": self.step_name,
            "state_data": self.state_data,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateCheckpoint':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['phase'] = WorkflowPhase(data['phase'])
        return cls(**data)


@dataclass
class StateTransition:
    """Record of state transition"""
    transition_id: str
    from_phase: WorkflowPhase
    to_phase: WorkflowPhase
    trigger: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)


class WorkflowState(BaseModel):
    """Complete workflow state with history"""
    workflow_id: str
    name: str = ""
    phase: WorkflowPhase = WorkflowPhase.INITIALIZING
    current_step: Optional[str] = None
    
    # State data
    state_data: Dict[str, Any] = Field(default_factory=dict)
    step_results: Dict[str, Any] = Field(default_factory=dict)
    
    # Agent assignments
    assigned_agents: List[str] = Field(default_factory=list)
    agent_states: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Progress tracking
    total_steps: int = 0
    completed_steps: int = 0
    progress_percentage: float = 0.0
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    
    # Error handling
    error: Optional[str] = None
    error_count: int = 0
    
    # History
    checkpoints: List[Dict[str, Any]] = Field(default_factory=list)
    transitions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class StateChangeEvent(BaseModel):
    """Event emitted on state change"""
    event_id: str
    workflow_id: str
    change_type: StateChangeType
    old_phase: Optional[WorkflowPhase] = None
    new_phase: Optional[WorkflowPhase] = None
    changed_fields: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EnhancedStateManager:
    """
    Advanced workflow state management with:
    - Redis-backed persistence
    - Checkpointing and rollback
    - State transitions with hooks
    - Distributed coordination
    - Event emission
    """
    
    def __init__(
        self,
        persistence_dir: Optional[str] = None,
        enable_redis: bool = True,
    ):
        self.persistence_dir = Path(persistence_dir or "./workflow_states")
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self.enable_redis = enable_redis
        self._redis: Optional[aioredis.Redis] = None
        self._state_cache: Dict[str, WorkflowState] = {}
        self._transition_hooks: Dict[str, List[Callable]] = {}
        self._change_listeners: List[Callable] = []
        self.logger = logger
    
    async def _get_redis(self) -> Optional[aioredis.Redis]:
        """Get Redis connection"""
        if not self.enable_redis:
            return None
        
        if self._redis is None:
            try:
                self._redis = aioredis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    password=settings.REDIS_PASSWORD,
                    db=settings.REDIS_DB,
                    decode_responses=False,
                )
                await self._redis.ping()
            except Exception as e:
                self.logger.warning(f"Redis not available: {e}")
                self._redis = None
        
        return self._redis
    
    def _get_redis_key(self, workflow_id: str) -> str:
        """Get Redis key for workflow state"""
        return f"workflow:state:{workflow_id}"
    
    def _get_file_path(self, workflow_id: str) -> Path:
        """Get file path for workflow state"""
        return self.persistence_dir / f"{workflow_id}.json"
    
    # ========================================================================
    # State CRUD Operations
    # ========================================================================
    
    async def create_state(
        self,
        workflow_id: str,
        name: str = "",
        initial_data: Optional[Dict[str, Any]] = None,
        assigned_agents: Optional[List[str]] = None,
        total_steps: int = 0,
        deadline: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> WorkflowState:
        """Create new workflow state"""
        state = WorkflowState(
            workflow_id=workflow_id,
            name=name,
            state_data=initial_data or {},
            assigned_agents=assigned_agents or [],
            total_steps=total_steps,
            deadline=deadline,
            metadata=metadata or {},
            tags=tags or [],
        )
        
        await self._save_state(state)
        await self._emit_change_event(state, StateChangeType.CREATE)
        
        self.logger.info(f"Created workflow state: {workflow_id}")
        return state
    
    async def get_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Get workflow state by ID"""
        # Check cache first
        if workflow_id in self._state_cache:
            return self._state_cache[workflow_id]
        
        # Try Redis
        redis = await self._get_redis()
        if redis:
            try:
                data = await redis.get(self._get_redis_key(workflow_id))
                if data:
                    state_dict = pickle.loads(data)
                    state = WorkflowState(**state_dict)
                    self._state_cache[workflow_id] = state
                    return state
            except Exception as e:
                self.logger.warning(f"Redis read failed: {e}")
        
        # Fallback to file
        file_path = self._get_file_path(workflow_id)
        if file_path.exists():
            with open(file_path, 'r') as f:
                state_dict = json.load(f)
                # Convert datetime strings
                for key in ['created_at', 'updated_at', 'started_at', 'completed_at', 'deadline']:
                    if state_dict.get(key) and isinstance(state_dict[key], str):
                        state_dict[key] = datetime.fromisoformat(state_dict[key])
                state = WorkflowState(**state_dict)
                self._state_cache[workflow_id] = state
                return state
        
        return None
    
    async def _save_state(self, state: WorkflowState):
        """Save state to Redis and file"""
        state.updated_at = datetime.utcnow()
        
        # Update cache
        self._state_cache[state.workflow_id] = state
        
        state_dict = state.model_dump()
        
        # Save to Redis
        redis = await self._get_redis()
        if redis:
            try:
                pickled = pickle.dumps(state_dict)
                await redis.setex(
                    self._get_redis_key(state.workflow_id),
                    86400 * 7,
                    pickled
                )
            except Exception as e:
                self.logger.warning(f"Redis save failed: {e}")
        
        # Save to file as backup
        file_path = self._get_file_path(state.workflow_id)
        with open(file_path, 'w') as f:
            json.dump(state_dict, f, default=str, indent=2)
    
    async def update_state(
        self,
        workflow_id: str,
        state_data: Optional[Dict[str, Any]] = None,
        current_step: Optional[str] = None,
        step_result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        increment_completed: bool = False,
    ) -> Optional[WorkflowState]:
        """Update workflow state"""
        state = await self.get_state(workflow_id)
        if not state:
            return None
        
        changed_fields = []
        
        if state_data:
            state.state_data.update(state_data)
            changed_fields.append("state_data")
        
        if current_step:
            state.current_step = current_step
            changed_fields.append("current_step")
        
        if step_result:
            step_name = current_step or "unnamed"
            state.step_results[step_name] = step_result
            changed_fields.append("step_results")
        
        if error:
            state.error = error
            state.error_count += 1
            changed_fields.append("error")
        
        if metadata:
            state.metadata.update(metadata)
            changed_fields.append("metadata")
        
        if increment_completed:
            state.completed_steps += 1
            state.progress_percentage = (state.completed_steps / max(state.total_steps, 1)) * 100
            changed_fields.append("progress")
        
        await self._save_state(state)
        await self._emit_change_event(state, StateChangeType.UPDATE, changed_fields)
        
        return state
    
    async def delete_state(self, workflow_id: str) -> bool:
        """Delete workflow state"""
        # Remove from cache
        self._state_cache.pop(workflow_id, None)
        
        # Remove from Redis
        redis = await self._get_redis()
        if redis:
            await redis.delete(self._get_redis_key(workflow_id))
        
        # Remove file
        file_path = self._get_file_path(workflow_id)
        if file_path.exists():
            file_path.unlink()
        
        self.logger.info(f"Deleted workflow state: {workflow_id}")
        return True
    
    # ========================================================================
    # Phase Transitions
    # ========================================================================
    
    async def transition_phase(
        self,
        workflow_id: str,
        to_phase: WorkflowPhase,
        trigger: str = "manual",
        data: Optional[Dict[str, Any]] = None,
    ) -> Optional[WorkflowState]:
        """Transition workflow to a new phase"""
        state = await self.get_state(workflow_id)
        if not state:
            return None
        
        from_phase = state.phase
        
        # Validate transition
        if not self._is_valid_transition(from_phase, to_phase):
            self.logger.warning(f"Invalid transition: {from_phase} -> {to_phase}")
            return None
        
        # Record transition
        import uuid
        transition = StateTransition(
            transition_id=str(uuid.uuid4()),
            from_phase=from_phase,
            to_phase=to_phase,
            trigger=trigger,
            data=data or {},
        )
        state.transitions.append({
            "transition_id": transition.transition_id,
            "from_phase": transition.from_phase.value,
            "to_phase": transition.to_phase.value,
            "trigger": transition.trigger,
            "timestamp": transition.timestamp.isoformat(),
            "data": transition.data,
        })
        
        # Update phase
        state.phase = to_phase
        
        # Update timing based on phase
        if to_phase == WorkflowPhase.EXECUTING and not state.started_at:
            state.started_at = datetime.utcnow()
        elif to_phase in [WorkflowPhase.COMPLETED, WorkflowPhase.FAILED, WorkflowPhase.CANCELLED]:
            state.completed_at = datetime.utcnow()
        
        await self._save_state(state)
        
        # Execute transition hooks
        await self._execute_hooks(f"{from_phase.value}_to_{to_phase.value}", state)
        
        # Emit event
        await self._emit_change_event(
            state, 
            StateChangeType.TRANSITION,
            changed_fields=["phase"],
            old_phase=from_phase,
            new_phase=to_phase
        )
        
        self.logger.info(f"Workflow {workflow_id} transitioned: {from_phase.value} -> {to_phase.value}")
        return state
    
    def _is_valid_transition(self, from_phase: WorkflowPhase, to_phase: WorkflowPhase) -> bool:
        """Check if phase transition is valid"""
        # Define valid transitions
        valid_transitions = {
            WorkflowPhase.INITIALIZING: [WorkflowPhase.PLANNING, WorkflowPhase.EXECUTING, WorkflowPhase.CANCELLED],
            WorkflowPhase.PLANNING: [WorkflowPhase.EXECUTING, WorkflowPhase.PAUSED, WorkflowPhase.CANCELLED],
            WorkflowPhase.EXECUTING: [WorkflowPhase.VALIDATING, WorkflowPhase.PAUSED, WorkflowPhase.FAILED, WorkflowPhase.CANCELLED],
            WorkflowPhase.VALIDATING: [WorkflowPhase.COMPLETING, WorkflowPhase.EXECUTING, WorkflowPhase.FAILED],
            WorkflowPhase.COMPLETING: [WorkflowPhase.COMPLETED, WorkflowPhase.FAILED],
            WorkflowPhase.PAUSED: [WorkflowPhase.EXECUTING, WorkflowPhase.CANCELLED],
            WorkflowPhase.ROLLING_BACK: [WorkflowPhase.FAILED, WorkflowPhase.PAUSED],
            WorkflowPhase.COMPLETED: [],
            WorkflowPhase.FAILED: [WorkflowPhase.ROLLING_BACK],
            WorkflowPhase.CANCELLED: [],
        }
        
        return to_phase in valid_transitions.get(from_phase, [])
    
    def register_transition_hook(
        self,
        transition: str,
        hook: Callable[[WorkflowState], Any],
    ):
        """Register a hook for phase transitions"""
        if transition not in self._transition_hooks:
            self._transition_hooks[transition] = []
        self._transition_hooks[transition].append(hook)
    
    async def _execute_hooks(self, transition: str, state: WorkflowState):
        """Execute transition hooks"""
        hooks = self._transition_hooks.get(transition, [])
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(state)
                else:
                    hook(state)
            except Exception as e:
                self.logger.error(f"Hook execution error: {e}")
    
    # ========================================================================
    # Checkpoints and Rollback
    # ========================================================================
    
    async def create_checkpoint(
        self,
        workflow_id: str,
        step_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[StateCheckpoint]:
        """Create a checkpoint for rollback capability"""
        state = await self.get_state(workflow_id)
        if not state:
            return None
        
        checkpoint_id = hashlib.md5(
            f"{workflow_id}:{step_name}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()
        
        checkpoint = StateCheckpoint(
            checkpoint_id=checkpoint_id,
            workflow_id=workflow_id,
            phase=state.phase,
            step_name=step_name,
            state_data=state.state_data.copy(),
            metadata=metadata or {},
        )
        
        state.checkpoints.append(checkpoint.to_dict())
        await self._save_state(state)
        
        await self._emit_change_event(state, StateChangeType.CHECKPOINT)
        
        self.logger.info(f"Created checkpoint {checkpoint_id} for workflow {workflow_id}")
        return checkpoint
    
    async def rollback_to_checkpoint(
        self,
        workflow_id: str,
        checkpoint_id: str,
    ) -> Optional[WorkflowState]:
        """Rollback workflow to a checkpoint"""
        state = await self.get_state(workflow_id)
        if not state:
            return None
        
        # Find checkpoint
        checkpoint_dict = None
        checkpoint_index = -1
        for i, cp in enumerate(state.checkpoints):
            if cp["checkpoint_id"] == checkpoint_id:
                checkpoint_dict = cp
                checkpoint_index = i
                break
        
        if not checkpoint_dict:
            self.logger.error(f"Checkpoint {checkpoint_id} not found")
            return None
        
        checkpoint = StateCheckpoint.from_dict(checkpoint_dict)
        
        # Transition to rolling back
        old_phase = state.phase
        state.phase = WorkflowPhase.ROLLING_BACK
        
        # Restore state
        state.state_data = checkpoint.state_data.copy()
        state.current_step = checkpoint.step_name
        
        # Remove checkpoints after this one
        state.checkpoints = state.checkpoints[:checkpoint_index + 1]
        
        # After rollback, transition to paused
        state.phase = WorkflowPhase.PAUSED
        
        await self._save_state(state)
        await self._emit_change_event(
            state,
            StateChangeType.ROLLBACK,
            old_phase=old_phase,
            new_phase=state.phase
        )
        
        self.logger.info(f"Rolled back workflow {workflow_id} to checkpoint {checkpoint_id}")
        return state
    
    async def list_checkpoints(self, workflow_id: str) -> List[StateCheckpoint]:
        """List all checkpoints for a workflow"""
        state = await self.get_state(workflow_id)
        if not state:
            return []
        
        return [StateCheckpoint.from_dict(cp) for cp in state.checkpoints]
    
    # ========================================================================
    # Agent State Management
    # ========================================================================
    
    async def update_agent_state(
        self,
        workflow_id: str,
        agent_id: str,
        agent_state: Dict[str, Any],
    ) -> Optional[WorkflowState]:
        """Update state for a specific agent in the workflow"""
        state = await self.get_state(workflow_id)
        if not state:
            return None
        
        state.agent_states[agent_id] = {
            **state.agent_states.get(agent_id, {}),
            **agent_state,
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        await self._save_state(state)
        return state
    
    async def get_agent_state(
        self,
        workflow_id: str,
        agent_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get state for a specific agent"""
        state = await self.get_state(workflow_id)
        if not state:
            return None
        
        return state.agent_states.get(agent_id)
    
    # ========================================================================
    # Event System
    # ========================================================================
    
    def add_change_listener(self, listener: Callable[[StateChangeEvent], Any]):
        """Add a listener for state changes"""
        self._change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable):
        """Remove a change listener"""
        self._change_listeners = [l for l in self._change_listeners if l != listener]
    
    async def _emit_change_event(
        self,
        state: WorkflowState,
        change_type: StateChangeType,
        changed_fields: Optional[List[str]] = None,
        old_phase: Optional[WorkflowPhase] = None,
        new_phase: Optional[WorkflowPhase] = None,
    ):
        """Emit state change event to listeners"""
        import uuid
        event = StateChangeEvent(
            event_id=str(uuid.uuid4()),
            workflow_id=state.workflow_id,
            change_type=change_type,
            old_phase=old_phase,
            new_phase=new_phase,
            changed_fields=changed_fields or [],
        )
        
        for listener in self._change_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(event)
                else:
                    listener(event)
            except Exception as e:
                self.logger.error(f"Change listener error: {e}")
    
    # ========================================================================
    # Query Operations
    # ========================================================================
    
    async def list_workflows(
        self,
        phase: Optional[WorkflowPhase] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[WorkflowState]:
        """List workflows with optional filters"""
        workflows = []
        
        # Get from files
        for file_path in self.persistence_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    state_dict = json.load(f)
                    for key in ['created_at', 'updated_at', 'started_at', 'completed_at', 'deadline']:
                        if state_dict.get(key) and isinstance(state_dict[key], str):
                            state_dict[key] = datetime.fromisoformat(state_dict[key])
                    state = WorkflowState(**state_dict)
                    
                    # Apply filters
                    if phase and state.phase != phase:
                        continue
                    if tags and not any(t in state.tags for t in tags):
                        continue
                    
                    workflows.append(state)
            except Exception as e:
                self.logger.warning(f"Failed to load state from {file_path}: {e}")
        
        # Sort by updated_at descending
        workflows.sort(key=lambda w: w.updated_at, reverse=True)
        return workflows[:limit]
    
    async def get_workflow_summary(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of workflow state"""
        state = await self.get_state(workflow_id)
        if not state:
            return None
        
        duration = None
        if state.started_at and state.completed_at:
            duration = (state.completed_at - state.started_at).total_seconds()
        elif state.started_at:
            duration = (datetime.utcnow() - state.started_at).total_seconds()
        
        return {
            "workflow_id": state.workflow_id,
            "name": state.name,
            "phase": state.phase.value,
            "progress": state.progress_percentage,
            "completed_steps": state.completed_steps,
            "total_steps": state.total_steps,
            "error_count": state.error_count,
            "duration_seconds": duration,
            "agent_count": len(state.assigned_agents),
            "checkpoint_count": len(state.checkpoints),
            "tags": state.tags,
        }


# ============================================================================
# Singleton Instance
# ============================================================================

_state_manager: Optional[EnhancedStateManager] = None


async def get_enhanced_state_manager() -> EnhancedStateManager:
    """Get or create enhanced state manager singleton"""
    global _state_manager
    if _state_manager is None:
        _state_manager = EnhancedStateManager()
    return _state_manager

