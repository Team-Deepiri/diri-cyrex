"""
Workflow State Manager
Persistent state tracking for multi-step workflows
Supports checkpoints, rollback, and distributed execution
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import json
import pickle
import hashlib
from pathlib import Path
import redis
from pydantic import BaseModel, Field
from ..logging_config import get_logger
from ..settings import settings

logger = get_logger("cyrex.state_manager")


class StateStatus(str, Enum):
    """Workflow state status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CHECKPOINTED = "checkpointed"


class WorkflowCheckpoint(BaseModel):
    """Checkpoint for workflow state"""
    checkpoint_id: str
    workflow_id: str
    step_name: str
    state_data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowState(BaseModel):
    """Complete workflow state"""
    workflow_id: str
    status: StateStatus = StateStatus.PENDING
    current_step: Optional[str] = None
    state_data: Dict[str, Any] = Field(default_factory=dict)
    checkpoints: List[WorkflowCheckpoint] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class WorkflowStateManager:
    """
    Manages persistent workflow state
    Supports Redis for distributed systems, file system for local
    """
    
    def __init__(
        self,
        use_redis: bool = True,
        redis_client: Optional[redis.Redis] = None,
        persistence_dir: Optional[str] = None,
    ):
        self.use_redis = use_redis
        self.persistence_dir = Path(persistence_dir or "./workflow_states")
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Redis if available
        if use_redis:
            try:
                self.redis_client = redis_client or redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    password=settings.REDIS_PASSWORD,
                    db=settings.REDIS_DB,
                    decode_responses=False,  # Store pickled data
                )
                self.redis_client.ping()
                logger.info("Connected to Redis for state management")
            except Exception as e:
                logger.warning(f"Redis not available, using file system: {e}")
                self.use_redis = False
                self.redis_client = None
        else:
            self.redis_client = None
    
    def _get_redis_key(self, workflow_id: str) -> str:
        """Get Redis key for workflow state"""
        return f"workflow:state:{workflow_id}"
    
    def _get_file_path(self, workflow_id: str) -> Path:
        """Get file path for workflow state"""
        return self.persistence_dir / f"{workflow_id}.json"
    
    def create_state(self, workflow_id: str, initial_data: Optional[Dict[str, Any]] = None) -> WorkflowState:
        """Create new workflow state"""
        state = WorkflowState(
            workflow_id=workflow_id,
            state_data=initial_data or {},
        )
        self.save_state(state)
        logger.info(f"Created workflow state: {workflow_id}")
        return state
    
    def get_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Get workflow state by ID"""
        try:
            if self.use_redis and self.redis_client:
                # Try Redis first
                key = self._get_redis_key(workflow_id)
                data = self.redis_client.get(key)
                if data:
                    state_dict = pickle.loads(data)
                    return WorkflowState(**state_dict)
            
            # Fallback to file system
            file_path = self._get_file_path(workflow_id)
            if file_path.exists():
                with open(file_path, 'r') as f:
                    state_dict = json.load(f)
                    # Convert datetime strings back
                    state_dict['created_at'] = datetime.fromisoformat(state_dict['created_at'])
                    state_dict['updated_at'] = datetime.fromisoformat(state_dict['updated_at'])
                    if state_dict.get('checkpoints'):
                        for cp in state_dict['checkpoints']:
                            cp['timestamp'] = datetime.fromisoformat(cp['timestamp'])
                    return WorkflowState(**state_dict)
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to get workflow state: {e}", exc_info=True)
            return None
    
    def save_state(self, state: WorkflowState):
        """Save workflow state"""
        try:
            state.updated_at = datetime.now()
            state_dict = state.model_dump()
            
            if self.use_redis and self.redis_client:
                # Save to Redis
                key = self._get_redis_key(state.workflow_id)
                pickled = pickle.dumps(state_dict)
                self.redis_client.setex(key, 86400 * 7, pickled)  # 7 days TTL
            
            # Always save to file system as backup
            file_path = self._get_file_path(state.workflow_id)
            with open(file_path, 'w') as f:
                json.dump(state_dict, f, default=str, indent=2)
            
            logger.debug(f"Saved workflow state: {state.workflow_id}")
        
        except Exception as e:
            logger.error(f"Failed to save workflow state: {e}", exc_info=True)
            raise
    
    def update_state(
        self,
        workflow_id: str,
        state_data: Optional[Dict[str, Any]] = None,
        status: Optional[StateStatus] = None,
        current_step: Optional[str] = None,
        error: Optional[str] = None,
        **kwargs
    ) -> WorkflowState:
        """Update workflow state"""
        state = self.get_state(workflow_id)
        if not state:
            state = self.create_state(workflow_id)
        
        if state_data:
            state.state_data.update(state_data)
        
        if status:
            state.status = status
        
        if current_step:
            state.current_step = current_step
        
        if error:
            state.error = error
            state.status = StateStatus.FAILED
        
        if kwargs:
            state.metadata.update(kwargs)
        
        self.save_state(state)
        return state
    
    def create_checkpoint(
        self,
        workflow_id: str,
        step_name: str,
        state_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowCheckpoint:
        """Create checkpoint for workflow"""
        state = self.get_state(workflow_id)
        if not state:
            state = self.create_state(workflow_id)
        
        checkpoint_id = hashlib.md5(
            f"{workflow_id}:{step_name}:{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        checkpoint = WorkflowCheckpoint(
            checkpoint_id=checkpoint_id,
            workflow_id=workflow_id,
            step_name=step_name,
            state_data=state_data or state.state_data.copy(),
            metadata=metadata or {},
        )
        
        state.checkpoints.append(checkpoint)
        state.status = StateStatus.CHECKPOINTED
        self.save_state(state)
        
        logger.info(f"Created checkpoint {checkpoint_id} for workflow {workflow_id} at step {step_name}")
        return checkpoint
    
    def rollback_to_checkpoint(self, workflow_id: str, checkpoint_id: str) -> WorkflowState:
        """Rollback workflow to checkpoint"""
        state = self.get_state(workflow_id)
        if not state:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Find checkpoint
        checkpoint = None
        checkpoint_index = -1
        for i, cp in enumerate(state.checkpoints):
            if cp.checkpoint_id == checkpoint_id:
                checkpoint = cp
                checkpoint_index = i
                break
        
        if not checkpoint:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        # Rollback state
        state.state_data = checkpoint.state_data.copy()
        state.current_step = checkpoint.step_name
        state.status = StateStatus.CHECKPOINTED
        
        # Remove checkpoints after this one
        state.checkpoints = state.checkpoints[:checkpoint_index + 1]
        
        self.save_state(state)
        logger.info(f"Rolled back workflow {workflow_id} to checkpoint {checkpoint_id}")
        return state
    
    def delete_state(self, workflow_id: str):
        """Delete workflow state"""
        try:
            if self.use_redis and self.redis_client:
                key = self._get_redis_key(workflow_id)
                self.redis_client.delete(key)
            
            file_path = self._get_file_path(workflow_id)
            if file_path.exists():
                file_path.unlink()
            
            logger.info(f"Deleted workflow state: {workflow_id}")
        
        except Exception as e:
            logger.error(f"Failed to delete workflow state: {e}", exc_info=True)
    
    def list_workflows(
        self,
        status: Optional[StateStatus] = None,
        limit: int = 100,
    ) -> List[WorkflowState]:
        """List workflows with optional status filter"""
        workflows = []
        
        try:
            if self.use_redis and self.redis_client:
                # List from Redis
                pattern = self._get_redis_key("*")
                keys = self.redis_client.keys(pattern)
                for key in keys[:limit]:
                    data = self.redis_client.get(key)
                    if data:
                        state_dict = pickle.loads(data)
                        state = WorkflowState(**state_dict)
                        if not status or state.status == status:
                            workflows.append(state)
            else:
                # List from file system
                for file_path in self.persistence_dir.glob("*.json"):
                    try:
                        with open(file_path, 'r') as f:
                            state_dict = json.load(f)
                            state_dict['created_at'] = datetime.fromisoformat(state_dict['created_at'])
                            state_dict['updated_at'] = datetime.fromisoformat(state_dict['updated_at'])
                            state = WorkflowState(**state_dict)
                            if not status or state.status == status:
                                workflows.append(state)
                    except Exception as e:
                        logger.warning(f"Failed to load state from {file_path}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to list workflows: {e}", exc_info=True)
        
        return workflows[:limit]

