"""
Task Execution Engine
Orchestrates tool execution, manages execution trees, and handles step-by-step decomposition
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.callbacks import CallbackManager
from ..logging_config import get_logger
from .tool_registry import ToolRegistry, get_tool_registry
from .state_manager import WorkflowStateManager, WorkflowState, StateStatus
from .guardrails import SafetyGuardrails, get_guardrails

logger = get_logger("cyrex.execution_engine")


class ExecutionStatus(str, Enum):
    """Execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionStep:
    """Single execution step"""
    step_id: str
    step_name: str
    tool_name: Optional[str] = None
    input_data: Dict[str, Any] = None
    output_data: Optional[Any] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class ExecutionTree:
    """Execution tree for complex workflows"""
    execution_id: str
    root_step: ExecutionStep
    child_steps: List['ExecutionTree'] = None
    parent: Optional['ExecutionTree'] = None
    
    def __post_init__(self):
        if self.child_steps is None:
            self.child_steps = []


class TaskExecutionEngine:
    """
    Orchestrates task execution with tool calls, state management, and safety checks
    Supports step-by-step decomposition and execution trees
    """
    
    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        state_manager: Optional[WorkflowStateManager] = None,
        guardrails: Optional[SafetyGuardrails] = None,
    ):
        self.tool_registry = tool_registry or get_tool_registry()
        self.state_manager = state_manager or WorkflowStateManager()
        self.guardrails = guardrails or get_guardrails()
        self.logger = logger
        self.execution_trees: Dict[str, ExecutionTree] = {}
    
    async def execute_workflow(
        self,
        workflow_id: str,
        steps: List[Dict[str, Any]],
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a multi-step workflow
        
        Args:
            workflow_id: Unique workflow identifier
            steps: List of step definitions
            initial_state: Initial workflow state
        
        Returns:
            Final workflow state and results
        """
        # Create or get workflow state
        state = self.state_manager.get_state(workflow_id)
        if not state:
            state = self.state_manager.create_state(workflow_id, initial_state)
        
        state.status = StateStatus.RUNNING
        self.state_manager.save_state(state)
        
        execution_tree = ExecutionTree(
            execution_id=workflow_id,
            root_step=ExecutionStep(
                step_id=f"{workflow_id}_root",
                step_name="root",
                status=ExecutionStatus.RUNNING,
            )
        )
        self.execution_trees[workflow_id] = execution_tree
        
        try:
            # Execute steps sequentially
            for i, step_def in enumerate(steps):
                step_name = step_def.get("name", f"step_{i}")
                tool_name = step_def.get("tool")
                step_input = step_def.get("input", {})
                
                # Update current step
                state.current_step = step_name
                self.state_manager.save_state(state)
                
                # Create checkpoint before step
                checkpoint = self.state_manager.create_checkpoint(
                    workflow_id,
                    step_name,
                    state_data=state.state_data.copy(),
                )
                
                # Execute step
                step_result = await self._execute_step(
                    workflow_id,
                    step_name,
                    tool_name,
                    step_input,
                    state.state_data,
                )
                
                # Update state with step result
                state.state_data[step_name] = step_result
                state.state_data.update(step_result.get("state_updates", {}))
                
                # Check for errors
                if step_result.get("error"):
                    state.status = StateStatus.FAILED
                    state.error = step_result["error"]
                    self.state_manager.save_state(state)
                    break
            
            if state.status != StateStatus.FAILED:
                state.status = StateStatus.COMPLETED
            
            self.state_manager.save_state(state)
            
            return {
                "workflow_id": workflow_id,
                "status": state.status.value,
                "final_state": state.state_data,
                "error": state.error,
            }
        
        except Exception as e:
            state.status = StateStatus.FAILED
            state.error = str(e)
            self.state_manager.save_state(state)
            self.logger.error(f"Workflow execution failed: {e}", exc_info=True)
            raise
    
    async def _execute_step(
        self,
        workflow_id: str,
        step_name: str,
        tool_name: Optional[str],
        step_input: Dict[str, Any],
        current_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a single step"""
        step_id = f"{workflow_id}:{step_name}"
        start_time = datetime.now()
        
        step = ExecutionStep(
            step_id=step_id,
            step_name=step_name,
            tool_name=tool_name,
            input_data=step_input,
            status=ExecutionStatus.RUNNING,
            started_at=start_time,
        )
        
        try:
            # Safety check on input
            if isinstance(step_input, dict) and "prompt" in step_input:
                safety_result = self.guardrails.check_prompt(step_input["prompt"])
                if self.guardrails.should_block(safety_result):
                    raise ValueError(f"Safety check failed: {safety_result.message}")
            
            # Execute tool if specified
            if tool_name:
                tool = self.tool_registry.get_tool(tool_name)
                if not tool:
                    raise ValueError(f"Tool not found: {tool_name}")
                
                # Merge current state into input
                tool_input = {**current_state, **step_input}
                
                # Execute tool
                if hasattr(tool, 'ainvoke'):
                    output = await tool.ainvoke(tool_input)
                else:
                    output = await asyncio.to_thread(tool.invoke, tool_input)
                
                # Safety check on output
                if isinstance(output, str):
                    output_safety = self.guardrails.check_output(output)
                    if self.guardrails.should_block(output_safety):
                        raise ValueError(f"Output safety check failed: {output_safety.message}")
                
                step.output_data = output
                step.status = ExecutionStatus.COMPLETED
            
            else:
                # No tool, just pass through
                step.output_data = step_input
                step.status = ExecutionStatus.COMPLETED
            
            end_time = datetime.now()
            step.completed_at = end_time
            step.duration_ms = (end_time - start_time).total_seconds() * 1000
            
            self.logger.info(f"Step {step_name} completed in {step.duration_ms:.2f}ms")
            
            return {
                "step_id": step_id,
                "output": step.output_data,
                "duration_ms": step.duration_ms,
            }
        
        except Exception as e:
            step.status = ExecutionStatus.FAILED
            step.error = str(e)
            step.completed_at = datetime.now()
            step.duration_ms = (step.completed_at - start_time).total_seconds() * 1000
            
            self.logger.error(f"Step {step_name} failed: {e}", exc_info=True)
            
            return {
                "step_id": step_id,
                "error": str(e),
                "duration_ms": step.duration_ms,
            }
    
    async def execute_chain(
        self,
        chain: Runnable,
        input_data: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
    ) -> Any:
        """
        Execute a LangChain chain/runnable
        
        Args:
            chain: LangChain Runnable
            input_data: Input data for chain
            config: Optional RunnableConfig
        
        Returns:
            Chain output
        """
        try:
            if hasattr(chain, 'ainvoke'):
                result = await chain.ainvoke(input_data, config=config)
            else:
                result = await asyncio.to_thread(chain.invoke, input_data, config)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Chain execution failed: {e}", exc_info=True)
            raise
    
    def decompose_task(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Decompose a high-level task into steps
        Uses chain-of-thought reasoning (hidden from user)
        
        Args:
            task_description: High-level task description
        
        Returns:
            List of step definitions
        """
        # Simple decomposition logic (can be enhanced with LLM)
        # For now, return a single step
        steps = [
            {
                "name": "execute_task",
                "description": task_description,
                "input": {"task": task_description},
            }
        ]
        
        return steps
    
    def get_execution_tree(self, execution_id: str) -> Optional[ExecutionTree]:
        """Get execution tree for an execution"""
        return self.execution_trees.get(execution_id)
    
    async def cancel_execution(self, workflow_id: str) -> bool:
        """Cancel a running execution"""
        state = self.state_manager.get_state(workflow_id)
        
        if not state:
            return False
        
        if state.status == StateStatus.RUNNING:
            state.status = StateStatus.CANCELLED
            self.state_manager.save_state(state)
            self.logger.info(f"Cancelled workflow: {workflow_id}")
            return True
        
        return False


def get_execution_engine() -> TaskExecutionEngine:
    """Get global execution engine instance"""
    return TaskExecutionEngine()

