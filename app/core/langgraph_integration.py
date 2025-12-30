"""
LangGraph Integration
State machine integration for agent workflows using LangGraph
"""
from typing import Dict, List, Optional, Any, Callable, TypedDict
from datetime import datetime
import asyncio
from ..core.types import LangGraphState
from ..database.postgres import get_postgres_manager
from ..logging_config import get_logger
import json

logger = get_logger("cyrex.langgraph")

# Try to import LangGraph
HAS_LANGGRAPH = False
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.graph.message import add_messages
    HAS_LANGGRAPH = True
except ImportError:
    logger.warning("LangGraph not available, using fallback implementation")
    StateGraph = None
    END = None
    START = None


class GraphState(TypedDict):
    """LangGraph state structure"""
    messages: List[Dict[str, Any]]
    workflow_id: str
    current_node: str
    data: Dict[str, Any]
    history: List[Dict[str, Any]]


class LangGraphWorkflow:
    """
    LangGraph workflow manager for agent state machines
    Handles workflow definition, execution, and state persistence
    """
    
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.graph: Optional[Any] = None
        self.nodes: Dict[str, Callable] = {}
        self.edges: List[tuple] = []
        self.logger = logger
    
    def add_node(self, name: str, handler: Callable):
        """Add a node to the workflow"""
        self.nodes[name] = handler
        self.logger.debug(f"Node added: {name}", workflow_id=self.workflow_id)
    
    def add_edge(self, from_node: str, to_node: str):
        """Add an edge between nodes"""
        self.edges.append((from_node, to_node))
        self.logger.debug(f"Edge added: {from_node} -> {to_node}", workflow_id=self.workflow_id)
    
    def build_graph(self):
        """Build the LangGraph state graph"""
        if not HAS_LANGGRAPH:
            self.logger.warning("LangGraph not available, using fallback")
            return
        
        try:
            workflow = StateGraph(GraphState)
            
            # Add all nodes
            for name, handler in self.nodes.items():
                workflow.add_node(name, handler)
            
            # Add edges
            for from_node, to_node in self.edges:
                if from_node == "START":
                    workflow.add_edge(START, to_node)
                elif to_node == "END":
                    workflow.add_edge(from_node, END)
                else:
                    workflow.add_edge(from_node, to_node)
            
            # Compile graph
            self.graph = workflow.compile()
            self.logger.info(f"LangGraph workflow built: {self.workflow_id}")
        except Exception as e:
            self.logger.error(f"Failed to build LangGraph: {e}")
    
    async def execute(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the workflow"""
        if not self.graph:
            self.logger.warning("Graph not built, building now...")
            self.build_graph()
        
        if not self.graph:
            # Fallback execution without LangGraph
            return await self._fallback_execute(initial_state or {})
        
        try:
            # Prepare initial state
            state: GraphState = {
                "messages": [],
                "workflow_id": self.workflow_id,
                "current_node": "START",
                "data": initial_state or {},
                "history": [],
            }
            
            # Execute graph
            result = await self.graph.ainvoke(state)
            
            # Persist state
            await self._persist_state(result)
            
            self.logger.info(f"Workflow executed: {self.workflow_id}")
            return result
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            raise
    
    async def _fallback_execute(self, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback execution without LangGraph"""
        state = {
            "workflow_id": self.workflow_id,
            "current_node": "START",
            "data": initial_data,
            "history": [],
        }
        
        # Simple sequential execution
        current = "START"
        visited = set()
        
        while current and current != "END" and current not in visited:
            visited.add(current)
            
            if current in self.nodes:
                handler = self.nodes[current]
                try:
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(state)
                    else:
                        result = handler(state)
                    
                    state["history"].append({
                        "node": current,
                        "result": result,
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                    
                    # Find next node
                    next_nodes = [to_node for from_node, to_node in self.edges if from_node == current]
                    if next_nodes:
                        current = next_nodes[0]
                    else:
                        current = "END"
                except Exception as e:
                    self.logger.error(f"Node execution failed: {current}", error=str(e))
                    current = "END"
            else:
                break
        
        await self._persist_state(state)
        return state
    
    async def _persist_state(self, state: Dict[str, Any]):
        """Persist workflow state to database"""
        try:
            postgres = await get_postgres_manager()
            await postgres.execute("""
                INSERT INTO langgraph_states (state_id, workflow_id, current_node, next_nodes, data, history, status, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (state_id) DO UPDATE SET
                    current_node = EXCLUDED.current_node,
                    next_nodes = EXCLUDED.next_nodes,
                    data = EXCLUDED.data,
                    history = EXCLUDED.history,
                    status = EXCLUDED.status,
                    updated_at = EXCLUDED.updated_at
            """, state.get("state_id", f"{self.workflow_id}_{datetime.utcnow().isoformat()}"),
                self.workflow_id, state.get("current_node", ""),
                json.dumps(state.get("next_nodes", [])), json.dumps(state.get("data", {})),
                json.dumps(state.get("history", [])), "completed", datetime.utcnow())
        except Exception as e:
            self.logger.warning(f"Failed to persist state: {e}")


class LangGraphManager:
    """Manager for LangGraph workflows"""
    
    def __init__(self):
        self._workflows: Dict[str, LangGraphWorkflow] = {}
        self.logger = logger
    
    async def initialize(self):
        """Initialize LangGraph manager and create database tables"""
        # Create states table
        postgres = await get_postgres_manager()
        await postgres.execute("""
            CREATE TABLE IF NOT EXISTS langgraph_states (
                state_id VARCHAR(255) PRIMARY KEY,
                workflow_id VARCHAR(255) NOT NULL,
                current_node VARCHAR(255),
                next_nodes JSONB,
                data JSONB,
                history JSONB,
                status VARCHAR(50) NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_langgraph_workflow_id ON langgraph_states(workflow_id);
            CREATE INDEX IF NOT EXISTS idx_langgraph_status ON langgraph_states(status);
        """)
        
        self.logger.info("LangGraph manager initialized")
    
    def create_workflow(self, workflow_id: str) -> LangGraphWorkflow:
        """Create a new workflow"""
        workflow = LangGraphWorkflow(workflow_id)
        self._workflows[workflow_id] = workflow
        return workflow
    
    def get_workflow(self, workflow_id: str) -> Optional[LangGraphWorkflow]:
        """Get a workflow by ID"""
        return self._workflows.get(workflow_id)


# Global LangGraph manager
_langgraph_manager: Optional[LangGraphManager] = None


async def get_langgraph_manager() -> LangGraphManager:
    """Get or create LangGraph manager singleton"""
    global _langgraph_manager
    if _langgraph_manager is None:
        _langgraph_manager = LangGraphManager()
        await _langgraph_manager.initialize()
    return _langgraph_manager

