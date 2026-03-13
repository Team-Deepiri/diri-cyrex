"""
Obligation Dependency Graph Builder
Builds graph of obligation dependencies across contracts/leases
"""
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import json
import re

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

from ..logging_config import get_logger
from ..integrations.llm_providers import get_llm_provider

logger = get_logger("cyrex.obligation_dependency")


class ObligationDependencyGraph:
    """
    Build and analyze obligation dependency graphs
    
    Identifies:
    - Which obligations trigger others
    - Cascade effects
    - Dependency chains
    - Critical path obligations
    """
    
    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider or get_llm_provider()
        if HAS_NETWORKX:
            self.graph = nx.DiGraph()  # Directed graph for dependencies
        else:
            self.graph = None
            logger.warning("NetworkX not available, graph analysis will be limited")
    
    async def build_graph(
        self,
        obligations: List[Dict[str, Any]],
        contracts: Optional[List[str]] = None,
        leases: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Build dependency graph from obligations
        
        Args:
            obligations: List of obligation dictionaries
            contracts: Optional list of contract IDs for context
            leases: Optional list of lease IDs for context
        
        Returns:
            Graph structure with nodes and edges
        """
        try:
            logger.info("Building obligation dependency graph",
                       obligation_count=len(obligations))
            
            if not HAS_NETWORKX:
                # Fallback: return simple structure without graph analysis
                dependencies = await self._analyze_dependencies(obligations)
                return {
                    "nodes": [{"id": obl.get("id"), "data": obl} for obl in obligations],
                    "edges": dependencies,
                    "analysis": {},
                }
            
            # Clear existing graph
            self.graph.clear()
            
            # Add obligations as nodes
            for obl in obligations:
                obligation_id = obl.get("id") or obl.get("obligation_id")
                if not obligation_id:
                    continue
                
                self.graph.add_node(
                    obligation_id,
                    **{
                        "description": obl.get("description", ""),
                        "type": obl.get("obligationType") or obl.get("obligation_type") or obl.get("type", ""),
                        "deadline": obl.get("deadline"),
                        "contract_id": obl.get("contractId") or obl.get("contract_id"),
                        "lease_id": obl.get("leaseId") or obl.get("lease_id"),
                        "party": obl.get("party", ""),
                    }
                )
            
            # Analyze dependencies using LLM
            dependencies = await self._analyze_dependencies(obligations)
            
            # Add dependency edges
            for dep in dependencies:
                source_id = dep.get("source_obligation_id")
                target_id = dep.get("target_obligation_id")
                dependency_type = dep.get("dependency_type", "TRIGGERS")
                
                if source_id and target_id and source_id in self.graph and target_id in self.graph:
                    self.graph.add_edge(
                        source_id,
                        target_id,
                        dependency_type=dependency_type,
                        description=dep.get("description"),
                        confidence=dep.get("confidence", 0.5),
                    )
            
            # Analyze graph structure
            analysis = self._analyze_graph_structure()
            
            logger.info("Dependency graph built",
                       nodes=len(self.graph.nodes()),
                       edges=len(self.graph.edges()))
            
            return {
                "nodes": [{"id": node, "data": self.graph.nodes[node]} for node in self.graph.nodes()],
                "edges": [
                    {
                        "source": source,
                        "target": target,
                        "dependency_type": data.get("dependency_type"),
                        "description": data.get("description"),
                        "confidence": data.get("confidence"),
                    }
                    for source, target, data in self.graph.edges(data=True)
                ],
                "analysis": analysis,
            }
            
        except Exception as e:
            logger.error("Error building dependency graph", error=str(e))
            raise
    
    async def _analyze_dependencies(
        self,
        obligations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to analyze obligation dependencies
        """
        llm = self.llm_provider.get_llm()
        
        # Prepare obligation descriptions for LLM
        obligation_descriptions = []
        for i, obl in enumerate(obligations):
            obligation_descriptions.append(
                f"Obligation {i+1} (ID: {obl.get('id', 'unknown')}):\n"
                f"Description: {obl.get('description', '')}\n"
                f"Type: {obl.get('obligationType') or obl.get('obligation_type') or obl.get('type', '')}\n"
                f"Deadline: {obl.get('deadline', 'N/A')}\n"
                f"Party: {obl.get('party', '')}\n"
                f"Contract: {obl.get('contractId') or obl.get('contract_id', 'N/A')}\n"
                f"Lease: {obl.get('leaseId') or obl.get('lease_id', 'N/A')}\n"
            )
        
        prompt = f"""Analyze these obligations and identify dependencies between them.

OBLIGATIONS:
{chr(10).join(obligation_descriptions)}

For each obligation, identify:
1. Which other obligations it TRIGGERS (completion of this obligation triggers another)
2. Which other obligations it BLOCKS (this obligation must complete before another can start)
3. Which other obligations it MODIFIES (this obligation changes terms of another)
4. Which other obligations it REQUIRES (this obligation requires another to be completed first)
5. Which other obligations it PRECEDES (this must occur before another)
6. Which other obligations it CONFLICTS with (mutually exclusive)

Return JSON array:
[
  {{
    "source_obligation_id": "obligation_id",
    "target_obligation_id": "obligation_id",
    "dependency_type": "TRIGGERS|BLOCKS|MODIFIES|REQUIRES|PRECEDES|CONFLICTS",
    "description": "explanation of dependency",
    "confidence": 0.0-1.0,
    "trigger_condition": "condition that creates dependency"
  }}
]

Only include dependencies you're confident about (confidence > 0.6)."""
        
        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            dependencies = json.loads(json_match.group(0))
        else:
            dependencies = []
        
        return dependencies
    
    def _analyze_graph_structure(self) -> Dict[str, Any]:
        """
        Analyze graph structure for insights
        """
        if not HAS_NETWORKX or not self.graph or len(self.graph.nodes()) == 0:
            return {}
        
        analysis = {
            "node_count": len(self.graph.nodes()),
            "edge_count": len(self.graph.edges()),
            "is_connected": nx.is_weakly_connected(self.graph),
            "has_cycles": len(list(nx.simple_cycles(self.graph))) > 0,
            "critical_paths": [],
            "high_degree_nodes": [],
        }
        
        # Find critical path obligations (high in-degree or out-degree)
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        
        # Obligations that many others depend on (high in-degree)
        high_in_degree = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        analysis["high_in_degree_nodes"] = [
            {"obligation_id": node, "dependents": degree}
            for node, degree in high_in_degree
        ]
        
        # Obligations that trigger many others (high out-degree)
        high_out_degree = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        analysis["high_out_degree_nodes"] = [
            {"obligation_id": node, "triggers": degree}
            for node, degree in high_out_degree
        ]
        
        # Find cycles (circular dependencies)
        cycles = list(nx.simple_cycles(self.graph))
        analysis["cycles"] = cycles[:5]  # Limit to first 5 cycles
        
        return analysis
    
    async def find_cascading_obligations(
        self,
        obligation_id: str,
        max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find all obligations that depend on this one (cascade effect)
        
        Returns list of obligations in dependency chain
        """
        if not HAS_NETWORKX or not self.graph or obligation_id not in self.graph:
            return []
        
        # Use BFS to find all reachable nodes
        cascading = []
        visited = set()
        queue = [(obligation_id, 0)]  # (node, depth)
        
        while queue:
            current, depth = queue.pop(0)
            
            if depth > max_depth or current in visited:
                continue
            
            visited.add(current)
            
            # Get all neighbors (obligations this one affects)
            for neighbor in self.graph.successors(current):
                if neighbor not in visited:
                    edge_data = self.graph.get_edge_data(current, neighbor)
                    cascading.append({
                        "obligation_id": neighbor,
                        "depth": depth + 1,
                        "dependency_type": edge_data.get("dependency_type") if edge_data else "UNKNOWN",
                        "description": edge_data.get("description") if edge_data else "",
                    })
                    queue.append((neighbor, depth + 1))
        
        return cascading

