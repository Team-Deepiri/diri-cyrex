"""
Clause Evolution Tracker
Tracks how contract clauses evolve across versions
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import difflib

from ..logging_config import get_logger
from ..integrations.llm_providers import get_llm_provider

logger = get_logger("cyrex.clause_evolution")


class ClauseEvolutionTracker:
    """
    Track clause changes across contract versions
    
    Identifies:
    - New clauses
    - Modified clauses
    - Deleted clauses
    - Language changes within clauses
    """
    
    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider or get_llm_provider()
    
    async def track_clause_changes(
        self,
        contract_id: str,
        old_version_clauses: List[Dict[str, Any]],
        new_version_clauses: List[Dict[str, Any]],
        old_version_number: int,
        new_version_number: int,
    ) -> Dict[str, Any]:
        """
        Compare clauses between two contract versions
        
        Returns:
            Dictionary with:
            - new_clauses: Clauses added in new version
            - modified_clauses: Clauses changed
            - deleted_clauses: Clauses removed
            - unchanged_clauses: Clauses that didn't change
            - summary: Text summary of changes
        """
        try:
            logger.info("Tracking clause changes",
                          contract_id=contract_id,
                          old_version=old_version_number,
                          new_version=new_version_number)
            
            # Build clause maps by identifier
            old_clause_map = self._build_clause_map(old_version_clauses)
            new_clause_map = self._build_clause_map(new_version_clauses)
            
            # Identify changes
            new_clauses = []
            modified_clauses = []
            deleted_clauses = []
            unchanged_clauses = []
            
            # Find new and modified clauses
            for clause_id, new_clause in new_clause_map.items():
                if clause_id not in old_clause_map:
                    new_clauses.append(new_clause)
                else:
                    old_clause = old_clause_map[clause_id]
                    changes = await self._compare_clauses(old_clause, new_clause)
                    if changes["has_changes"]:
                        modified_clauses.append({
                            "clause": new_clause,
                            "old_clause": old_clause,
                            "changes": changes,
                        })
                    else:
                        unchanged_clauses.append(new_clause)
            
            # Find deleted clauses
            for clause_id, old_clause in old_clause_map.items():
                if clause_id not in new_clause_map:
                    deleted_clauses.append(old_clause)
            
            # Generate summary using LLM
            summary = await self._generate_change_summary(
                new_clauses,
                modified_clauses,
                deleted_clauses,
                old_version_number,
                new_version_number,
            )
            
            result = {
                "contract_id": contract_id,
                "old_version": old_version_number,
                "new_version": new_version_number,
                "new_clauses": new_clauses,
                "modified_clauses": modified_clauses,
                "deleted_clauses": deleted_clauses,
                "unchanged_clauses": unchanged_clauses,
                "summary": summary,
                "statistics": {
                    "total_old_clauses": len(old_version_clauses),
                    "total_new_clauses": len(new_version_clauses),
                    "new_count": len(new_clauses),
                    "modified_count": len(modified_clauses),
                    "deleted_count": len(deleted_clauses),
                    "unchanged_count": len(unchanged_clauses),
                },
            }
            
            logger.info("Clause changes tracked",
                       contract_id=contract_id,
                       new_count=len(new_clauses),
                       modified_count=len(modified_clauses),
                       deleted_count=len(deleted_clauses))
            
            return result
            
        except Exception as e:
            logger.error("Error tracking clause changes",
                        contract_id=contract_id,
                        error=str(e))
            raise
    
    def _build_clause_map(self, clauses: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Build map of clauses by identifier"""
        clause_map = {}
        for clause in clauses:
            # Use clause number, type, or first 50 chars as identifier
            clause_id = (
                clause.get("clauseNumber") or
                clause.get("clauseTitle") or
                f"{clause.get('clauseType', 'UNKNOWN')}_{hash(clause.get('clauseText', '')[:50])}"
            )
            clause_map[str(clause_id)] = clause
        return clause_map
    
    async def _compare_clauses(
        self,
        old_clause: Dict[str, Any],
        new_clause: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare two clauses and identify changes"""
        changes = {
            "has_changes": False,
            "text_changed": False,
            "type_changed": False,
            "title_changed": False,
            "text_diff": None,
            "change_type": "UNCHANGED",
        }
        
        # Compare clause text
        old_text = old_clause.get("clauseText", "") or old_clause.get("clause_text", "")
        new_text = new_clause.get("clauseText", "") or new_clause.get("clause_text", "")
        
        if old_text != new_text:
            changes["text_changed"] = True
            changes["has_changes"] = True
            
            # Generate diff
            diff = list(difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                lineterm='',
            ))
            changes["text_diff"] = ''.join(diff)
        
        # Compare clause type
        old_type = old_clause.get("clauseType") or old_clause.get("clause_type")
        new_type = new_clause.get("clauseType") or new_clause.get("clause_type")
        if old_type != new_type:
            changes["type_changed"] = True
            changes["has_changes"] = True
        
        # Compare title
        old_title = old_clause.get("clauseTitle") or old_clause.get("clause_title")
        new_title = new_clause.get("clauseTitle") or new_clause.get("clause_title")
        if old_title != new_title:
            changes["title_changed"] = True
            changes["has_changes"] = True
        
        # Determine change type
        if changes["has_changes"]:
            if changes["text_changed"] and len(new_text) > len(old_text) * 1.5:
                changes["change_type"] = "SIGNIFICANT_MODIFICATION"
            elif changes["text_changed"]:
                changes["change_type"] = "MODIFICATION"
            elif changes["type_changed"]:
                changes["change_type"] = "TYPE_CHANGE"
            else:
                changes["change_type"] = "MINOR_MODIFICATION"
        
        return changes
    
    async def _generate_change_summary(
        self,
        new_clauses: List[Dict],
        modified_clauses: List[Dict],
        deleted_clauses: List[Dict],
        old_version: int,
        new_version: int,
    ) -> str:
        """Generate human-readable summary of changes"""
        llm = self.llm_provider.get_llm()
        
        prompt = f"""Summarize the changes between contract version {old_version} and {new_version}:

New Clauses ({len(new_clauses)}):
{json.dumps([c.get('clauseTitle') or c.get('clauseType') or c.get('clause_title') or c.get('clause_type') for c in new_clauses[:10]], indent=2)}

Modified Clauses ({len(modified_clauses)}):
{json.dumps([c['clause'].get('clauseTitle') or c['clause'].get('clauseType') or c['clause'].get('clause_title') or c['clause'].get('clause_type') for c in modified_clauses[:10]], indent=2)}

Deleted Clauses ({len(deleted_clauses)}):
{json.dumps([c.get('clauseTitle') or c.get('clauseType') or c.get('clause_title') or c.get('clause_type') for c in deleted_clauses[:10]], indent=2)}

Provide a concise summary (2-3 sentences) of the key changes."""
        
        response = await llm.ainvoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

