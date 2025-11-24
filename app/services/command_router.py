"""
Command Router
BERT/DeBERTa-based command routing to predefined abilities
Tier 1: Maximum Reliability & Control
"""
import torch
import torch.nn as nn
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from ..logging_config import get_logger
from ..settings import settings

logger = get_logger("cyrex.command_router")


class CommandRouter:
    """
    Command router for predefined ability system.
    Uses fine-tuned BERT/DeBERTa to route user commands to predefined abilities.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_abilities: int = 50,
        model_path: Optional[str] = None
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_abilities = num_abilities
        
        # Load tokenizer and model
        if model_path and Path(model_path).exists():
            logger.info(f"Loading fine-tuned model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=num_abilities
            ).to(self.device)
        else:
            logger.info(f"Loading base model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_abilities
            ).to(self.device)
        
        self.model.eval()
        
        # Load ability mappings
        self.ability_registry = self._load_ability_registry()
        self.role_ability_matrix = self._load_role_ability_matrix()
        
        logger.info(f"Command router initialized with {len(self.ability_registry)} abilities")
    
    def _load_ability_registry(self) -> Dict[int, Dict]:
        """Load ability ID to ability definition mapping"""
        # In production, load from database or config
        return {
            0: {"id": "summarize_text", "name": "Summarize Text", "category": "productivity"},
            1: {"id": "create_objective", "name": "Create Objective", "category": "gamification"},
            2: {"id": "activate_focus_boost", "name": "Activate Focus Boost", "category": "boost"},
            3: {"id": "activate_velocity_boost", "name": "Activate Velocity Boost", "category": "boost"},
            4: {"id": "generate_code_review", "name": "Generate Code Review", "category": "development"},
            5: {"id": "refactor_suggest", "name": "Refactor Suggestion", "category": "development"},
            6: {"id": "create_odyssey", "name": "Create Odyssey", "category": "gamification"},
            7: {"id": "schedule_break", "name": "Schedule Break", "category": "wellness"},
            # ... add all 50 abilities
        }
    
    def _load_role_ability_matrix(self) -> Dict[str, List[str]]:
        """Load role-specific ability mappings"""
        return {
            "software_engineer": [
                "generate_code_review", "refactor_suggest", "debug_assist",
                "documentation_gen", "test_generation", "commit_message_gen",
                "create_objective", "activate_focus_boost"
            ],
            "designer": [
                "design_critique", "color_palette_gen", "layout_suggest",
                "export_assets", "design_system_check", "create_objective"
            ],
            "product_manager": [
                "feature_breakdown", "user_story_gen", "sprint_planning",
                "roadmap_suggest", "stakeholder_update", "create_odyssey"
            ],
            "marketer": [
                "copy_gen", "campaign_suggest", "audience_analysis",
                "content_calendar", "seo_optimize", "create_objective"
            ],
            "general": list(self.ability_registry.keys())
        }
    
    def route_command(
        self,
        user_command: str,
        user_role: Optional[str] = None,
        context: Optional[Dict] = None,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Route user command to predefined abilities
        
        Args:
            user_command: User's natural language command
            user_role: User's role (filters relevant abilities)
            context: Additional context (current task, project, etc.)
            top_k: Number of top predictions to return
        
        Returns:
            List of ability predictions with confidence scores
        """
        # Tokenize input
        inputs = self.tokenizer(
            user_command,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, k=min(top_k, self.num_abilities), dim=-1)
        
        # Convert to ability definitions
        predictions = []
        for idx, prob in zip(top_indices[0], top_probs[0]):
            ability_id = idx.item()
            ability_def = self.ability_registry.get(ability_id)
            
            if not ability_def:
                continue
            
            # Filter by role if provided
            if user_role:
                role_abilities = self.role_ability_matrix.get(user_role, [])
                if ability_def["id"] not in role_abilities:
                    continue
            
            confidence = prob.item()
            
            # Extract parameters from command
            parameters = self._extract_parameters(
                ability_def["id"],
                user_command,
                context or {}
            )
            
            predictions.append({
                "ability_id": ability_def["id"],
                "ability_name": ability_def["name"],
                "category": ability_def["category"],
                "confidence": confidence,
                "parameters": parameters
            })
        
        return predictions
    
    def route_with_confidence_threshold(
        self,
        user_command: str,
        user_role: Optional[str] = None,
        min_confidence: float = 0.7,
        context: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Route with confidence threshold - returns single best match if above threshold
        """
        predictions = self.route_command(user_command, user_role, context, top_k=1)
        
        if not predictions:
            return None
        
        best = predictions[0]
        if best["confidence"] >= min_confidence:
            return best
        
        return None
    
    def _extract_parameters(
        self,
        ability_id: str,
        command: str,
        context: Dict
    ) -> Dict:
        """Extract parameters for specific ability from command and context"""
        params = {}
        
        # Extract common parameters
        if "duration" in command.lower() or "minute" in command.lower() or "hour" in command.lower():
            params["duration"] = self._extract_duration(command)
        
        if "title" in context:
            params["title"] = context["title"]
        elif ability_id == "create_objective":
            params["title"] = self._extract_title(command)
        
        # Ability-specific parameter extraction
        if ability_id == "create_objective":
            params["momentum_reward"] = self._extract_momentum_reward(command, default=10)
            params["deadline"] = context.get("deadline")
        
        elif ability_id in ["activate_focus_boost", "activate_velocity_boost"]:
            params["duration"] = params.get("duration", 60)
            params["source"] = "user_request"
        
        elif ability_id == "generate_code_review":
            params["file_path"] = context.get("file_path")
            params["focus_areas"] = self._extract_focus_areas(command)
        
        # Merge with context
        params.update({k: v for k, v in context.items() if k not in params})
        
        return params
    
    def _extract_title(self, command: str) -> str:
        """Extract title from command using simple heuristics"""
        # Remove common command words
        words = command.split()
        remove_words = ["create", "make", "add", "new", "generate", "build"]
        filtered = [w for w in words if w.lower() not in remove_words]
        return " ".join(filtered[:10])  # Limit to 10 words
    
    def _extract_duration(self, command: str) -> int:
        """Extract duration in minutes from command"""
        import re
        match = re.search(r'(\d+)\s*(minute|min|hour|hr|h)', command.lower())
        if match:
            value = int(match.group(1))
            unit = match.group(2)
            if 'hour' in unit or 'hr' in unit or unit == 'h':
                return value * 60
            return value
        return 60  # Default
    
    def _extract_momentum_reward(self, command: str, default: int = 10) -> int:
        """Extract momentum reward from command"""
        import re
        match = re.search(r'(\d+)\s*(momentum|points|xp)', command.lower())
        if match:
            return int(match.group(1))
        return default
    
    def _extract_focus_areas(self, command: str) -> List[str]:
        """Extract focus areas for code review"""
        focus_keywords = {
            "security": ["security", "vulnerability", "safe", "secure"],
            "performance": ["performance", "speed", "optimize", "fast"],
            "style": ["style", "format", "clean", "readable"],
            "bugs": ["bug", "error", "fix", "issue"]
        }
        
        found = []
        command_lower = command.lower()
        for area, keywords in focus_keywords.items():
            if any(kw in command_lower for kw in keywords):
                found.append(area)
        
        return found if found else ["general"]


# Singleton instance
_command_router = None

def get_command_router() -> CommandRouter:
    """Get singleton CommandRouter instance"""
    global _command_router
    if _command_router is None:
        model_path = getattr(settings, 'INTENT_CLASSIFIER_MODEL_PATH', None)
        _command_router = CommandRouter(model_path=model_path)
    return _command_router

