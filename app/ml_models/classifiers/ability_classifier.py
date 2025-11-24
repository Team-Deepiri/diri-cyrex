"""
Ability Classifier - BERT-based intent classification for predefined abilities
"""
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import json

class AbilityClassifier:
    def __init__(self, model_path: str = "bert-base-uncased", num_abilities: int = 50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_abilities
        ).to(self.device)
        
        # Load ability mappings
        self.ability_map = self._load_ability_map()
        self.role_abilities = self._load_role_abilities()
    
    def _load_ability_map(self) -> Dict[int, str]:
        """Load ability ID to name mapping"""
        # TODO: Load from config file
        return {
            0: "summarize_text",
            1: "create_objective",
            2: "activate_focus_boost",
            3: "activate_velocity_boost",
            4: "generate_code_review",
            5: "refactor_suggest",
            # ... add all 50 abilities
        }
    
    def _load_role_abilities(self) -> Dict[str, List[str]]:
        """Load role-specific abilities"""
        return {
            "software_engineer": [
                "code_review", "debug_assist", "refactor_suggest",
                "documentation_gen", "test_generation", "commit_message_gen"
            ],
            "designer": [
                "design_critique", "color_palette_gen", "layout_suggest",
                "export_assets", "design_system_check"
            ],
            "product_manager": [
                "feature_breakdown", "user_story_gen", "sprint_planning",
                "roadmap_suggest", "stakeholder_update"
            ],
            "marketer": [
                "copy_gen", "campaign_suggest", "audience_analysis",
                "content_calendar", "seo_optimize"
            ]
        }
    
    def classify(self, command: str, user_role: str = None, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Classify user command to predefined abilities
        
        Args:
            command: User's natural language command
            user_role: User's role (filters relevant abilities)
            top_k: Number of top predictions to return
        
        Returns:
            List of (ability_name, confidence) tuples
        """
        # Tokenize input
        inputs = self.tokenizer(
            command,
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
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
        
        # Convert to ability names
        results = []
        for idx, prob in zip(top_indices[0], top_probs[0]):
            ability_id = idx.item()
            ability_name = self.ability_map.get(ability_id, f"unknown_{ability_id}")
            confidence = prob.item()
            
            # Filter by role if provided
            if user_role:
                role_abilities = self.role_abilities.get(user_role, [])
                if ability_name not in role_abilities:
                    continue
            
            results.append((ability_name, confidence))
        
        return results
    
    def classify_with_context(
        self, 
        command: str, 
        context: Dict,
        user_role: str = None
    ) -> Dict:
        """
        Classify with additional context
        
        Args:
            command: User command
            context: Additional context (file, project, etc.)
            user_role: User's role
        
        Returns:
            Dict with ability, confidence, and extracted parameters
        """
        # Get classification
        predictions = self.classify(command, user_role, top_k=1)
        
        if not predictions:
            return {
                "ability": None,
                "confidence": 0.0,
                "parameters": {}
            }
        
        ability_name, confidence = predictions[0]
        
        # Extract parameters based on ability type
        parameters = self._extract_parameters(ability_name, command, context)
        
        return {
            "ability": ability_name,
            "confidence": confidence,
            "parameters": parameters
        }
    
    def _extract_parameters(
        self, 
        ability_name: str, 
        command: str, 
        context: Dict
    ) -> Dict:
        """Extract parameters for specific ability"""
        # TODO: Implement parameter extraction logic
        params = {}
        
        if ability_name == "create_objective":
            params["title"] = self._extract_title(command)
            params["momentum_reward"] = 10  # Default
        
        elif ability_name == "activate_focus_boost":
            params["duration"] = self._extract_duration(command)
        
        # Add context
        params.update(context)
        
        return params
    
    def _extract_title(self, command: str) -> str:
        """Extract title from command"""
        # Simple implementation - can be improved with NER
        words = command.split()
        if "create" in words:
            idx = words.index("create")
            return " ".join(words[idx+1:])
        return command
    
    def _extract_duration(self, command: str) -> int:
        """Extract duration in minutes"""
        # Simple regex-based extraction
        import re
        match = re.search(r'(\d+)\s*(minute|min|hour|hr)', command.lower())
        if match:
            value = int(match.group(1))
            unit = match.group(2)
            if 'hour' in unit or 'hr' in unit:
                return value * 60
            return value
        return 60  # Default

# Example usage
if __name__ == "__main__":
    classifier = AbilityClassifier()
    
    # Test classification
    command = "Can you help me review this code for security issues?"
    results = classifier.classify(command, user_role="software_engineer")
    
    print(f"Command: {command}")
    print(f"Predictions: {results}")

