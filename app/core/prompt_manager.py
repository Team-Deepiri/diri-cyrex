"""
Prompt Version Manager
Manages prompt templates, versioning, and A/B testing
"""
from typing import Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from ..logging_config import get_logger

logger = get_logger("cyrex.prompt_manager")


class PromptVersion(str, Enum):
    """Prompt version status"""
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"
    TESTING = "testing"


@dataclass
class PromptTemplate:
    """Prompt template definition"""
    name: str
    template: str
    version: str = "1.0.0"
    status: PromptVersion = PromptVersion.DRAFT
    variables: list[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class PromptVersionManager:
    """
    Manages prompt templates with versioning
    Supports A/B testing and prompt evolution
    """
    
    def __init__(self, prompts_dir: Optional[str] = None):
        self.prompts_dir = Path(prompts_dir or "./prompts")
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.templates: Dict[str, PromptTemplate] = {}
        self.logger = logger
        self._load_prompts()
    
    def _load_prompts(self):
        """Load prompts from directory"""
        for prompt_file in self.prompts_dir.glob("*.json"):
            try:
                with open(prompt_file, 'r') as f:
                    data = json.load(f)
                    template = PromptTemplate(**data)
                    self.templates[template.name] = template
            except Exception as e:
                self.logger.warning(f"Failed to load prompt {prompt_file}: {e}")
    
    def register_prompt(
        self,
        name: str,
        template: str,
        version: str = "1.0.0",
        status: PromptVersion = PromptVersion.DRAFT,
        variables: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptTemplate:
        """Register a new prompt template"""
        prompt = PromptTemplate(
            name=name,
            template=template,
            version=version,
            status=status,
            variables=variables or [],
            metadata=metadata or {},
        )
        
        self.templates[name] = prompt
        self._save_prompt(prompt)
        
        self.logger.info(f"Registered prompt: {name} v{version}")
        return prompt
    
    def _save_prompt(self, prompt: PromptTemplate):
        """Save prompt to file"""
        prompt_file = self.prompts_dir / f"{prompt.name}.json"
        with open(prompt_file, 'w') as f:
            json.dump({
                "name": prompt.name,
                "template": prompt.template,
                "version": prompt.version,
                "status": prompt.status.value,
                "variables": prompt.variables,
                "created_at": prompt.created_at.isoformat(),
                "updated_at": prompt.updated_at.isoformat(),
                "metadata": prompt.metadata,
            }, f, indent=2)
    
    def get_prompt(self, name: str, **kwargs) -> str:
        """
        Get prompt template and format with variables
        
        Args:
            name: Prompt template name
            **kwargs: Variables to fill in template
        
        Returns:
            Formatted prompt string
        """
        if name not in self.templates:
            # Return default prompt if not found
            self.logger.warning(f"Prompt {name} not found, using default")
            return kwargs.get("question", kwargs.get("input", ""))
        
        template = self.templates[name]
        prompt_text = template.template
        
        # Format with variables
        try:
            return prompt_text.format(**kwargs)
        except KeyError as e:
            self.logger.error(f"Missing variable in prompt {name}: {e}")
            return prompt_text
    
    def list_prompts(self, status: Optional[PromptVersion] = None) -> list[PromptTemplate]:
        """List prompts, optionally filtered by status"""
        prompts = list(self.templates.values())
        
        if status:
            prompts = [p for p in prompts if p.status == status]
        
        return prompts
    
    def update_prompt(
        self,
        name: str,
        template: Optional[str] = None,
        version: Optional[str] = None,
        status: Optional[PromptVersion] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptTemplate:
        """Update an existing prompt"""
        if name not in self.templates:
            raise ValueError(f"Prompt {name} not found")
        
        prompt = self.templates[name]
        
        if template:
            prompt.template = template
        if version:
            prompt.version = version
        if status:
            prompt.status = status
        if metadata:
            prompt.metadata.update(metadata)
        
        prompt.updated_at = datetime.now()
        self._save_prompt(prompt)
        
        self.logger.info(f"Updated prompt: {name} v{prompt.version}")
        return prompt


# Default prompts
DEFAULT_PROMPTS = {
    "general_qa": """You are a helpful AI assistant. Answer the following question clearly and concisely.

Question: {question}

Answer:""",
    
    "rag_qa": """Use the following context to answer the question. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:""",
    
    "tool_use": """You are an AI assistant with access to tools. Use the tools when appropriate to help the user.

User request: {input}

Available tools: {tools}

Think step by step and use tools as needed.""",
}


def get_prompt_manager() -> PromptVersionManager:
    """Get global prompt manager instance"""
    manager = PromptVersionManager()
    
    # Register default prompts if not already registered
    for name, template in DEFAULT_PROMPTS.items():
        if name not in manager.templates:
            manager.register_prompt(
                name=name,
                template=template,
                version="1.0.0",
                status=PromptVersion.ACTIVE,
            )
    
    return manager

