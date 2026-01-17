"""
Safety Guardrails
Content filtering, prompt injection detection, output validation
Enterprise-grade safety checks for AI interactions
"""
from typing import Dict, List, Optional, Any, Tuple, Type
from enum import Enum
from dataclasses import dataclass
import re
from datetime import datetime
from ..logging_config import get_logger

logger = get_logger("cyrex.guardrails")


class SafetyLevel(str, Enum):
    """Safety levels"""
    SAFE = "safe"
    WARNING = "warning"
    BLOCKED = "blocked"
    CRITICAL = "critical"


@dataclass
class SafetyCheckResult:
    """Result of a safety check"""
    level: SafetyLevel
    message: str
    score: float  # 0.0 (safe) to 1.0 (critical)
    details: Dict[str, Any] = None


class SafetyGuardrails:
    """
    Comprehensive safety guardrails for AI interactions
    Detects prompt injection, toxic content, PII, and validates outputs
    """
    
    def __init__(self):
        self.logger = logger
        self.blocked_patterns = self._load_blocked_patterns()
        self.pii_patterns = self._load_pii_patterns()
        self.injection_patterns = self._load_injection_patterns()
    
    def _load_blocked_patterns(self) -> List[re.Pattern]:
        """Load patterns for blocked content"""
        patterns = [
            # Explicit content
            re.compile(r'\b(?:explicit|nsfw|adult)\b', re.IGNORECASE),
            # Violence
            re.compile(r'\b(?:violence|harm|kill|attack)\b', re.IGNORECASE),
            # Hate speech indicators
            re.compile(r'\b(?:hate|discriminate|slur)\b', re.IGNORECASE),
        ]
        return patterns
    
    def _load_pii_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Load patterns for PII detection"""
        patterns = [
            # Email addresses excluded - commonly shared in business contexts
            # Phone (US format)
            (re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'), 'phone'),
            # SSN
            (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), 'ssn'),
            # Credit card
            (re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'), 'credit_card'),
        ]
        return patterns
    
    def _load_injection_patterns(self) -> List[re.Pattern]:
        """Load patterns for prompt injection detection"""
        patterns = [
            # Ignore previous instructions
            re.compile(r'ignore\s+(?:previous|all|above)\s+instructions?', re.IGNORECASE),
            # System override
            re.compile(r'system\s*:\s*', re.IGNORECASE),
            # New instructions
            re.compile(r'new\s+instructions?', re.IGNORECASE),
            # Jailbreak attempts
            re.compile(r'jailbreak|bypass|override', re.IGNORECASE),
        ]
        return patterns
    
    def check_prompt(self, prompt: str, user_id: Optional[str] = None) -> SafetyCheckResult:
        """
        Check prompt for safety issues
        
        Returns:
            SafetyCheckResult with safety level and details
        """
        checks = []
        max_score = 0.0
        
        # Check for prompt injection
        injection_result = self._check_prompt_injection(prompt)
        checks.append(injection_result)
        max_score = max(max_score, injection_result.score)
        
        # Check for blocked content
        content_result = self._check_blocked_content(prompt)
        checks.append(content_result)
        max_score = max(max_score, content_result.score)
        
        # Check for PII
        pii_result = self._check_pii(prompt)
        checks.append(pii_result)
        max_score = max(max_score, pii_result.score)
        
        # Determine overall level
        if max_score >= 0.8:
            level = SafetyLevel.CRITICAL
        elif max_score >= 0.6:
            level = SafetyLevel.BLOCKED
        elif max_score >= 0.3:
            level = SafetyLevel.WARNING
        else:
            level = SafetyLevel.SAFE
        
        message = f"Prompt safety check: {level.value} (score: {max_score:.2f})"
        
        return SafetyCheckResult(
            level=level,
            message=message,
            score=max_score,
            details={
                "checks": [c.__dict__ for c in checks],
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
            }
        )
    
    def _check_prompt_injection(self, prompt: str) -> SafetyCheckResult:
        """Check for prompt injection attempts"""
        matches = []
        
        for pattern in self.injection_patterns:
            if pattern.search(prompt):
                matches.append(pattern.pattern)
        
        if matches:
            score = min(0.9, 0.5 + len(matches) * 0.1)
            return SafetyCheckResult(
                level=SafetyLevel.BLOCKED,
                message=f"Potential prompt injection detected: {len(matches)} patterns matched",
                score=score,
                details={"matched_patterns": matches}
            )
        
        return SafetyCheckResult(
            level=SafetyLevel.SAFE,
            message="No prompt injection detected",
            score=0.0
        )
    
    def _check_blocked_content(self, text: str) -> SafetyCheckResult:
        """Check for blocked content patterns"""
        matches = []
        
        for pattern in self.blocked_patterns:
            if pattern.search(text):
                matches.append(pattern.pattern)
        
        if matches:
            score = min(0.8, 0.4 + len(matches) * 0.2)
            return SafetyCheckResult(
                level=SafetyLevel.WARNING if score < 0.6 else SafetyLevel.BLOCKED,
                message=f"Blocked content patterns detected: {len(matches)} matches",
                score=score,
                details={"matched_patterns": matches}
            )
        
        return SafetyCheckResult(
            level=SafetyLevel.SAFE,
            message="No blocked content detected",
            score=0.0
        )
    
    def _check_pii(self, text: str) -> SafetyCheckResult:
        """Check for personally identifiable information"""
        found_pii = []
        
        for pattern, pii_type in self.pii_patterns:
            matches = pattern.findall(text)
            if matches:
                found_pii.append({
                    "type": pii_type,
                    "count": len(matches),
                    "samples": matches[:3]  # First 3 samples
                })
        
        if found_pii:
            score = min(0.7, 0.3 + len(found_pii) * 0.2)
            return SafetyCheckResult(
                level=SafetyLevel.WARNING,
                message=f"PII detected: {len(found_pii)} types",
                score=score,
                details={"pii_types": found_pii}
            )
        
        return SafetyCheckResult(
            level=SafetyLevel.SAFE,
            message="No PII detected",
            score=0.0
        )
    
    def check_output(
        self,
        output: str,
        expected_format: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> SafetyCheckResult:
        """
        Check output for safety and format compliance
        
        Args:
            output: Generated output to check
            expected_format: Expected format (e.g., "json", "markdown")
            max_length: Maximum allowed length
        """
        checks = []
        max_score = 0.0
        
        # Check length
        if max_length and len(output) > max_length:
            checks.append(SafetyCheckResult(
                level=SafetyLevel.WARNING,
                message=f"Output exceeds max length: {len(output)} > {max_length}",
                score=0.5,
            ))
            max_score = 0.5
        
        # Check for blocked content in output
        content_result = self._check_blocked_content(output)
        checks.append(content_result)
        max_score = max(max_score, content_result.score)
        
        # Check format if specified
        if expected_format == "json":
            try:
                import json
                json.loads(output)
            except:
                checks.append(SafetyCheckResult(
                    level=SafetyLevel.WARNING,
                    message="Output is not valid JSON",
                    score=0.3,
                ))
                max_score = max(max_score, 0.3)
        
        # Determine level
        if max_score >= 0.6:
            level = SafetyLevel.BLOCKED
        elif max_score >= 0.3:
            level = SafetyLevel.WARNING
        else:
            level = SafetyLevel.SAFE
        
        return SafetyCheckResult(
            level=level,
            message=f"Output safety check: {level.value}",
            score=max_score,
            details={"checks": [c.__dict__ for c in checks]}
        )
    
    def validate_tool_output(
        self,
        tool_name: str,
        output: Any,
        expected_type: Optional[Type] = None,
    ) -> SafetyCheckResult:
        """Validate tool output"""
        try:
            # Type checking
            if expected_type and not isinstance(output, expected_type):
                return SafetyCheckResult(
                    level=SafetyLevel.WARNING,
                    message=f"Tool output type mismatch: expected {expected_type}, got {type(output)}",
                    score=0.4,
                )
            
            # String output safety check
            if isinstance(output, str):
                return self.check_output(output)
            
            return SafetyCheckResult(
                level=SafetyLevel.SAFE,
                message="Tool output validated",
                score=0.0,
            )
        
        except Exception as e:
            return SafetyCheckResult(
                level=SafetyLevel.WARNING,
                message=f"Tool output validation error: {str(e)}",
                score=0.5,
            )
    
    def should_block(self, result: SafetyCheckResult) -> bool:
        """Determine if result should be blocked"""
        return result.level in [SafetyLevel.BLOCKED, SafetyLevel.CRITICAL]


def get_guardrails() -> SafetyGuardrails:
    """Get global guardrails instance"""
    return SafetyGuardrails()

