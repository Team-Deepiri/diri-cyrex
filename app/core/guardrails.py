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


@dataclass
class SafetyThresholds:
    """Configurable safety thresholds"""
    # Safety level thresholds (0.0 to 1.0)
    critical_threshold: float = 0.8
    blocked_threshold: float = 0.6
    warning_threshold: float = 0.3
    
    # Prompt injection scoring
    injection_base_score: float = 0.5
    injection_increment: float = 0.1
    injection_max_score: float = 0.9
    
    # Blocked content scoring
    content_base_score: float = 0.4
    content_increment: float = 0.2
    content_max_score: float = 0.8
    content_blocked_threshold: float = 0.6
    
    # PII scoring
    pii_base_score: float = 0.3
    pii_increment: float = 0.2
    pii_max_score: float = 0.7
    
    # Output validation scoring
    output_length_score: float = 0.5
    output_json_invalid_score: float = 0.3
    output_blocked_threshold: float = 0.6
    output_warning_threshold: float = 0.3


class SafetyGuardrails:
    """
    Comprehensive safety guardrails for AI interactions
    Detects prompt injection, toxic content, PII, and validates outputs
    """
    
    def __init__(self, thresholds: Optional[SafetyThresholds] = None):
        self.logger = logger
        self.thresholds = thresholds or SafetyThresholds()
        self.blocked_patterns = self._load_blocked_patterns()
        self.pii_patterns = self._load_pii_patterns()
        self.injection_patterns = self._load_injection_patterns()
    
    def _load_blocked_patterns(self) -> List[re.Pattern]:
        """Load patterns for blocked content (Updated 2024)"""
        patterns = [
            # Explicit content
            re.compile(r'\b(?:explicit|nsfw|adult|pornographic)\b', re.IGNORECASE),
            # Violence
            re.compile(r'\b(?:violence|harm|kill|attack|assassinate|murder)\b', re.IGNORECASE),
            re.compile(r'\b(?:how\s+to\s+)?(?:make|create|build)\s+(?:a\s+)?(?:bomb|weapon|explosive)', re.IGNORECASE),
            # Hate speech indicators
            re.compile(r'\b(?:hate|discriminate|slur|racist|sexist|bigot)\b', re.IGNORECASE),
            # Illegal activities
            re.compile(r'\b(?:illegal|fraudulent|scam|phishing)\s+(?:activity|scheme)', re.IGNORECASE),
            # Self-harm
            re.compile(r'\b(?:how\s+to\s+)?(?:commit\s+suicide|kill\s+myself|end\s+my\s+life)', re.IGNORECASE),
        ]
        return patterns
    
    def _load_pii_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Load patterns for PII detection (Updated 2024) - Email addresses excluded"""
        patterns = [
            # Phone (US format)
            (re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'), 'phone'),
            # SSN
            (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), 'ssn'),
            # Credit card
            (re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'), 'credit_card'),
            # API keys and tokens
            (re.compile(r'\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|bearer\s+token)\s*[:=]\s*\S+', re.IGNORECASE), 'api_key'),
            # Passwords
            (re.compile(r'\b(?:password|passwd|pwd|pass)\s*[:=]\s*\S+', re.IGNORECASE), 'password'),
        ]
        return patterns
    
    def _load_injection_patterns(self) -> List[re.Pattern]:
        """Load patterns for prompt injection detection (Updated 2024)"""
        patterns = [
            # Ignore previous instructions
            re.compile(r'ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?|directives?)', re.IGNORECASE),
            re.compile(r'disregard\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?|rules?)', re.IGNORECASE),
            re.compile(r'forget\s+(?:everything|all|what)\s+(?:above|before|you\s+were\s+told)', re.IGNORECASE),
            # System override
            re.compile(r'system\s*:\s*|<\s*\|?(?:system|assistant|user)\s*\|?\s*>', re.IGNORECASE),
            # New instructions
            re.compile(r'new\s+(?:system\s+)?(?:instructions?|prompts?|rules?)\s*:', re.IGNORECASE),
            # Jailbreak attempts
            re.compile(r'jailbreak|bypass|override|unrestricted|unfiltered', re.IGNORECASE),
            re.compile(r'DAN\s+mode|developer\s+mode|evil\s+mode', re.IGNORECASE),
            # Roleplay-based bypasses
            re.compile(r'(?:pretend|act|roleplay|simulate)\s+(?:you\s+are|to\s+be|as)\s+(?:an?\s+)?(?:evil|malicious|unrestricted)', re.IGNORECASE),
            # Encoding attempts
            re.compile(r'(?:base64|hex|unicode|url\s+encode|decode|obfuscate)\s+(?:this|the|your)', re.IGNORECASE),
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
        
        # Determine overall level using configurable thresholds
        if max_score >= self.thresholds.critical_threshold:
            level = SafetyLevel.CRITICAL
        elif max_score >= self.thresholds.blocked_threshold:
            level = SafetyLevel.BLOCKED
        elif max_score >= self.thresholds.warning_threshold:
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
            score = min(
                self.thresholds.injection_max_score,
                self.thresholds.injection_base_score + len(matches) * self.thresholds.injection_increment
            )
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
            score = min(
                self.thresholds.content_max_score,
                self.thresholds.content_base_score + len(matches) * self.thresholds.content_increment
            )
            level = SafetyLevel.WARNING if score < self.thresholds.content_blocked_threshold else SafetyLevel.BLOCKED
            return SafetyCheckResult(
                level=level,
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
            score = min(
                self.thresholds.pii_max_score,
                self.thresholds.pii_base_score + len(found_pii) * self.thresholds.pii_increment
            )
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
                score=self.thresholds.output_length_score,
            ))
            max_score = self.thresholds.output_length_score
        
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
                    score=self.thresholds.output_json_invalid_score,
                ))
                max_score = max(max_score, self.thresholds.output_json_invalid_score)
        
        # Determine level using configurable thresholds
        if max_score >= self.thresholds.output_blocked_threshold:
            level = SafetyLevel.BLOCKED
        elif max_score >= self.thresholds.output_warning_threshold:
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

