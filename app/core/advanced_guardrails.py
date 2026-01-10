"""
Advanced Guardrails System
Comprehensive safety, validation, and policy enforcement for AI agents
"""
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import re
import json
from ..logging_config import get_logger

logger = get_logger("cyrex.advanced_guardrails")


class RiskLevel(str, Enum):
    """Risk assessment levels"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GuardrailAction(str, Enum):
    """Actions to take on guardrail violations"""
    ALLOW = "allow"
    WARN = "warn"
    MODIFY = "modify"
    BLOCK = "block"
    ESCALATE = "escalate"
    LOG = "log"


class GuardrailCategory(str, Enum):
    """Categories of guardrails"""
    CONTENT_SAFETY = "content_safety"
    PROMPT_INJECTION = "prompt_injection"
    DATA_PRIVACY = "data_privacy"
    OUTPUT_VALIDATION = "output_validation"
    RATE_LIMITING = "rate_limiting"
    TOOL_SAFETY = "tool_safety"
    CONTEXT_BOUNDARY = "context_boundary"
    ETHICAL = "ethical"


@dataclass
class GuardrailResult:
    """Result of a guardrail check"""
    passed: bool
    risk_level: RiskLevel = RiskLevel.SAFE
    action: GuardrailAction = GuardrailAction.ALLOW
    category: GuardrailCategory = GuardrailCategory.CONTENT_SAFETY
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    modified_content: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "risk_level": self.risk_level.value,
            "action": self.action.value,
            "category": self.category.value,
            "message": self.message,
            "details": self.details,
            "modified_content": self.modified_content,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class GuardrailPolicy:
    """Policy configuration for guardrails"""
    name: str
    category: GuardrailCategory
    enabled: bool = True
    risk_threshold: RiskLevel = RiskLevel.MEDIUM
    action_on_violation: GuardrailAction = GuardrailAction.BLOCK
    patterns: List[str] = field(default_factory=list)
    allow_list: List[str] = field(default_factory=list)
    block_list: List[str] = field(default_factory=list)
    custom_check: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedGuardrails:
    """
    Comprehensive guardrail system for AI safety
    
    Features:
    - Content safety (toxicity, violence, etc.)
    - Prompt injection detection
    - PII/PHI data protection
    - Output validation
    - Rate limiting
    - Tool execution safety
    - Context boundary enforcement
    - Ethical guidelines
    """
    
    def __init__(self):
        self.policies: Dict[str, GuardrailPolicy] = {}
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        self._rate_limits: Dict[str, Dict[str, Any]] = {}
        self._blocked_tools: Set[str] = set()
        self.logger = logger
        
        # Initialize default policies
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default guardrail policies"""
        
        # Prompt Injection Detection
        self.add_policy(GuardrailPolicy(
            name="prompt_injection",
            category=GuardrailCategory.PROMPT_INJECTION,
            risk_threshold=RiskLevel.HIGH,
            action_on_violation=GuardrailAction.BLOCK,
            patterns=[
                r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?)",
                r"disregard\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions?|prompts?)",
                r"forget\s+(?:everything|all)\s+(?:above|before)",
                r"system\s*:\s*",
                r"<\|?(?:system|assistant|user)\|?>",
                r"new\s+(?:system\s+)?instructions?\s*:",
                r"jailbreak",
                r"DAN\s+mode",
                r"developer\s+mode",
                r"bypass\s+(?:safety|restrictions?|filters?)",
                r"pretend\s+(?:you\s+are|to\s+be)",
                r"roleplay\s+as\s+(?:an?\s+)?(?:evil|malicious)",
            ],
        ))
        
        # Content Safety
        self.add_policy(GuardrailPolicy(
            name="harmful_content",
            category=GuardrailCategory.CONTENT_SAFETY,
            risk_threshold=RiskLevel.HIGH,
            action_on_violation=GuardrailAction.BLOCK,
            patterns=[
                r"\b(?:how\s+to\s+)?(?:make|create|build)\s+(?:a\s+)?(?:bomb|weapon|explosive)",
                r"\b(?:how\s+to\s+)?(?:harm|hurt|kill|attack)\s+(?:someone|people)",
                r"\b(?:synthesize|produce)\s+(?:drugs?|narcotics?|poison)",
            ],
        ))
        
        # PII Detection
        self.add_policy(GuardrailPolicy(
            name="pii_detection",
            category=GuardrailCategory.DATA_PRIVACY,
            risk_threshold=RiskLevel.MEDIUM,
            action_on_violation=GuardrailAction.WARN,
            patterns=[
                r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b",  # SSN
                r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",  # Credit card
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
                r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # Phone
                r"\b(?:password|passwd|pwd)\s*[:=]\s*\S+",  # Passwords
                r"\b(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*\S+",  # API keys
            ],
        ))
        
        # Output Length Limits
        self.add_policy(GuardrailPolicy(
            name="output_limits",
            category=GuardrailCategory.OUTPUT_VALIDATION,
            risk_threshold=RiskLevel.LOW,
            action_on_violation=GuardrailAction.MODIFY,
            metadata={
                "max_length": 10000,
                "max_tokens": 4000,
            },
        ))
        
        # Tool Safety
        self.add_policy(GuardrailPolicy(
            name="tool_safety",
            category=GuardrailCategory.TOOL_SAFETY,
            risk_threshold=RiskLevel.HIGH,
            action_on_violation=GuardrailAction.BLOCK,
            block_list=[
                "execute_shell",
                "run_command",
                "delete_database",
                "drop_table",
                "rm_rf",
            ],
        ))
        
        # Ethical Guidelines
        self.add_policy(GuardrailPolicy(
            name="ethical_guidelines",
            category=GuardrailCategory.ETHICAL,
            risk_threshold=RiskLevel.MEDIUM,
            action_on_violation=GuardrailAction.WARN,
            patterns=[
                r"\b(?:discriminate|hate|racist|sexist)\b",
                r"\b(?:illegal|fraudulent|scam)\s+(?:activity|scheme)",
            ],
        ))
    
    def add_policy(self, policy: GuardrailPolicy):
        """Add a guardrail policy"""
        self.policies[policy.name] = policy
        
        # Compile patterns
        if policy.patterns:
            self._compiled_patterns[policy.name] = [
                re.compile(p, re.IGNORECASE) for p in policy.patterns
            ]
        
        self.logger.info(f"Added guardrail policy: {policy.name}")
    
    def remove_policy(self, policy_name: str):
        """Remove a guardrail policy"""
        self.policies.pop(policy_name, None)
        self._compiled_patterns.pop(policy_name, None)
    
    def enable_policy(self, policy_name: str, enabled: bool = True):
        """Enable or disable a policy"""
        if policy_name in self.policies:
            self.policies[policy_name].enabled = enabled
    
    # ========================================================================
    # Check Methods
    # ========================================================================
    
    async def check_input(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> GuardrailResult:
        """Check input for violations"""
        results = []
        
        for policy_name, policy in self.policies.items():
            if not policy.enabled:
                continue
            
            if policy.category in [
                GuardrailCategory.PROMPT_INJECTION,
                GuardrailCategory.CONTENT_SAFETY,
                GuardrailCategory.DATA_PRIVACY,
                GuardrailCategory.ETHICAL,
            ]:
                result = await self._check_patterns(input_text, policy)
                if not result.passed:
                    results.append(result)
        
        # Return worst result or safe
        if results:
            # Sort by risk level (highest first)
            risk_order = [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]
            results.sort(key=lambda r: risk_order.index(r.risk_level) if r.risk_level in risk_order else 99)
            return results[0]
        
        return GuardrailResult(
            passed=True,
            risk_level=RiskLevel.SAFE,
            action=GuardrailAction.ALLOW,
            message="Input passed all guardrail checks",
        )
    
    async def check_output(
        self,
        output_text: str,
        expected_format: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> GuardrailResult:
        """Check output for violations and format compliance"""
        results = []
        
        # Check length limits
        output_policy = self.policies.get("output_limits")
        if output_policy and output_policy.enabled:
            max_len = max_length or output_policy.metadata.get("max_length", 10000)
            if len(output_text) > max_len:
                results.append(GuardrailResult(
                    passed=False,
                    risk_level=RiskLevel.LOW,
                    action=GuardrailAction.MODIFY,
                    category=GuardrailCategory.OUTPUT_VALIDATION,
                    message=f"Output exceeds max length ({len(output_text)} > {max_len})",
                    modified_content=output_text[:max_len] + "... [truncated]",
                ))
        
        # Check for data leaks in output
        pii_policy = self.policies.get("pii_detection")
        if pii_policy and pii_policy.enabled:
            result = await self._check_patterns(output_text, pii_policy)
            if not result.passed:
                result.message = f"Output contains sensitive data: {result.message}"
                results.append(result)
        
        # Validate format if specified
        if expected_format == "json":
            try:
                json.loads(output_text)
            except json.JSONDecodeError as e:
                results.append(GuardrailResult(
                    passed=False,
                    risk_level=RiskLevel.LOW,
                    action=GuardrailAction.WARN,
                    category=GuardrailCategory.OUTPUT_VALIDATION,
                    message=f"Invalid JSON format: {str(e)}",
                ))
        
        # Check for harmful content in output
        for policy_name, policy in self.policies.items():
            if not policy.enabled:
                continue
            if policy.category == GuardrailCategory.CONTENT_SAFETY:
                result = await self._check_patterns(output_text, policy)
                if not result.passed:
                    result.message = f"Output contains harmful content: {result.message}"
                    results.append(result)
        
        if results:
            results.sort(key=lambda r: [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW].index(r.risk_level) if r.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW] else 99)
            return results[0]
        
        return GuardrailResult(
            passed=True,
            risk_level=RiskLevel.SAFE,
            action=GuardrailAction.ALLOW,
            message="Output passed all guardrail checks",
        )
    
    async def check_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: Optional[str] = None,
    ) -> GuardrailResult:
        """Check tool call for safety"""
        tool_policy = self.policies.get("tool_safety")
        if not tool_policy or not tool_policy.enabled:
            return GuardrailResult(passed=True, action=GuardrailAction.ALLOW)
        
        # Check block list
        if tool_name in tool_policy.block_list or tool_name in self._blocked_tools:
            return GuardrailResult(
                passed=False,
                risk_level=RiskLevel.HIGH,
                action=GuardrailAction.BLOCK,
                category=GuardrailCategory.TOOL_SAFETY,
                message=f"Tool '{tool_name}' is blocked by policy",
            )
        
        # Check parameters for dangerous patterns
        params_str = json.dumps(parameters)
        dangerous_patterns = [
            r";\s*(?:rm|del|drop|delete)",
            r"--\s*",
            r"'\s*(?:OR|AND)\s*'",
            r"<script",
            r"javascript:",
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, params_str, re.IGNORECASE):
                return GuardrailResult(
                    passed=False,
                    risk_level=RiskLevel.HIGH,
                    action=GuardrailAction.BLOCK,
                    category=GuardrailCategory.TOOL_SAFETY,
                    message=f"Dangerous pattern detected in tool parameters",
                    details={"pattern": pattern},
                )
        
        return GuardrailResult(
            passed=True,
            action=GuardrailAction.ALLOW,
            message="Tool call passed safety checks",
        )
    
    async def check_rate_limit(
        self,
        user_id: str,
        action: str = "default",
        limit: int = 100,
        window_seconds: int = 60,
    ) -> GuardrailResult:
        """Check rate limiting"""
        key = f"{user_id}:{action}"
        now = datetime.utcnow()
        
        if key not in self._rate_limits:
            self._rate_limits[key] = {"count": 0, "window_start": now}
        
        rate_info = self._rate_limits[key]
        
        # Reset window if expired
        if (now - rate_info["window_start"]).total_seconds() > window_seconds:
            rate_info["count"] = 0
            rate_info["window_start"] = now
        
        rate_info["count"] += 1
        
        if rate_info["count"] > limit:
            return GuardrailResult(
                passed=False,
                risk_level=RiskLevel.MEDIUM,
                action=GuardrailAction.BLOCK,
                category=GuardrailCategory.RATE_LIMITING,
                message=f"Rate limit exceeded: {rate_info['count']}/{limit} in {window_seconds}s",
                details={
                    "current_count": rate_info["count"],
                    "limit": limit,
                    "window_seconds": window_seconds,
                },
            )
        
        return GuardrailResult(
            passed=True,
            action=GuardrailAction.ALLOW,
            details={"remaining": limit - rate_info["count"]},
        )
    
    async def check_context_boundary(
        self,
        content: str,
        allowed_topics: Optional[List[str]] = None,
        blocked_topics: Optional[List[str]] = None,
    ) -> GuardrailResult:
        """Check if content stays within allowed context boundaries"""
        content_lower = content.lower()
        
        # Check blocked topics
        if blocked_topics:
            for topic in blocked_topics:
                if topic.lower() in content_lower:
                    return GuardrailResult(
                        passed=False,
                        risk_level=RiskLevel.MEDIUM,
                        action=GuardrailAction.WARN,
                        category=GuardrailCategory.CONTEXT_BOUNDARY,
                        message=f"Content touches blocked topic: {topic}",
                    )
        
        return GuardrailResult(
            passed=True,
            action=GuardrailAction.ALLOW,
            message="Content within context boundaries",
        )
    
    async def _check_patterns(
        self,
        text: str,
        policy: GuardrailPolicy,
    ) -> GuardrailResult:
        """Check text against policy patterns"""
        compiled = self._compiled_patterns.get(policy.name, [])
        
        for pattern in compiled:
            match = pattern.search(text)
            if match:
                return GuardrailResult(
                    passed=False,
                    risk_level=policy.risk_threshold,
                    action=policy.action_on_violation,
                    category=policy.category,
                    message=f"Pattern match: {match.group()}",
                    details={
                        "policy": policy.name,
                        "match": match.group(),
                        "position": match.start(),
                    },
                )
        
        return GuardrailResult(
            passed=True,
            action=GuardrailAction.ALLOW,
            category=policy.category,
        )
    
    # ========================================================================
    # Comprehensive Check
    # ========================================================================
    
    async def check(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Comprehensive guardrail check (backward compatible)"""
        result = await self.check_input(input_text, context)
        
        return {
            "safe": result.passed,
            "action": result.action.value,
            "risk_level": result.risk_level.value,
            "message": result.message,
            "details": result.details,
        }
    
    # ========================================================================
    # Tool Blocking
    # ========================================================================
    
    def block_tool(self, tool_name: str):
        """Block a specific tool"""
        self._blocked_tools.add(tool_name)
    
    def unblock_tool(self, tool_name: str):
        """Unblock a specific tool"""
        self._blocked_tools.discard(tool_name)
    
    def get_blocked_tools(self) -> List[str]:
        """Get list of blocked tools"""
        return list(self._blocked_tools)


# ============================================================================
# Singleton Instance
# ============================================================================

_guardrails: Optional[AdvancedGuardrails] = None


async def get_advanced_guardrails() -> AdvancedGuardrails:
    """Get or create advanced guardrails singleton"""
    global _guardrails
    if _guardrails is None:
        _guardrails = AdvancedGuardrails()
    return _guardrails

