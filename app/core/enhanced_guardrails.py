"""
Enhanced Guardrails System
Comprehensive safety, content filtering, and policy enforcement
"""
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import re
import asyncio
from ..database.postgres import get_postgres_manager
from ..logging_config import get_logger
import json

logger = get_logger("cyrex.guardrails")


class GuardrailRule:
    """Individual guardrail rule definition"""
    
    def __init__(
        self,
        rule_id: str,
        name: str,
        pattern: str,
        action: str = "block",  # "block", "warn", "modify", "log"
        severity: str = "medium",  # "low", "medium", "high", "critical"
        description: str = "",
    ):
        self.rule_id = rule_id
        self.name = name
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.action = action
        self.severity = severity
        self.description = description
    
    def check(self, text: str) -> Optional[Dict[str, Any]]:
        """Check if text violates this rule"""
        match = self.pattern.search(text)
        if match:
            return {
                "rule_id": self.rule_id,
                "name": self.name,
                "action": self.action,
                "severity": self.severity,
                "match": match.group(),
                "description": self.description,
            }
        return None


class EnhancedGuardrails:
    """
    Enhanced guardrails system with rule-based and ML-based filtering
    Supports custom rules, content moderation, and policy enforcement
    """
    
    def __init__(self):
        self._rules: Dict[str, GuardrailRule] = {}
        self._custom_validators: List[Callable] = []
        self.logger = logger
    
    async def initialize(self):
        """Initialize guardrails and create database tables"""
        # Create guardrails tables in cyrex schema
        postgres = await get_postgres_manager()
        await postgres.execute("CREATE SCHEMA IF NOT EXISTS cyrex")
        await postgres.execute("""
            CREATE TABLE IF NOT EXISTS cyrex.guardrail_rules (
                rule_id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                pattern TEXT NOT NULL,
                action VARCHAR(50) NOT NULL,
                severity VARCHAR(50) NOT NULL,
                description TEXT,
                enabled BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_guardrails_enabled ON cyrex.guardrail_rules(enabled);
        """)
        
        # Create violations log table
        await postgres.execute("""
            CREATE TABLE IF NOT EXISTS cyrex.guardrail_violations (
                violation_id VARCHAR(255) PRIMARY KEY,
                rule_id VARCHAR(255) NOT NULL,
                content TEXT,
                action_taken VARCHAR(50),
                severity VARCHAR(50),
                metadata JSONB,
                timestamp TIMESTAMP NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_violations_rule_id ON cyrex.guardrail_violations(rule_id);
            CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON cyrex.guardrail_violations(timestamp);
        """)
        
        # Load default rules
        await self._load_default_rules()
        
        # Load rules from database
        await self._load_rules_from_db()
        
        self.logger.info("Enhanced guardrails initialized")
    
    async def _load_default_rules(self):
        """Load default safety rules with comprehensive, up-to-date patterns"""
        default_rules = [
            # Prompt Injection Detection
            {
                "rule_id": "prompt_injection_1",
                "name": "Prompt Injection - Ignore Instructions",
                "pattern": r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?|directives?)",
                "action": "block",
                "severity": "critical",
                "description": "Detects attempts to ignore previous instructions",
            },
            {
                "rule_id": "prompt_injection_2",
                "name": "Prompt Injection - System Override",
                "pattern": r"(?:system|assistant|user)\s*:\s*|<\s*\|?(?:system|assistant|user)\s*\|?\s*>",
                "action": "block",
                "severity": "critical",
                "description": "Detects system prompt override attempts",
            },
            {
                "rule_id": "prompt_injection_3",
                "name": "Prompt Injection - New Instructions",
                "pattern": r"new\s+(?:system\s+)?(?:instructions?|prompts?|rules?)\s*:",
                "action": "block",
                "severity": "critical",
                "description": "Detects attempts to inject new instructions",
            },
            {
                "rule_id": "prompt_injection_4",
                "name": "Prompt Injection - Jailbreak",
                "pattern": r"(?:jailbreak|bypass|override|unrestricted|unfiltered|developer\s+mode|DAN\s+mode|evil\s+mode)",
                "action": "block",
                "severity": "critical",
                "description": "Detects jailbreak attempts",
            },
            {
                "rule_id": "prompt_injection_5",
                "name": "Prompt Injection - Roleplay Bypass",
                "pattern": r"(?:pretend|act|roleplay|simulate)\s+(?:you\s+are|to\s+be|as)\s+(?:an?\s+)?(?:evil|malicious|unrestricted|unfiltered)",
                "action": "block",
                "severity": "critical",
                "description": "Detects roleplay-based bypass attempts",
            },
            {
                "rule_id": "prompt_injection_6",
                "name": "Prompt Injection - Encoding Bypass",
                "pattern": r"(?:base64|hex|unicode|url\s+encode|decode|obfuscate)",
                "action": "warn",
                "severity": "high",
                "description": "Detects potential encoding-based bypass attempts",
            },
            # PII Detection (Email addresses excluded - commonly shared in business contexts)
            {
                "rule_id": "pii_ssn",
                "name": "PII Detection - SSN",
                "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
                "action": "warn",
                "severity": "high",
                "description": "Detects Social Security Numbers",
            },
            {
                "rule_id": "pii_credit_card",
                "name": "PII Detection - Credit Card",
                "pattern": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
                "action": "warn",
                "severity": "high",
                "description": "Detects credit card numbers",
            },
            {
                "rule_id": "pii_phone",
                "name": "PII Detection - Phone Number",
                "pattern": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
                "action": "warn",
                "severity": "medium",
                "description": "Detects phone numbers",
            },
            {
                "rule_id": "pii_api_key",
                "name": "PII Detection - API Keys",
                "pattern": r"\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|bearer\s+token)\s*[:=]\s*\S+",
                "action": "block",
                "severity": "critical",
                "description": "Detects exposed API keys or tokens",
            },
            {
                "rule_id": "pii_password",
                "name": "PII Detection - Passwords",
                "pattern": r"\b(?:password|passwd|pwd|pass)\s*[:=]\s*\S+",
                "action": "block",
                "severity": "critical",
                "description": "Detects exposed passwords",
            },
            # Content Safety
            {
                "rule_id": "harmful_content_1",
                "name": "Harmful Content - Violence",
                "pattern": r"\b(?:how\s+to\s+)?(?:make|create|build|synthesize)\s+(?:a\s+)?(?:bomb|weapon|explosive|poison)",
                "action": "block",
                "severity": "critical",
                "description": "Detects instructions for creating harmful items",
            },
            {
                "rule_id": "harmful_content_2",
                "name": "Harmful Content - Harm Instructions",
                "pattern": r"\b(?:how\s+to\s+)?(?:harm|hurt|kill|attack|assassinate)\s+(?:someone|people|person)",
                "action": "block",
                "severity": "critical",
                "description": "Detects instructions to harm others",
            },
            {
                "rule_id": "harmful_content_3",
                "name": "Harmful Content - Illegal Drugs",
                "pattern": r"\b(?:synthesize|produce|make|create)\s+(?:drugs?|narcotics?|illegal\s+substances?)",
                "action": "block",
                "severity": "critical",
                "description": "Detects instructions for illegal drug production",
            },
            # Toxicity
            {
                "rule_id": "toxicity_1",
                "name": "Toxicity Detection - Hate Speech",
                "pattern": r"\b(?:hate|discriminate|racist|sexist|bigot)\b",
                "action": "warn",
                "severity": "high",
                "description": "Detects potentially toxic language",
            },
            {
                "rule_id": "toxicity_2",
                "name": "Toxicity Detection - Profanity",
                "pattern": r"\b(?:fuck|shit|damn|bitch|asshole|bastard)\b",
                "action": "warn",
                "severity": "medium",
                "description": "Detects profane language",
            },
            # Ethical Guidelines
            {
                "rule_id": "ethical_1",
                "name": "Ethical - Illegal Activities",
                "pattern": r"\b(?:illegal|fraudulent|scam|phishing)\s+(?:activity|scheme|operation)",
                "action": "block",
                "severity": "high",
                "description": "Detects references to illegal activities",
            },
        ]
        
        for rule_data in default_rules:
            rule = GuardrailRule(**rule_data)
            self._rules[rule.rule_id] = rule
    
    async def _load_rules_from_db(self):
        """Load rules from database"""
        postgres = await get_postgres_manager()
        rows = await postgres.fetch("SELECT * FROM cyrex.guardrail_rules WHERE enabled = TRUE")
        
        for row in rows:
            rule = GuardrailRule(
                rule_id=row['rule_id'],
                name=row['name'],
                pattern=row['pattern'],
                action=row['action'],
                severity=row['severity'],
                description=row['description'] or "",
            )
            self._rules[rule.rule_id] = rule
    
    async def add_rule(
        self,
        rule_id: str,
        name: str,
        pattern: str,
        action: str = "block",
        severity: str = "medium",
        description: str = "",
    ):
        """Add a custom guardrail rule"""
        rule = GuardrailRule(rule_id, name, pattern, action, severity, description)
        self._rules[rule_id] = rule
        
        # Persist to database
        postgres = await get_postgres_manager()
        await postgres.execute("""
            INSERT INTO cyrex.guardrail_rules (rule_id, name, pattern, action, severity, description, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (rule_id) DO UPDATE SET
                name = EXCLUDED.name,
                pattern = EXCLUDED.pattern,
                action = EXCLUDED.action,
                severity = EXCLUDED.severity,
                description = EXCLUDED.description,
                updated_at = EXCLUDED.updated_at
        """, rule_id, name, pattern, action, severity, description, datetime.utcnow(), datetime.utcnow())
        
        self.logger.info(f"Guardrail rule added: {rule_id}")
    
    async def check(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Check text against all guardrails
        
        Returns:
            {
                "safe": bool,
                "violations": List[Dict],
                "action": str,  # "allow", "block", "warn", "modify"
                "modified_text": Optional[str],
            }
        """
        violations = []
        action = "allow"
        modified_text = text
        
        # Check all rules
        for rule in self._rules.values():
            result = rule.check(text)
            if result:
                violations.append(result)
                
                # Determine action based on severity
                if rule.action == "block" or rule.severity == "critical":
                    action = "block"
                elif action != "block" and (rule.action == "warn" or rule.severity == "high"):
                    action = "warn"
                elif action == "allow" and rule.action == "modify":
                    action = "modify"
                    # Apply modification (simple example - remove match)
                    modified_text = rule.pattern.sub("[REDACTED]", modified_text)
        
        # Run custom validators
        for validator in self._custom_validators:
            try:
                if asyncio.iscoroutinefunction(validator):
                    result = await validator(text, context or {})
                else:
                    result = validator(text, context or {})
                
                if result and not result.get("safe", True):
                    violations.append({
                        "rule_id": "custom",
                        "name": "Custom Validator",
                        "action": result.get("action", "warn"),
                        "severity": result.get("severity", "medium"),
                        "description": result.get("description", "Custom validation failed"),
                    })
                    if result.get("action") == "block":
                        action = "block"
            except Exception as e:
                self.logger.warning(f"Custom validator error: {e}")
        
        # Log violations
        if violations:
            await self._log_violations(violations, text, action)
        
        return {
            "safe": len(violations) == 0 or action != "block",
            "violations": violations,
            "action": action,
            "modified_text": modified_text if action == "modify" else text,
        }
    
    async def _log_violations(
        self,
        violations: List[Dict[str, Any]],
        content: str,
        action_taken: str,
    ):
        """Log guardrail violations"""
        postgres = await get_postgres_manager()
        
        for violation in violations:
            violation_id = f"{violation['rule_id']}_{datetime.utcnow().isoformat()}"
            await postgres.execute("""
                INSERT INTO cyrex.guardrail_violations (violation_id, rule_id, content, action_taken, severity, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, violation_id, violation['rule_id'], content[:1000], action_taken,
                violation['severity'], datetime.utcnow())
    
    def add_validator(self, validator: Callable):
        """Add a custom validator function"""
        self._custom_validators.append(validator)
        self.logger.debug("Custom validator added")


# Global guardrails instance
_enhanced_guardrails: Optional[EnhancedGuardrails] = None


async def get_enhanced_guardrails() -> EnhancedGuardrails:
    """Get or create enhanced guardrails singleton"""
    global _enhanced_guardrails
    if _enhanced_guardrails is None:
        _enhanced_guardrails = EnhancedGuardrails()
        await _enhanced_guardrails.initialize()
    return _enhanced_guardrails

