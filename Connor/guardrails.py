import re
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any, Dict

class SafetyLevel(str, Enum):
    SAFE = "safe"
    WARNING = "warning"
    BLOCKED = "blocked"
    CRITICAL = "critical"

@dataclass
class SafetyCheckResult:
    level: SafetyLevel
    message: str
    score: float
    details: Dict[str, Any] = None

class SafetyGuardrails:
    def __init__(self):
        self.secrets = self.load_secrets()
        self.credentials = self.load_credentials()
        self.exploits = self.load_exploits()

    def load_secrets(self):
        return [
            r"sk-[a-zA-Z0-9]{20,}",
            r"AKIA[0-9A-Z]{16}",
            r"AIza[0-9A-Za-z\-_]{35}",
            r"xox[baprs]-[0-9a-zA-Z-]{10,}",
            r"-----BEGIN (RSA|EC|DSA) PRIVATE KEY-----",
        ]

    def load_credentials(self):
        return [
            r"password\s*=\s*['\"].+['\"]",
            r"pwd\s*=",
            r"username\s*=\s*['\"].+['\"]",
        ]

    def load_exploits(self):
        return [
            r"rm\s+-rf\s+/",
            r";\s*shutdown",
            r"\|\s*nc\s+",
            r"\bexec\b",
            r"\beval\b",
            r"\bsubprocess\b",
        ]

    def _check_patterns(self, prompt: str, patterns, message_prefix: str) -> SafetyCheckResult:
        matches = [p for p in patterns if re.search(p, prompt)]
        if matches:
            return SafetyCheckResult(
                level=SafetyLevel.BLOCKED,
                message=f"{message_prefix} found: {len(matches)}",
                score=min(0.9, 0.5 + len(matches) * 0.1),
                details={"matches": matches},
            )
        return SafetyCheckResult(level=SafetyLevel.SAFE, message="No issues", score=0.0)

    async def check_prompt(self, prompt: str, user_id: Optional[str] = None) -> str:
        # Run all checks
        results = [
            self._check_patterns(prompt, self.exploits, "Exploit"),
            self._check_patterns(prompt, self.secrets, "Secret"),
            self._check_patterns(prompt, self.credentials, "Credential"),
        ]

        max_score = max(r.score for r in results)
        if max_score >= 0.8:
            level = SafetyLevel.CRITICAL
        elif max_score >= 0.6:
            level = SafetyLevel.BLOCKED
        elif max_score >= 0.3:
            level = SafetyLevel.WARNING
        else:
            level = SafetyLevel.SAFE

        if level in [SafetyLevel.BLOCKED, SafetyLevel.CRITICAL]:
            raise ValueError(f"Prompt blocked: {level.value}")

        return prompt  # safe to continue

def get_guardrails_two() -> SafetyGuardrails:
    return SafetyGuardrails()
