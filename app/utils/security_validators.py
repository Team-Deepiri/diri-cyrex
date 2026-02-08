"""
Password and secret validation framework for Deepiri platform.
Environment-aware validation: development allows weak passwords,
production enforces strong passwords with complexity requirements.
"""
import os
import re
import secrets
import string
from enum import Enum
from typing import Optional


class EnvironmentType(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


WEAK_PASSWORDS = frozenset({
    "password", "admin", "deepiripassword", "redispassword",
    "minioadmin", "adminpassword", "your_jwt_secret_key_here",
    "your-influxdb-token", "change-me", "default", "123456",
    "postgres", "root", "test", "dev", "development",
    "secret", "pass", "qwerty", "letmein", "welcome",
    "default-secret-change-in-production",
})


def detect_environment() -> EnvironmentType:
    """Detect current environment from NODE_ENV or ENVIRONMENT variable."""
    env = os.getenv("NODE_ENV", os.getenv("ENVIRONMENT", "development")).lower()
    if env in ("production", "prod"):
        return EnvironmentType.PRODUCTION
    if env in ("staging", "stage"):
        return EnvironmentType.STAGING
    return EnvironmentType.DEVELOPMENT


class PasswordValidator:
    """Environment-aware password validator."""

    def __init__(self, environment: Optional[EnvironmentType] = None):
        self.environment = environment or detect_environment()

    def validate(self, password: str, field_name: str = "password") -> str:
        """
        Validate password strength based on environment.
        Returns the password if valid, raises ValueError otherwise.
        """
        if self.environment == EnvironmentType.DEVELOPMENT:
            return password

        if not password:
            raise ValueError(
                f"{field_name}: Password is required in {self.environment.value} environment"
            )

        if password.lower() in WEAK_PASSWORDS:
            raise ValueError(
                f"{field_name}: '{password}' is a known weak password. "
                f"Generate a secure password with: openssl rand -base64 32"
            )

        min_length = 12
        if len(password) < min_length:
            raise ValueError(
                f"{field_name}: Password must be at least {min_length} characters "
                f"in {self.environment.value} environment (got {len(password)})"
            )

        if self.environment == EnvironmentType.PRODUCTION:
            self._validate_complexity(password, field_name)

        return password

    def _validate_complexity(self, password: str, field_name: str) -> None:
        """Enforce complexity requirements for production."""
        checks = {
            "uppercase letter": r"[A-Z]",
            "lowercase letter": r"[a-z]",
            "digit": r"\d",
            "special character": r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>/?`~]",
        }
        missing = [name for name, pattern in checks.items() if not re.search(pattern, password)]
        if missing:
            raise ValueError(
                f"{field_name}: Production password must contain at least one "
                f"{', '.join(missing)}. Generate with: openssl rand -base64 32"
            )

    def validate_jwt_secret(self, secret: str, field_name: str = "JWT_SECRET") -> str:
        """Validate JWT secret - must be at least 32 characters in non-dev environments."""
        if self.environment == EnvironmentType.DEVELOPMENT:
            return secret

        if not secret:
            raise ValueError(
                f"{field_name}: JWT secret is required in {self.environment.value} environment"
            )

        if secret.lower() in WEAK_PASSWORDS:
            raise ValueError(
                f"{field_name}: '{secret}' is a known weak secret. "
                f"Generate with: openssl rand -base64 48"
            )

        if len(secret) < 32:
            raise ValueError(
                f"{field_name}: JWT secret must be at least 32 characters "
                f"in {self.environment.value} environment (got {len(secret)})"
            )

        return secret


def generate_secure_password(length: int = 32) -> str:
    """Generate a cryptographically secure password."""
    alphabet = string.ascii_letters + string.digits + string.punctuation
    while True:
        password = "".join(secrets.choice(alphabet) for _ in range(length))
        if (
            any(c.isupper() for c in password)
            and any(c.islower() for c in password)
            and any(c.isdigit() for c in password)
            and any(c in string.punctuation for c in password)
        ):
            return password


def generate_jwt_secret(length: int = 48) -> str:
    """Generate a secure JWT secret (URL-safe base64)."""
    return secrets.token_urlsafe(length)


def generate_api_token(length: int = 32) -> str:
    """Generate a secure API token."""
    return secrets.token_urlsafe(length)
