from pydantic_settings import BaseSettings
from pydantic import ConfigDict, field_validator
from pydantic import ValidationInfo
from typing import Optional
import os
import logging

from .utils.security_validators import PasswordValidator, detect_environment

logger = logging.getLogger("cyrex.settings")


class Settings(BaseSettings):
    # API Configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    CORS_ORIGIN: str = "http://localhost:5173"
    # Default to api-gateway service name for Docker, fallback to localhost for local dev
    NODE_BACKEND_URL: str = "http://api-gateway:5000"
    CYREX_API_KEY: Optional[str] = None
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    
    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    
    # Performance Configuration
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 30
    LOCAL_LLM_TIMEOUT: int = 300  # Timeout for local LLMs (5 minutes - accommodates slower CPU inference)
    
    # AI Configuration
    AI_TEMPERATURE: float = 0.7
    AI_MAX_TOKENS: int = 2000
    AI_TOP_P: float = 0.9
    
    # Deepiri AI Model Paths
    INTENT_CLASSIFIER_MODEL_PATH: Optional[str] = None
    PRODUCTIVITY_AGENT_MODEL_PATH: Optional[str] = None
    
    # Vector Store Configuration
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    
    # LangChain Configuration
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: Optional[str] = "deepiri"
    
    # Local LLM Configuration
    LOCAL_LLM_BACKEND: str = "ollama"  # "ollama", "llama_cpp", or "transformers"
    LOCAL_LLM_MODEL: str = "llama3:8b"  # Model name/identifier
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLAMA_CPP_MODEL_PATH: Optional[str] = None  # Path to .gguf model file
    
    # PostgreSQL Configuration
    # Default to 'postgres' (Docker service name) instead of 'localhost' for containerized deployments
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "deepiri"
    POSTGRES_USER: str = "deepiri"
    POSTGRES_PASSWORD: str = "deepiripassword"

    # JWT Configuration
    JWT_SECRET: str = "default-secret-change-in-production"

    # Health Check Configuration
    HEALTH_CHECK_INTERVAL: int = 30

    @field_validator('POSTGRES_PASSWORD')
    @classmethod
    def validate_postgres_password(cls, v: str) -> str:
        validator = PasswordValidator()
        return validator.validate(v, "POSTGRES_PASSWORD")

    @field_validator('REDIS_PASSWORD')
    @classmethod
    def validate_redis_password(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            env = detect_environment()
            if env.value == "production":
                raise ValueError(
                    "REDIS_PASSWORD is required in production. "
                    "Generate with: openssl rand -base64 32"
                )
            return v
        validator = PasswordValidator()
        return validator.validate(v, "REDIS_PASSWORD")

    @field_validator('JWT_SECRET')
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        validator = PasswordValidator()
        return validator.validate_jwt_secret(v, "JWT_SECRET")

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )


# Initialize settings
settings = Settings()

# Configure logging on import
from .logging_config import configure_logging
configure_logging(
    log_level=settings.LOG_LEVEL,
    log_file=settings.LOG_FILE
)



