"""
Authentication System
API key and token-based authentication for agents and API endpoints
"""
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import secrets
import jwt
from ..database.postgres import get_postgres_manager
from ..logging_config import get_logger
from ..settings import settings

logger = get_logger("cyrex.authentication")


class AuthType(str, Enum):
    """Authentication types"""
    API_KEY = "api_key"
    JWT = "jwt"
    BEARER = "bearer"
    SESSION = "session"
    NONE = "none"


@dataclass
class AuthToken:
    """Authentication token"""
    token_id: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    token_type: AuthType = AuthType.API_KEY
    expires_at: Optional[datetime] = None
    scopes: list = None
    metadata: Dict[str, Any] = None
    
    def is_expired(self) -> bool:
        """Check if token is expired"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def has_scope(self, scope: str) -> bool:
        """Check if token has a specific scope"""
        if not self.scopes:
            return True  # No scopes = all access
        return scope in self.scopes


class AuthenticationManager:
    """
    Manages authentication for agents and API endpoints
    Supports API keys, JWT tokens, and session-based auth
    """
    
    def __init__(self):
        self.logger = logger
        self._jwt_secret = getattr(settings, 'JWT_SECRET', 'default-secret-change-in-production')
        self._api_key_header = 'x-api-key'
        self._auth_header = 'authorization'
        self._initialized = False
    
    async def initialize(self):
        """Initialize authentication tables"""
        if self._initialized:
            return
        
        postgres = await get_postgres_manager()
        await postgres.execute("""
            CREATE TABLE IF NOT EXISTS auth_tokens (
                token_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255),
                agent_id VARCHAR(255),
                token_type VARCHAR(50) NOT NULL,
                token_hash VARCHAR(255) NOT NULL,
                expires_at TIMESTAMP,
                scopes JSONB DEFAULT '[]'::jsonb,
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            );
            CREATE INDEX IF NOT EXISTS idx_auth_tokens_user_id ON auth_tokens(user_id);
            CREATE INDEX IF NOT EXISTS idx_auth_tokens_agent_id ON auth_tokens(agent_id);
            CREATE INDEX IF NOT EXISTS idx_auth_tokens_hash ON auth_tokens(token_hash);
            CREATE INDEX IF NOT EXISTS idx_auth_tokens_active ON auth_tokens(is_active);
        """)
        
        self._initialized = True
        logger.info("Authentication manager initialized")
    
    async def create_api_key(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        scopes: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new API key
        
        Returns:
            The API key (plain text - store securely)
        """
        await self.initialize()
        
        # Generate API key
        api_key = secrets.token_urlsafe(32)
        token_hash = self._hash_token(api_key)
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # Store in database
        postgres = await get_postgres_manager()
        token_id = secrets.token_urlsafe(16)
        
        await postgres.execute("""
            INSERT INTO auth_tokens (token_id, user_id, agent_id, token_type, token_hash, expires_at, scopes, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """, token_id, user_id, agent_id, AuthType.API_KEY.value, token_hash, expires_at, scopes or [], metadata or {})
        
        logger.info(f"Created API key: {token_id}", user_id=user_id, agent_id=agent_id)
        return api_key
    
    async def validate_api_key(self, api_key: str) -> Optional[AuthToken]:
        """Validate an API key"""
        await self.initialize()
        
        token_hash = self._hash_token(api_key)
        
        postgres = await get_postgres_manager()
        row = await postgres.fetchrow("""
            SELECT * FROM auth_tokens
            WHERE token_hash = $1 AND is_active = TRUE
        """, token_hash)
        
        if not row:
            return None
        
        # Check expiration
        if row['expires_at'] and datetime.utcnow() > row['expires_at']:
            return None
        
        # Update last used
        await postgres.execute("""
            UPDATE auth_tokens SET last_used_at = CURRENT_TIMESTAMP WHERE token_id = $1
        """, row['token_id'])
        
        return AuthToken(
            token_id=row['token_id'],
            user_id=row['user_id'],
            agent_id=row['agent_id'],
            token_type=AuthType(row['token_type']),
            expires_at=row['expires_at'],
            scopes=row['scopes'] or [],
            metadata=row['metadata'] or {},
        )
    
    async def create_jwt_token(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        expires_in_hours: int = 24,
        scopes: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a JWT token"""
        payload = {
            "user_id": user_id,
            "agent_id": agent_id,
            "scopes": scopes or [],
            "exp": datetime.utcnow() + timedelta(hours=expires_in_hours),
            "iat": datetime.utcnow(),
        }
        
        token = jwt.encode(payload, self._jwt_secret, algorithm="HS256")
        logger.info(f"Created JWT token for user: {user_id}")
        return token
    
    async def validate_jwt_token(self, token: str) -> Optional[AuthToken]:
        """Validate a JWT token"""
        try:
            payload = jwt.decode(token, self._jwt_secret, algorithms=["HS256"])
            
            return AuthToken(
                token_id=payload.get("jti", "jwt"),
                user_id=payload.get("user_id"),
                agent_id=payload.get("agent_id"),
                token_type=AuthType.JWT,
                expires_at=datetime.fromtimestamp(payload.get("exp", 0)),
                scopes=payload.get("scopes", []),
                metadata=payload.get("metadata", {}),
            )
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    async def authenticate_request(
        self,
        headers: Dict[str, str],
    ) -> Optional[AuthToken]:
        """
        Authenticate a request from headers
        
        Checks:
        1. x-api-key header
        2. Authorization: Bearer <token>
        3. Authorization: Bearer <jwt>
        
        Returns:
            AuthToken if authenticated, None otherwise
        """
        # Check API key
        api_key = headers.get(self._api_key_header) or headers.get('X-API-Key')
        if api_key:
            token = await self.validate_api_key(api_key)
            if token:
                return token
        
        # Check Authorization header
        auth_header = headers.get(self._auth_header) or headers.get('Authorization')
        if auth_header:
            # Bearer token
            if auth_header.startswith('Bearer '):
                token_str = auth_header[7:]  # Remove "Bearer "
                
                # Try JWT first
                token = await self.validate_jwt_token(token_str)
                if token:
                    return token
                
                # Try as API key
                token = await self.validate_api_key(token_str)
                if token:
                    return token
        
        return None
    
    async def revoke_token(self, token_id: str) -> bool:
        """Revoke a token"""
        await self.initialize()
        
        postgres = await get_postgres_manager()
        result = await postgres.execute("""
            UPDATE auth_tokens SET is_active = FALSE WHERE token_id = $1
        """, token_id)
        
        logger.info(f"Revoked token: {token_id}")
        return True
    
    def _hash_token(self, token: str) -> str:
        """Hash a token for storage"""
        return hashlib.sha256(token.encode()).hexdigest()


# Global authentication manager
_auth_manager: Optional[AuthenticationManager] = None


async def get_authentication_manager() -> AuthenticationManager:
    """Get or create authentication manager singleton"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthenticationManager()
        await _auth_manager.initialize()
    return _auth_manager


async def require_auth(headers: Dict[str, str], required_scope: Optional[str] = None) -> AuthToken:
    """
    Require authentication for a request
    
    Raises:
        ValueError: If authentication fails
    """
    auth_manager = await get_authentication_manager()
    token = await auth_manager.authenticate_request(headers)
    
    if not token:
        raise ValueError("Authentication required")
    
    if token.is_expired():
        raise ValueError("Token expired")
    
    if required_scope and not token.has_scope(required_scope):
        raise ValueError(f"Required scope '{required_scope}' not granted")
    
    return token

