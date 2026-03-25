"""
Comprehensive logging configuration for the Python backend.
"""
import logging
import structlog
import sys
from pathlib import Path
from pythonjsonlogger import jsonlogger
from typing import Any, Dict
from collections import defaultdict
from threading import Lock


# Rate limiting for uvicorn access logs
_uvicorn_access_counts = defaultdict(int)
_uvicorn_access_lock = Lock()

class RateLimitedAccessLogFilter(logging.Filter):
    """Filter to rate-limit uvicorn access logs for polling endpoints."""
    
    RATE_LIMITED_PATHS = [
        "/health",
        "/metrics",
        "/orchestration/status",
        "/orchestration/health-comprehensive",
    ]
    
    def _extract_path(self, msg: str) -> str:
        """Extract path from uvicorn log message."""
        # Uvicorn format variations:
        # "IP:PORT - "METHOD PATH" STATUS"
        # "METHOD PATH HTTP/1.1" STATUS
        # Try multiple patterns to extract path
        import re
        
        # Pattern 1: Quoted format "METHOD PATH"
        match = re.search(r'"([A-Z]+)\s+([^"]+)"', msg)
        if match:
            return match.group(2).split('?')[0]  # Remove query params
        
        # Pattern 2: Unquoted format METHOD PATH HTTP/1.1
        match = re.search(r'\b(GET|POST|PUT|DELETE|PATCH|OPTIONS)\s+([^\s]+)', msg)
        if match:
            return match.group(2).split('?')[0]  # Remove query params
        
        # Pattern 3: Look for /api/agent/.../conversation pattern directly
        match = re.search(r'(/api/agent/[^/\s]+/conversation)', msg)
        if match:
            return match.group(1)
        
        # Pattern 4: Look for any path ending with /conversation
        match = re.search(r'([^\s"]+/conversation)', msg)
        if match:
            return match.group(1).split('?')[0]
        
        return ""
    
    def filter(self, record):
        """Filter log records - return False to skip logging."""
        msg = record.getMessage()
        
        # Aggressive check: if message contains "conversation" and looks like an HTTP access log, suppress it
        # This catches various uvicorn log formats
        import re
        if "conversation" in msg.lower():
            # Check if this looks like an HTTP access log (has method, path, and status)
            # Patterns: "GET /path", "POST /path", "200", "HTTP/1.1", etc.
            has_http_method = bool(re.search(r'\b(GET|POST|PUT|DELETE|PATCH|OPTIONS)\b', msg, re.IGNORECASE))
            has_status = bool(re.search(r'\b(200|201|204|400|401|403|404|500|502|503)\b', msg) or 
                            re.search(r'HTTP/1\.[01]', msg))
            
            # If it has HTTP method or status, and contains conversation, suppress it
            if (has_http_method or has_status) and re.search(r'/conversation', msg, re.IGNORECASE):
                return False
        
        path = self._extract_path(msg)
        
        if not path:
            # If we can't extract path, allow the log
            return True
        
        # Completely suppress conversation polling endpoints
        if path.endswith("/conversation"):
            return False
        
        # Check if this is a rate-limited endpoint
        is_rate_limited = any(path.startswith(limited_path) for limited_path in self.RATE_LIMITED_PATHS)
        
        if is_rate_limited:
            # Use path as key for rate limiting (more reliable than full message)
            with _uvicorn_access_lock:
                _uvicorn_access_counts[path] += 1
                count = _uvicorn_access_counts[path]
                should_log = (count % 10 == 0)
                if should_log:
                    _uvicorn_access_counts[path] = 0  # Reset counter
                return should_log
        
        # Always log non-polling endpoints
        return True


def configure_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    
    # Add rate-limiting filter to uvicorn access logger and all its handlers
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    filter_instance = RateLimitedAccessLogFilter()
    uvicorn_access_logger.addFilter(filter_instance)
    # Also apply to all existing handlers
    for handler in uvicorn_access_logger.handlers:
        handler.addFilter(filter_instance)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class RequestLogger:
    """Middleware for logging HTTP requests with structured data."""
    
    def __init__(self, logger_name: str = "cyrex.requests"):
        self.logger = get_logger(logger_name)
    
    def log_request(self, request_id: str, method: str, path: str, 
                   status_code: int, duration_ms: float, 
                   user_id: str = None, **kwargs) -> None:
        """
        Log HTTP request details.
        
        Args:
            request_id: Unique request identifier
            method: HTTP method
            path: Request path
            status_code: Response status code
            duration_ms: Request duration in milliseconds
            user_id: Optional user identifier
            **kwargs: Additional context data
        """
        self.logger.info(
            "HTTP request completed",
            request_id=request_id,
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            user_id=user_id,
            **kwargs
        )


class ErrorLogger:
    """Specialized logger for error tracking."""
    
    def __init__(self, logger_name: str = "cyrex.errors"):
        self.logger = get_logger(logger_name)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """
        Log error with context.
        
        Args:
            error: Exception instance
            context: Additional context data
        """
        self.logger.error(
            "Error occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {}
        )
    
    def log_api_error(self, error: Exception, request_id: str, 
                     endpoint: str, user_id: str = None) -> None:
        """
        Log API-specific errors.
        
        Args:
            error: Exception instance
            request_id: Request identifier
            endpoint: API endpoint
            user_id: Optional user identifier
        """
        self.logger.error(
            "API error occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            request_id=request_id,
            endpoint=endpoint,
            user_id=user_id
        )

