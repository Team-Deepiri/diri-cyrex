"""
Custom exceptions for Milvus operations.

Provides clear error handling for different Milvus failure modes.
"""


class MilvusError(Exception):
    """Base exception for Milvus operations"""
    pass


class MilvusConnectionError(MilvusError):
    """Raised when connection to Milvus fails"""
    pass


class MilvusCollectionError(MilvusError):
    """Raised when collection operations fail"""
    pass


class MilvusUnavailableError(MilvusError):
    """
    Raised when Milvus is unavailable.

    Attributes:
        using_fallback: Whether the system is using in-memory fallback
    """
    def __init__(self, message: str, using_fallback: bool = True):
        super().__init__(message)
        self.using_fallback = using_fallback
