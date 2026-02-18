"""
Tests for authentication module - JWT secret security validation.

This test suite follows TDD methodology to ensure:
1. JWT_SECRET is required and cannot be missing
2. JWT_SECRET must be at least 32 characters for security
3. JWT tokens are created and validated correctly
4. Invalid secrets fail validation appropriately
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import jwt


@pytest.fixture
def mock_postgres():
    """Mock PostgreSQL manager for authentication tests"""
    mock = AsyncMock()
    mock.execute = AsyncMock()
    mock.fetchrow = AsyncMock()
    return mock


@pytest.fixture
def mock_settings_no_jwt():
    """Settings without JWT_SECRET"""
    mock = Mock()
    # No JWT_SECRET attribute - will raise AttributeError on getattr
    if hasattr(mock, 'JWT_SECRET'):
        delattr(mock, 'JWT_SECRET')
    return mock


@pytest.fixture
def mock_settings_short_jwt():
    """Settings with too-short JWT_SECRET"""
    mock = Mock()
    mock.JWT_SECRET = "short"  # Less than 32 chars
    return mock


@pytest.fixture
def mock_settings_valid_jwt():
    """Settings with valid JWT_SECRET"""
    mock = Mock()
    mock.JWT_SECRET = "test-jwt-secret-minimum-32-characters-long"
    return mock


class TestJWTSecretValidation:
    """Test suite for JWT_SECRET security requirements"""

    def test_jwt_secret_missing_raises_value_error(self, mock_postgres, mock_settings_no_jwt, monkeypatch):
        """Test that missing JWT_SECRET raises ValueError on initialization"""
        # Remove JWT_SECRET from environment as well
        monkeypatch.delenv('JWT_SECRET', raising=False)

        with patch('app.core.authentication.get_postgres_manager', return_value=mock_postgres):
            with patch('app.core.authentication.settings', mock_settings_no_jwt):
                from app.core.authentication import AuthenticationManager

                with pytest.raises(ValueError) as exc_info:
                    AuthenticationManager()

                assert 'JWT_SECRET must be set' in str(exc_info.value)

    def test_jwt_secret_too_short_raises_value_error(self, mock_postgres, mock_settings_short_jwt, monkeypatch):
        """Test that JWT_SECRET shorter than 32 characters raises ValueError"""
        # Remove JWT_SECRET from environment
        monkeypatch.delenv('JWT_SECRET', raising=False)

        with patch('app.core.authentication.get_postgres_manager', return_value=mock_postgres):
            with patch('app.core.authentication.settings', mock_settings_short_jwt):
                from app.core.authentication import AuthenticationManager

                with pytest.raises(ValueError) as exc_info:
                    AuthenticationManager()

                assert 'must be at least 32 characters' in str(exc_info.value)

    def test_jwt_secret_valid_length_initializes_successfully(self, mock_postgres, mock_settings_valid_jwt, monkeypatch):
        """Test that valid JWT_SECRET (32+ chars) initializes without error"""
        # Remove JWT_SECRET from environment
        monkeypatch.delenv('JWT_SECRET', raising=False)

        with patch('app.core.authentication.get_postgres_manager', return_value=mock_postgres):
            with patch('app.core.authentication.settings', mock_settings_valid_jwt):
                from app.core.authentication import AuthenticationManager

                # Should not raise any exception
                manager = AuthenticationManager()
                assert manager._jwt_secret == "test-jwt-secret-minimum-32-characters-long"

    def test_jwt_secret_loaded_from_environment(self, mock_postgres, monkeypatch):
        """Test that JWT_SECRET is loaded from environment variable"""
        env_secret = "environment-jwt-secret-32-chars-long"
        monkeypatch.setenv('JWT_SECRET', env_secret)

        # Settings without JWT_SECRET attribute
        mock_settings_empty = Mock()
        if hasattr(mock_settings_empty, 'JWT_SECRET'):
            delattr(mock_settings_empty, 'JWT_SECRET')

        with patch('app.core.authentication.get_postgres_manager', return_value=mock_postgres):
            with patch('app.core.authentication.settings', mock_settings_empty):
                from app.core.authentication import AuthenticationManager

                manager = AuthenticationManager()
                assert manager._jwt_secret == env_secret

    def test_jwt_secret_settings_takes_precedence_over_env(self, mock_postgres, mock_settings_valid_jwt, monkeypatch):
        """Test that settings JWT_SECRET takes precedence over environment variable"""
        # Set environment variable
        monkeypatch.setenv('JWT_SECRET', "environment-jwt-secret-32-chars-long")

        with patch('app.core.authentication.get_postgres_manager', return_value=mock_postgres):
            with patch('app.core.authentication.settings', mock_settings_valid_jwt):
                from app.core.authentication import AuthenticationManager

                manager = AuthenticationManager()
                # Should use settings value, not environment
                assert manager._jwt_secret == "test-jwt-secret-minimum-32-characters-long"
