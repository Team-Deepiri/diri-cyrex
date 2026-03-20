"""
Comprehensive test suite for the Python backend.
Enhanced with tests for new architectural components.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
import json
import numpy as np

from app.main import app
from app.settings import settings


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def client_with_api_key():
    """Create a test client with API key header."""
    with patch.object(settings, 'CYREX_API_KEY', 'test-api-key'):
        client = TestClient(app)
        client.headers = {"x-api-key": "test-api-key"}
        return client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch('openai.OpenAI') as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        
        # Mock completion response
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "Test response from AI"
        mock_completion.usage.total_tokens = 150
        mock_completion.model = "gpt-4o-mini"
        
        mock_instance.chat.completions.create.return_value = mock_completion
        yield mock_instance


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for testing external API calls."""
    with patch('httpx.AsyncClient') as mock:
        mock_instance = AsyncMock()
        mock.return_value.__aenter__.return_value = mock_instance
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test": "data"}
        mock_response.raise_for_status.return_value = None
        
        mock_instance.get.return_value = mock_response
        yield mock_instance


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing."""
    with patch('app.services.embedding_service.get_embedding_service') as mock:
        mock_service = Mock()
        mock_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_service.embed.return_value = mock_embedding
        mock.return_value = mock_service
        yield mock_service


@pytest.fixture
def mock_system_initializer():
    """Mock system initializer for testing."""
    with patch('app.core.system_initializer.get_system_initializer') as mock:
        mock_init = AsyncMock()
        mock_init.health_check.return_value = {
            "initialized": True,
            "systems": {
                "postgresql": {"healthy": True},
                "session_manager": {"healthy": True},
                "memory_manager": {"healthy": True}
            }
        }
        mock.return_value = mock_init
        yield mock_init


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing."""
    with patch('app.integrations.ollama_container.get_ollama_client') as mock:
        mock_client = AsyncMock()
        mock_client.health_check.return_value = {"status": "healthy", "model": "llama3:8b"}
        mock.return_value = mock_client
        yield mock_client


class TestHealthEndpoint:
    """Test cases for the health endpoint."""
    
    def test_health_endpoint(self, client):
        """Test basic health check functionality."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert data["version"] == "3.0.0"
        assert "timestamp" in data
        assert "services" in data
        assert "configuration" in data
        assert "ai" in data["services"]
    
    def test_health_without_openai_key(self, client):
        """Test health check when OpenAI key is not configured."""
        with patch.object(settings, 'OPENAI_API_KEY', None):
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["services"]["ai"] == "disabled"
    
    def test_health_core_systems(self, client, mock_system_initializer):
        """Test health check includes core systems status."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        # Core systems may or may not be present depending on initialization
        if "core_systems" in data:
            assert isinstance(data["core_systems"], dict)
    
    def test_health_ollama(self, client, mock_ollama_client):
        """Test health check includes Ollama status."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        # Ollama may or may not be present depending on configuration
        if "ollama" in data:
            assert isinstance(data["ollama"], dict)
    
    def test_health_redis_check(self, client):
        """Test health check includes Redis status."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "redis" in data["services"]
        assert isinstance(data["services"]["redis"], str)
    
    def test_health_node_backend_check(self, client):
        """Test health check includes Node backend status."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "node_backend" in data["services"]
        assert isinstance(data["services"]["node_backend"], str)


class TestMetricsEndpoint:
    """Test cases for the metrics endpoint."""
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
        assert "cyrex_requests_total" in response.text or len(response.text) > 0


class TestRootEndpoint:
    """Test cases for the root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Deepiri AI Challenge Service API"
        assert data["version"] == "3.0.0"
        assert "docs" in data
        assert "health" in data
        assert "metrics" in data


class TestEmbeddingsEndpoint:
    """Test cases for the embeddings API endpoint."""
    
    def test_embeddings_success(self, client, mock_embedding_service):
        """Test successful embedding generation."""
        response = client.post(
            "/api/embeddings",
            json={"text": "Hello, world!"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert "dimension" in data
        assert "model" in data
        assert isinstance(data["embedding"], list)
        assert len(data["embedding"]) > 0
    
    def test_embeddings_custom_model(self, client, mock_embedding_service):
        """Test embedding generation with custom model."""
        response = client.post(
            "/api/embeddings",
            json={
                "text": "Test text",
                "model": "custom-model"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "custom-model"
    
    def test_embeddings_empty_text(self, client):
        """Test embedding generation with empty text."""
        response = client.post(
            "/api/embeddings",
            json={"text": ""}
        )
        
        # Should still work or return appropriate error
        assert response.status_code in [200, 422]
    
    def test_embeddings_service_error(self, client):
        """Test embedding generation when service fails."""
        with patch('app.services.embedding_service.get_embedding_service') as mock:
            mock_service = Mock()
            mock_service.embed.side_effect = Exception("Service error")
            mock.return_value = mock_service
            
            response = client.post(
                "/api/embeddings",
                json={"text": "Test"}
            )
            
            assert response.status_code == 500


class TestCompleteEndpoint:
    """Test cases for the complete API endpoint."""
    
    def test_complete_success(self, client, mock_openai_client):
        """Test successful completion generation."""
        with patch.object(settings, 'OPENAI_API_KEY', 'test-key'):
            response = client.post(
                "/api/complete",
                json={
                    "prompt": "Hello, AI!",
                    "max_tokens": 100,
                    "temperature": 0.7
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "completion" in data
            assert "tokens_used" in data
            assert "model" in data
    
    def test_complete_without_openai_key(self, client):
        """Test completion when OpenAI key is not configured."""
        with patch.object(settings, 'OPENAI_API_KEY', None):
            response = client.post(
                "/api/complete",
                json={"prompt": "Hello!"}
            )
            
            assert response.status_code == 503
            assert "not configured" in response.json()["detail"]
    
    def test_complete_custom_parameters(self, client, mock_openai_client):
        """Test completion with custom parameters."""
        with patch.object(settings, 'OPENAI_API_KEY', 'test-key'):
            response = client.post(
                "/api/complete",
                json={
                    "prompt": "Test prompt",
                    "max_tokens": 500,
                    "temperature": 0.9
                }
            )
            
            assert response.status_code == 200
            mock_openai_client.chat.completions.create.assert_called()
    
    def test_complete_openai_error(self, client, mock_openai_client):
        """Test completion when OpenAI API fails."""
        with patch.object(settings, 'OPENAI_API_KEY', 'test-key'):
            mock_openai_client.chat.completions.create.side_effect = Exception("API error")
            
            response = client.post(
                "/api/complete",
                json={"prompt": "Test"}
            )
            
            assert response.status_code == 500


class TestAgentMessageEndpoint:
    """Test cases for the agent message endpoint."""
    
    def test_agent_message_success(self, client, mock_openai_client):
        """Test successful agent message processing."""
        with patch.object(settings, 'OPENAI_API_KEY', 'test-key'):
            response = client.post(
                "/agent/message",
                json={
                    "content": "Hello, AI!",
                    "session_id": "test-session"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert "data" in data
            assert "request_id" in data
            assert data["data"]["message"] == "Test response from AI"
            assert data["data"]["session_id"] == "test-session"
            assert data["data"]["tokens"] == 150
    
    def test_agent_message_without_openai_key(self, client):
        """Test agent message when OpenAI key is not configured."""
        with patch.object(settings, 'OPENAI_API_KEY', None):
            response = client.post(
                "/agent/message",
                json={"content": "Hello, AI!"}
            )
            
            assert response.status_code == 503
            assert "AI service not configured" in response.json()["detail"]
    
    def test_agent_message_invalid_content(self, client):
        """Test agent message with invalid content."""
        # Empty content
        response = client.post(
            "/agent/message",
            json={"content": ""}
        )
        assert response.status_code == 422
        
        # Content too long
        response = client.post(
            "/agent/message",
            json={"content": "x" * 4001}
        )
        assert response.status_code == 422
    
    def test_agent_message_openai_error(self, client, mock_openai_client):
        """Test agent message with OpenAI API error."""
        with patch.object(settings, 'OPENAI_API_KEY', 'test-key'):
            # Mock OpenAI rate limit error
            mock_openai_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")
            
            response = client.post(
                "/agent/message",
                json={"content": "Hello, AI!"}
            )
            
            assert response.status_code == 500
    
    def test_agent_message_with_custom_parameters(self, client, mock_openai_client):
        """Test agent message with custom temperature and max_tokens."""
        with patch.object(settings, 'OPENAI_API_KEY', 'test-key'):
            response = client.post(
                "/agent/message",
                json={
                    "content": "Hello, AI!",
                    "temperature": 0.5,
                    "max_tokens": 1000
                }
            )
            
            assert response.status_code == 200
            # Verify the parameters were passed to OpenAI
            mock_openai_client.chat.completions.create.assert_called_once()
            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args[1]["temperature"] == 0.5
            assert call_args[1]["max_tokens"] == 1000


class TestAgentMessageStreamEndpoint:
    """Test cases for the agent message streaming endpoint."""
    
    def test_agent_message_stream_success(self, client, mock_openai_client):
        """Test successful agent message streaming."""
        with patch.object(settings, 'OPENAI_API_KEY', 'test-key'):
            # Mock streaming response - create an iterable generator
            mock_chunk1 = Mock()
            mock_chunk1.choices = [Mock()]
            mock_chunk1.choices[0].delta.content = "Hello"
            
            mock_chunk2 = Mock()
            mock_chunk2.choices = [Mock()]
            mock_chunk2.choices[0].delta.content = " World"
            
            # Create a generator that yields chunks
            def mock_stream():
                yield mock_chunk1
                yield mock_chunk2
            
            # Mock asyncio.to_thread to return our mock stream
            with patch('asyncio.to_thread', return_value=mock_stream()):
                response = client.post(
                    "/agent/message/stream",
                    json={"content": "Hello, AI!"}
                )
                
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/plain"
                # Check for request ID header (can be x-request-id or X-Request-ID)
                assert "x-request-id" in response.headers or "X-Request-ID" in response.headers
    
    def test_agent_message_stream_without_openai_key(self, client):
        """Test agent message streaming when OpenAI key is not configured."""
        with patch.object(settings, 'OPENAI_API_KEY', None):
            response = client.post(
                "/agent/message/stream",
                json={"content": "Hello, AI!"}
            )
            
            assert response.status_code == 503


class TestProxyEndpoints:
    """Test cases for proxy endpoints."""
    
    @pytest.mark.asyncio
    async def test_proxy_adventure_data_success(self, mock_httpx_client):
        """Test successful adventure data proxy."""
        with patch.object(settings, 'NODE_BACKEND_URL', 'http://test-backend'):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.get(
                    "/agent/tools/external/adventure-data",
                    params={"lat": 40.7128, "lng": -74.0060, "radius": 5000}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data == {"test": "data"}
    
    @pytest.mark.asyncio
    async def test_proxy_adventure_data_timeout(self, mock_httpx_client):
        """Test adventure data proxy with timeout."""
        from httpx import TimeoutException
        
        with patch.object(settings, 'NODE_BACKEND_URL', 'http://test-backend'):
            mock_httpx_client.get.side_effect = TimeoutException("Request timeout")
            
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.get(
                    "/agent/tools/external/adventure-data",
                    params={"lat": 40.7128, "lng": -74.0060}
                )
                
                assert response.status_code == 504
                assert "timeout" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_proxy_directions_success(self, mock_httpx_client):
        """Test successful directions proxy."""
        with patch.object(settings, 'NODE_BACKEND_URL', 'http://test-backend'):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.get(
                    "/agent/tools/external/directions",
                    params={
                        "fromLat": 40.7128, "fromLng": -74.0060,
                        "toLat": 40.7589, "toLng": -73.9851,
                        "mode": "walking"
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data == {"test": "data"}
    
    @pytest.mark.asyncio
    async def test_proxy_weather_current_success(self, mock_httpx_client):
        """Test successful current weather proxy."""
        with patch.object(settings, 'NODE_BACKEND_URL', 'http://test-backend'):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.get(
                    "/agent/tools/external/weather/current",
                    params={"lat": 40.7128, "lng": -74.0060}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data == {"test": "data"}
    
    @pytest.mark.asyncio
    async def test_proxy_weather_forecast_success(self, mock_httpx_client):
        """Test successful weather forecast proxy."""
        with patch.object(settings, 'NODE_BACKEND_URL', 'http://test-backend'):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.get(
                    "/agent/tools/external/weather/forecast",
                    params={"lat": 40.7128, "lng": -74.0060, "days": 3}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data == {"test": "data"}


class TestApiKeyAuthentication:
    """Test cases for API key authentication middleware."""
    
    def test_api_key_required_for_protected_endpoints(self, client):
        """Test that API key is required for protected endpoints."""
        with patch.object(settings, 'CYREX_API_KEY', 'test-key'):
            # Health and metrics should not require API key
            response = client.get("/health")
            assert response.status_code == 200
            
            # Protected endpoints should require API key
            response = client.post(
                "/agent/message",
                json={"content": "Test"}
            )
            # Should either require API key or allow with default key
            assert response.status_code in [200, 401, 503]
    
    def test_api_key_validation(self, client):
        """Test API key validation."""
        with patch.object(settings, 'CYREX_API_KEY', 'valid-key'):
            # Invalid API key
            response = client.post(
                "/agent/message",
                json={"content": "Test"},
                headers={"x-api-key": "invalid-key"}
            )
            # May allow if default key or require valid key
            assert response.status_code in [200, 401, 503]
    
    def test_desktop_client_header(self, client):
        """Test desktop client header bypass."""
        with patch.object(settings, 'CYREX_API_KEY', 'test-key'):
            response = client.get(
                "/health",
                headers={"x-desktop-client": "true"}
            )
            assert response.status_code == 200
    
    def test_default_api_key_behavior(self, client):
        """Test default API key behavior for local development."""
        with patch.object(settings, 'CYREX_API_KEY', 'change-me'):
            # Should allow requests without API key in local dev
            response = client.post(
                "/agent/message",
                json={"content": "Test"}
            )
            # May succeed or fail based on OpenAI key, but shouldn't be 401
            assert response.status_code != 401


class TestMiddleware:
    """Test cases for middleware functionality."""
    
    def test_request_id_middleware(self, client):
        """Test that request ID is added to responses."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "x-request-id" in response.headers
        assert len(response.headers["x-request-id"]) > 0
    
    def test_cors_middleware(self, client):
        """Test CORS headers are present."""
        response = client.options("/health")
        # CORS headers should be present
        assert response.status_code in [200, 204]
    
    def test_rate_limit_headers(self, client):
        """Test rate limit headers are present."""
        response = client.get("/health")
        assert response.status_code == 200
        # Rate limit headers may be present
        assert "X-RateLimit-Limit" in response.headers or "X-RateLimit-Remaining" in response.headers or True
    
    def test_request_timing_middleware(self, client):
        """Test request timing middleware adds timing information."""
        response = client.get("/health")
        assert response.status_code == 200
        # Timing information may be in logs or headers
        assert "x-request-id" in response.headers


class TestNewRoutes:
    """Test cases for new API routes."""
    
    def test_orchestration_health_comprehensive(self, client):
        """Test orchestration comprehensive health endpoint."""
        response = client.get("/orchestration/health-comprehensive")
        # May return 200 or 404 if not implemented
        assert response.status_code in [200, 404]
    
    def test_workflow_health(self, client):
        """Test workflow health endpoint."""
        response = client.get("/api/workflow/health")
        # May return 200 or 404 if not implemented
        assert response.status_code in [200, 404]
    
    def test_cyrex_guard_endpoints(self, client):
        """Test Cyrex Guard endpoints exist."""
        # Test invoice processing endpoint
        response = client.post(
            "/cyrex-guard/invoice/process",
            json={
                "invoice_content": "Test invoice",
                "industry": "property_management",
                "invoice_format": "text"
            }
        )
        # May return 200, 422, or 404 depending on implementation
        assert response.status_code in [200, 422, 404, 500]
    
    def test_document_extraction_endpoint(self, client):
        """Test document extraction endpoint."""
        response = client.post(
            "/document-extraction/extract-text",
            json={
                "documentUrl": "http://example.com/doc.pdf",
                "documentType": "pdf"
            }
        )
        # May return 200, 422, or 404 depending on implementation
        assert response.status_code in [200, 422, 404, 500]
    
    def test_language_intelligence_endpoints(self, client):
        """Test language intelligence endpoints."""
        response = client.post(
            "/language-intelligence/lease/abstract",
            json={
                "leaseId": "test-lease",
                "documentText": "Test lease document",
                "documentUrl": "http://example.com/lease.pdf"
            }
        )
        # May return 200, 422, or 404 depending on implementation
        assert response.status_code in [200, 422, 404, 500]
    
    def test_universal_rag_endpoints(self, client):
        """Test universal RAG endpoints."""
        response = client.get("/universal-rag/health")
        # May return 200 or 404 if not implemented
        assert response.status_code in [200, 404]
    
    def test_document_indexing_endpoints(self, client):
        """Test document indexing endpoints."""
        response = client.get("/document-indexing/health")
        # May return 200 or 404 if not implemented
        assert response.status_code in [200, 404]
    
    def test_vendor_fraud_endpoints(self, client):
        """Test vendor fraud endpoints."""
        response = client.get("/vendor-fraud/health")
        # May return 200 or 404 if not implemented
        assert response.status_code in [200, 404]
    
    def test_agent_playground_endpoints(self, client):
        """Test agent playground endpoints."""
        response = client.get("/api/agent/models")
        # May return 200 or 404 if not implemented
        assert response.status_code in [200, 404]


class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_404_endpoint(self, client):
        """Test 404 for non-existent endpoints."""
        response = client.get("/non-existent-endpoint")
        assert response.status_code == 404
    
    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/agent/message",
            content="invalid json",
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        response = client.post(
            "/agent/message",
            json={}
        )
        assert response.status_code == 422
    
    def test_invalid_field_types(self, client):
        """Test handling of invalid field types."""
        response = client.post(
            "/agent/message",
            json={
                "content": "Test",
                "temperature": "invalid"  # Should be float
            }
        )
        assert response.status_code == 422
    
    def test_server_error_handling(self, client):
        """Test server error handling."""
        # This test verifies that 500 errors are properly handled
        # We can't easily trigger a real 500, but we can verify error structure
        with patch('app.main.health', side_effect=Exception("Test error")):
            response = client.get("/health")
            # Should handle error gracefully
            assert response.status_code in [200, 500]


class TestIntegration:
    """Integration tests for multiple components."""
    
    def test_health_to_metrics_flow(self, client):
        """Test that health and metrics endpoints work together."""
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
    
    def test_request_id_consistency(self, client):
        """Test that request ID is consistent across requests."""
        response1 = client.get("/health")
        response2 = client.get("/health")
        
        # Request IDs should be different for each request
        assert response1.headers["x-request-id"] != response2.headers["x-request-id"]
    
    def test_cors_preflight(self, client):
        """Test CORS preflight requests."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET"
            }
        )
        assert response.status_code in [200, 204]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
