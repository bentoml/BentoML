"""
Tests for BentoML issue #5552 memory leak fix in readiness checks.

This test suite validates:
1. AsyncHTTPClient.is_ready() properly closes HTTP responses
2. Dependency.get() caches resolved values to avoid repeated remote proxy creation
3. Dependency.close() clears cached values for proper cleanup
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import httpx
    from _bentoml_impl.client.http import AsyncHTTPClient
    from _bentoml_sdk.service.dependency import Dependency
except ImportError:
    pytest.skip("Required modules not available", allow_module_level=True)


class TestAsyncHTTPClientFix:
    """Test AsyncHTTPClient.is_ready() response closure fix."""

    @pytest.mark.asyncio
    async def test_is_ready_closes_response_on_success(self):
        """Test that is_ready() calls resp.aclose() on successful response."""
        
        # Mock response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aclose = AsyncMock()
        
        # Mock client
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        
        # Create and configure AsyncHTTPClient
        client = AsyncHTTPClient(url="http://test.example.com")
        client.client = mock_client
        client._readyz_endpoint = "/readyz"
        
        # Call is_ready
        result = await client.is_ready(timeout=30)
        
        # Verify
        assert result is True
        mock_client.get.assert_called_once_with("/readyz", timeout=30)
        mock_response.aclose.assert_called_once()

    @pytest.mark.asyncio  
    async def test_is_ready_closes_response_on_failure(self):
        """Test that is_ready() calls resp.aclose() on failed response."""
        
        # Mock response with failure status
        mock_response = AsyncMock()
        mock_response.status_code = 503
        mock_response.aclose = AsyncMock()
        
        # Mock client
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        
        # Create and configure AsyncHTTPClient
        client = AsyncHTTPClient(url="http://test.example.com")
        client.client = mock_client
        client._readyz_endpoint = "/readyz"
        
        # Call is_ready
        result = await client.is_ready(timeout=30)
        
        # Verify
        assert result is False
        mock_client.get.assert_called_once_with("/readyz", timeout=30)
        mock_response.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_ready_handles_timeout_without_response(self):
        """Test that is_ready() handles timeout exceptions gracefully."""
        
        # Mock client that raises timeout
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.TimeoutException("timeout")
        
        # Create and configure AsyncHTTPClient  
        client = AsyncHTTPClient(url="http://test.example.com")
        client.client = mock_client
        client._readyz_endpoint = "/readyz"
        
        # Call is_ready
        result = await client.is_ready(timeout=5)
        
        # Verify
        assert result is False
        mock_client.get.assert_called_once_with("/readyz", timeout=5)
        # No aclose() should be called since no response was received


class TestDependencyCachingFix:
    """Test Dependency caching and cleanup fixes."""

    def test_dependency_get_caches_resolved_value(self):
        """Test that get() returns cached value on subsequent calls."""
        
        # Mock RemoteProxy and service
        mock_service = MagicMock()
        mock_proxy = MagicMock()
        mock_proxy.as_service.return_value = mock_service
        
        with patch('_bentoml_sdk.service.dependency.RemoteProxy', return_value=mock_proxy):
            # Create dependency
            dep = Dependency(url="http://test.example.com")
            
            # First call should create and cache
            result1 = dep.get()
            assert result1 == mock_service
            assert dep._resolved == mock_service
            
            # Second call should return cached value without creating new proxy
            result2 = dep.get()
            assert result2 == mock_service
            assert result1 is result2  # Same object reference
            
            # RemoteProxy should only be instantiated once
            from _bentoml_sdk.service.dependency import RemoteProxy
            RemoteProxy.assert_called_once()

    def test_dependency_get_caches_across_multiple_dependencies(self):
        """Test that different dependencies maintain separate caches."""
        
        mock_service1 = MagicMock()
        mock_service2 = MagicMock()
        
        with patch('_bentoml_sdk.service.dependency.RemoteProxy') as mock_proxy_cls:
            # Configure mock to return different services for different URLs
            def side_effect(*args, **kwargs):
                proxy = MagicMock()
                if 'test1' in args[0]:
                    proxy.as_service.return_value = mock_service1
                else:
                    proxy.as_service.return_value = mock_service2
                return proxy
            mock_proxy_cls.side_effect = side_effect
            
            # Create two different dependencies
            dep1 = Dependency(url="http://test1.example.com")
            dep2 = Dependency(url="http://test2.example.com")
            
            # Get services
            result1 = dep1.get()
            result2 = dep2.get()
            
            # Verify they're different and cached separately
            assert result1 == mock_service1
            assert result2 == mock_service2
            assert result1 is not result2
            assert dep1._resolved == mock_service1
            assert dep2._resolved == mock_service2

    @pytest.mark.asyncio
    async def test_dependency_close_clears_resolved(self):
        """Test that close() clears the cached _resolved value."""
        
        # Create dependency with mock resolved value
        dep = Dependency(url="http://test.example.com")
        mock_resolved = MagicMock()
        mock_resolved.close = AsyncMock()
        dep._resolved = mock_resolved
        
        # Verify initial state
        assert dep._resolved is not None
        
        # Call close
        await dep.close()
        
        # Verify cleanup
        mock_resolved.close.assert_called_once()
        assert dep._resolved is None

    @pytest.mark.asyncio
    async def test_dependency_close_handles_none_resolved(self):
        """Test that close() handles None _resolved value gracefully."""
        
        # Create dependency without resolved value
        dep = Dependency(url="http://test.example.com")
        assert dep._resolved is None
        
        # close() should not raise any errors
        await dep.close()
        assert dep._resolved is None

    @pytest.mark.asyncio
    async def test_dependency_close_skips_non_async_close(self):
        """Test that close() handles resolved objects without async close method."""
        
        # Create dependency with resolved object that has no close method
        dep = Dependency(url="http://test.example.com") 
        mock_resolved = MagicMock()
        # Don't add close method to mock
        dep._resolved = mock_resolved
        
        # close() should not raise errors and should clear _resolved
        await dep.close()
        assert dep._resolved is None


class TestMemoryLeakScenario:
    """Integration tests simulating the memory leak scenario."""

    @pytest.mark.asyncio
    async def test_repeated_readiness_checks_use_cache(self):
        """Simulate intensive readiness checks and verify caching prevents object churn."""
        
        # Track RemoteProxy creation calls
        proxy_creation_count = 0
        
        def track_proxy_creation(*args, **kwargs):
            nonlocal proxy_creation_count
            proxy_creation_count += 1
            proxy = MagicMock()
            proxy.as_service.return_value = MagicMock()
            return proxy
        
        with patch('_bentoml_sdk.service.dependency.RemoteProxy', side_effect=track_proxy_creation):
            dep = Dependency(url="http://test.example.com")
            
            # Simulate many readiness checks (like Kubernetes probes)
            services = []
            for i in range(100):
                service = dep.get()
                services.append(service)
            
            # Verify all services are the same object (cached)
            assert all(s is services[0] for s in services)
            
            # Verify RemoteProxy was only created once
            assert proxy_creation_count == 1

    @pytest.mark.asyncio
    async def test_cleanup_cycle_memory_pattern(self):
        """Test the full lifecycle: create → use → cleanup pattern."""
        
        dependencies = []
        
        # Create multiple dependencies (simulating different services)
        for i in range(10):
            dep = Dependency(url=f"http://test{i}.example.com")
            dependencies.append(dep)
        
        # Resolve all dependencies (simulate readiness checks)
        with patch('_bentoml_sdk.service.dependency.RemoteProxy') as mock_proxy_cls:
            mock_proxy_cls.return_value.as_service.return_value = MagicMock()
            
            for dep in dependencies:
                # Multiple calls to simulate readiness checks
                for _ in range(10):
                    service = dep.get()
                    assert service is not None
            
            # Verify each dependency only created one proxy
            assert mock_proxy_cls.call_count == len(dependencies)
        
        # Cleanup all dependencies
        for dep in dependencies:
            assert dep._resolved is not None
            await dep.close()
            assert dep._resolved is None


if __name__ == "__main__":
    # Run tests directly for development
    import asyncio
    
    async def run_async_tests():
        test_client = TestAsyncHTTPClientFix()
        test_dependency = TestDependencyCachingFix()
        test_scenario = TestMemoryLeakScenario()
        
        print("Running AsyncHTTPClient tests...")
        await test_client.test_is_ready_closes_response_on_success()
        await test_client.test_is_ready_closes_response_on_failure()  
        await test_client.test_is_ready_handles_timeout_without_response()
        print("✓ AsyncHTTPClient tests passed")
        
        print("Running Dependency caching tests...")
        test_dependency.test_dependency_get_caches_resolved_value()
        test_dependency.test_dependency_get_caches_across_multiple_dependencies()
        await test_dependency.test_dependency_close_clears_resolved()
        await test_dependency.test_dependency_close_handles_none_resolved()
        await test_dependency.test_dependency_close_skips_non_async_close()
        print("✓ Dependency caching tests passed")
        
        print("Running memory leak scenario tests...")
        await test_scenario.test_repeated_readiness_checks_use_cache()
        await test_scenario.test_cleanup_cycle_memory_pattern()
        print("✓ Memory leak scenario tests passed")
        
        print("\n🎉 All tests passed!")
    
    asyncio.run(run_async_tests())