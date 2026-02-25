#!/usr/bin/env python3
"""
Test script to verify the readiness check memory leak fix for issue #5552.

This script tests:
1. Async readiness response closure
2. Dependency caching to avoid repeated remote proxy creation
3. Proper cleanup in dependency close()
"""

import asyncio
import gc
import sys
import tracemalloc
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx

# Add the src directory to Python path for imports
sys.path.insert(0, "src")

from _bentoml_impl.client.http import AsyncHTTPClient
from _bentoml_sdk.service.dependency import Dependency


class TestReadinessMemoryLeakFix:
    """Test the memory leak fixes in readiness checks."""

    async def test_async_is_ready_closes_response(self):
        """Test that async is_ready properly closes the response."""

        # Mock httpx.AsyncClient and response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aclose = AsyncMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        # Create AsyncHTTPClient instance
        client = AsyncHTTPClient(url="http://test.example.com")
        client.client = mock_client
        client._readyz_endpoint = "/readyz"

        # Call is_ready
        result = await client.is_ready(timeout=30)

        # Verify response was closed
        assert result is True
        mock_client.get.assert_called_once_with("/readyz", timeout=30)
        mock_response.aclose.assert_called_once()

    async def test_async_is_ready_handles_timeout(self):
        """Test that async is_ready handles timeouts properly."""

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.TimeoutException("timeout")

        client = AsyncHTTPClient(url="http://test.example.com")
        client.client = mock_client
        client._readyz_endpoint = "/readyz"

        result = await client.is_ready(timeout=5)

        assert result is False
        mock_client.get.assert_called_once_with("/readyz", timeout=5)

    def test_dependency_caches_resolved_value(self):
        """Test that Dependency.get() caches the resolved value."""

        # Mock RemoteProxy
        mock_proxy = MagicMock()
        mock_service = MagicMock()
        mock_proxy.as_service.return_value = mock_service

        with patch(
            "_bentoml_sdk.service.dependency.RemoteProxy", return_value=mock_proxy
        ):
            dep = Dependency(url="http://test.example.com")

            # First call should create new proxy
            result1 = dep.get()
            assert result1 == mock_service
            assert dep._resolved == mock_service

            # Second call should return cached value
            result2 = dep.get()
            assert result2 == mock_service
            assert result1 is result2  # Same object

            # RemoteProxy should only be created once
            from _bentoml_sdk.service.dependency import RemoteProxy

            RemoteProxy.assert_called_once()

    async def test_dependency_close_clears_resolved(self):
        """Test that Dependency.close() clears _resolved."""

        # Create dependency with mock resolved value
        dep = Dependency(url="http://test.example.com")
        mock_resolved = MagicMock()
        mock_resolved.close = AsyncMock()
        dep._resolved = mock_resolved

        # Call close
        await dep.close()

        # Verify close was called on resolved object and _resolved was cleared
        mock_resolved.close.assert_called_once()
        assert dep._resolved is None

    async def test_dependency_close_handles_no_resolved(self):
        """Test that Dependency.close() handles None _resolved gracefully."""

        dep = Dependency(url="http://test.example.com")
        assert dep._resolved is None

        # Should not raise any errors
        await dep.close()
        assert dep._resolved is None


async def test_memory_usage_simulation():
    """Simulate the memory leak scenario and verify fix."""

    print("Starting memory usage simulation...")
    tracemalloc.start()

    # Create multiple dependency instances and repeatedly call get()
    dependencies = []
    for i in range(10):
        dep = Dependency(url=f"http://test{i}.example.com")
        dependencies.append(dep)

    # Simulate repeated readiness checks (like what happens in production)
    for cycle in range(100):
        for dep in dependencies:
            with patch("_bentoml_sdk.service.dependency.RemoteProxy") as mock_proxy_cls:
                mock_proxy = MagicMock()
                mock_service = MagicMock()
                mock_proxy.as_service.return_value = mock_service
                mock_proxy_cls.return_value = mock_proxy

                # First call creates proxy
                result1 = dep.get()

                # Second call should return cached value (not create new proxy)
                result2 = dep.get()

                # Verify proxy was only created once due to caching
                mock_proxy_cls.assert_called_once()
                assert result1 is result2

    # Cleanup
    for dep in dependencies:
        await dep.close()
        assert dep._resolved is None

    # Force garbage collection
    gc.collect()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Memory usage: current={current / 1024:.2f} KB, peak={peak / 1024:.2f} KB")
    print("Memory simulation completed successfully")


if __name__ == "__main__":
    import asyncio

    print("Testing BentoML readiness check memory leak fix...")

    # Run the test
    async def run_tests():
        test_instance = TestReadinessMemoryLeakFix()

        print("1. Testing async is_ready response closure...")
        await test_instance.test_async_is_ready_closes_response()
        print("✓ PASS")

        print("2. Testing async is_ready timeout handling...")
        await test_instance.test_async_is_ready_handles_timeout()
        print("✓ PASS")

        print("3. Testing dependency caching...")
        test_instance.test_dependency_caches_resolved_value()
        print("✓ PASS")

        print("4. Testing dependency close...")
        await test_instance.test_dependency_close_clears_resolved()
        print("✓ PASS")

        print("5. Testing dependency close with None resolved...")
        await test_instance.test_dependency_close_handles_no_resolved()
        print("✓ PASS")

        print("6. Running memory usage simulation...")
        await test_memory_usage_simulation()
        print("✓ PASS")

        print("\nAll tests passed! 🎉")
        print("The memory leak fix is working correctly.")

    asyncio.run(run_tests())
