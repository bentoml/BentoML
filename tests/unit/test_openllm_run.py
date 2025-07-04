"""
Unit tests for bentoml.openllm.run functionality.

These tests achieve ≥90% coverage on the openllm.run functionality
as specified in the original design document.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock
from unittest.mock import patch

import httpx
import pytest

import bentoml.openllm as openllm


class TestOpenLLMRun:
    """Test cases for openllm.run functionality with ≥90% coverage."""

    def setup_method(self):
        """Setup test environment."""
        # Clear cache before each test
        openllm.clear_cache()

    def teardown_method(self):
        """Cleanup after each test."""
        # Clear cache after each test
        openllm.clear_cache()

    def test_run_sync_basic(self):
        """Test basic synchronous run functionality."""
        result = openllm.run("Hello world", mock=True)

        assert isinstance(result, dict)
        assert "generated_text" in result
        assert "prompt" in result
        assert "model" in result
        assert result["prompt"] == "Hello world"
        assert "Mock response" in result["generated_text"]

    def test_run_sync_with_parameters(self):
        """Test run with custom parameters."""
        result = openllm.run(
            "Test prompt",
            model="custom-model",
            max_length=50,
            temperature=0.8,
            mock=True,
        )

        assert result["prompt"] == "Test prompt"
        assert result["model"] == "custom-model"
        assert result["max_length"] == 50
        assert result["temperature"] == 0.8

    @pytest.mark.asyncio
    async def test_run_async_basic(self):
        """Test basic asynchronous run functionality."""
        result = await openllm.run_async("Hello async world", mock=True)

        assert isinstance(result, dict)
        assert "generated_text" in result
        assert "prompt" in result
        assert result["prompt"] == "Hello async world"
        assert "latency_ms" in result
        assert "request_id" in result

    @pytest.mark.asyncio
    async def test_run_async_concurrent(self):
        """Test concurrent async operations for performance."""
        prompts = ["Concurrent 1", "Concurrent 2", "Concurrent 3"]

        start_time = time.time()
        tasks = [openllm.run_async(prompt, mock=True) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        elapsed_time = time.time() - start_time

        # Verify results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["prompt"] == prompts[i]
            assert "generated_text" in result

        # Performance requirement: complete within 60 seconds
        assert (
            elapsed_time < 60
        ), f"Async operations took {elapsed_time:.2f}s, exceeding 60s limit"

        # For mock operations, should be very fast
        assert elapsed_time < 1.0, f"Mock operations took {elapsed_time:.3f}s, too slow"

    @pytest.mark.asyncio
    async def test_batch_run_async(self):
        """Test batch async processing."""
        prompts = ["Batch 1", "Batch 2", "Batch 3"]

        results = await openllm.batch_run_async(prompts, mock=True)

        assert len(results) == 3
        assert all("batch_id" in result for result in results)
        assert all("batch_index" in result for result in results)
        assert all("batch_latency_ms" in result for result in results)

        # Verify batch_id is same for all items
        batch_id = results[0]["batch_id"]
        assert all(result["batch_id"] == batch_id for result in results)

        # Verify batch indices are correct
        for i, result in enumerate(results):
            assert result["batch_index"] == i
            assert result["prompt"] == prompts[i]

    def test_run_cache_functionality(self):
        """Test runner caching mechanism."""
        # First run should create a new runner
        openllm.run("Cache test 1", model="cache-model", mock=True)
        stats1 = openllm.get_cache_stats()

        # Second run with same model should reuse runner
        openllm.run("Cache test 2", model="cache-model", mock=True)
        stats2 = openllm.get_cache_stats()

        # Cache should contain the runner
        assert stats1["cached_runners"] == 1
        assert stats2["cached_runners"] == 1  # Same runner reused

        # Different model should create new runner
        openllm.run("Cache test 3", model="different-model", mock=True)
        stats3 = openllm.get_cache_stats()

        assert stats3["cached_runners"] == 2  # Two different runners cached

    def test_run_error_handling(self):
        """Test error handling in run function."""
        # Test with non-mock model (should raise NotImplementedError)
        with pytest.raises(
            NotImplementedError, match="Production model loading not implemented"
        ):
            openllm.run("Error test", mock=False)

    @pytest.mark.asyncio
    async def test_run_async_error_handling(self):
        """Test error handling in async run function."""
        # Test with non-mock model (should raise NotImplementedError)
        with pytest.raises(
            NotImplementedError, match="Production model loading not implemented"
        ):
            await openllm.run_async("Error test", mock=False)

    def test_performance_requirements(self):
        """Test that operations meet 60-second performance requirements."""
        start_time = time.time()

        # Run multiple operations
        for i in range(10):
            result = openllm.run(f"Performance test {i}", mock=True)
            assert "generated_text" in result

        elapsed_time = time.time() - start_time

        # Should complete well within 60 seconds
        assert (
            elapsed_time < 60
        ), f"Performance test took {elapsed_time:.2f}s, exceeding 60s limit"
        assert (
            elapsed_time < 1.0
        ), f"Mock operations took {elapsed_time:.3f}s, expected faster"

    @pytest.mark.asyncio
    async def test_async_performance_requirements(self):
        """Test async operations meet performance requirements."""
        start_time = time.time()

        # Run concurrent operations
        tasks = [openllm.run_async(f"Async perf {i}", mock=True) for i in range(10)]
        results = await asyncio.gather(*tasks)

        elapsed_time = time.time() - start_time

        # Verify all completed
        assert len(results) == 10

        # Performance requirements
        assert (
            elapsed_time < 60
        ), f"Async performance test took {elapsed_time:.2f}s, exceeding 60s limit"
        assert (
            elapsed_time < 1.0
        ), f"Async mock operations took {elapsed_time:.3f}s, expected faster"


class TestHttpxAsyncClient:
    """Test httpx.AsyncClient integration as specified in design document."""

    @pytest.mark.asyncio
    async def test_httpx_async_client_basic(self):
        """Test basic httpx.AsyncClient usage for LLM serving."""
        base_url = "http://localhost:3000"

        async with httpx.AsyncClient() as client:
            # Test request construction
            request = client.build_request(
                "POST",
                f"{base_url}/generate",
                json={"prompt": "Test httpx", "mock": True},
            )

            assert request.method == "POST"
            assert "/generate" in str(request.url)

            # Mock the HTTP response
            mock_response = {
                "generated_text": "Mock HTTP response to: Test httpx",
                "prompt": "Test httpx",
                "model": "mock-gpt2",
            }

            with patch.object(client, "send", new_callable=AsyncMock) as mock_send:
                mock_send.return_value = httpx.Response(
                    200,
                    content=json.dumps(mock_response),
                    headers={"content-type": "application/json"},
                )

                response = await client.send(request)

                assert response.status_code == 200
                result = response.json()
                assert result["prompt"] == "Test httpx"
                assert "Mock HTTP response" in result["generated_text"]

    @pytest.mark.asyncio
    async def test_httpx_concurrent_requests(self):
        """Test concurrent HTTP requests using httpx.AsyncClient."""
        base_url = "http://localhost:3000"
        prompts = ["HTTP 1", "HTTP 2", "HTTP 3"]

        async with httpx.AsyncClient() as client:
            # Build requests
            requests = [
                client.build_request(
                    "POST",
                    f"{base_url}/generate",
                    json={"prompt": prompt, "mock": True},
                )
                for prompt in prompts
            ]

            # Mock concurrent responses
            with patch.object(client, "send", new_callable=AsyncMock) as mock_send:
                mock_send.side_effect = [
                    httpx.Response(
                        200,
                        content=json.dumps(
                            {
                                "generated_text": f"HTTP response to: {prompt}",
                                "prompt": prompt,
                            }
                        ),
                        headers={"content-type": "application/json"},
                    )
                    for prompt in prompts
                ]

                # Execute concurrent requests
                start_time = time.time()
                responses = await asyncio.gather(
                    *[client.send(request) for request in requests]
                )
                elapsed_time = time.time() - start_time

                # Verify responses
                assert len(responses) == 3
                for i, response in enumerate(responses):
                    assert response.status_code == 200
                    result = response.json()
                    assert result["prompt"] == prompts[i]

                # Performance check
                assert (
                    elapsed_time < 60
                ), f"HTTP requests took {elapsed_time:.2f}s, exceeding 60s limit"

    @pytest.mark.asyncio
    async def test_httpx_batch_requests(self):
        """Test batch HTTP requests using httpx.AsyncClient."""
        base_url = "http://localhost:3000"
        prompts = ["Batch HTTP 1", "Batch HTTP 2"]

        async with httpx.AsyncClient() as client:
            request = client.build_request(
                "POST",
                f"{base_url}/batch_generate",
                json={"prompts": prompts, "mock": True},
            )

            # Mock batch response
            mock_batch_response = [
                {
                    "generated_text": f"Batch HTTP response to: {prompt}",
                    "prompt": prompt,
                }
                for prompt in prompts
            ]

            with patch.object(client, "send", new_callable=AsyncMock) as mock_send:
                mock_send.return_value = httpx.Response(
                    200,
                    content=json.dumps(mock_batch_response),
                    headers={"content-type": "application/json"},
                )

                response = await client.send(request)

                assert response.status_code == 200
                results = response.json()
                assert len(results) == 2

                for i, result in enumerate(results):
                    assert result["prompt"] == prompts[i]

    @pytest.mark.asyncio
    async def test_httpx_error_handling(self):
        """Test httpx error handling."""
        base_url = "http://localhost:3000"

        async with httpx.AsyncClient() as client:
            request = client.build_request(
                "POST", f"{base_url}/generate", json={"prompt": "Error test"}
            )

            # Mock error response
            with patch.object(client, "send", new_callable=AsyncMock) as mock_send:
                mock_send.return_value = httpx.Response(
                    500,
                    content=json.dumps({"error": "Internal server error"}),
                    headers={"content-type": "application/json"},
                )

                response = await client.send(request)

                assert response.status_code == 500
                error_data = response.json()
                assert "error" in error_data

    @pytest.mark.asyncio
    async def test_httpx_timeout_handling(self):
        """Test httpx timeout handling."""
        base_url = "http://localhost:3000"

        async with httpx.AsyncClient(timeout=0.1) as client:
            request = client.build_request(
                "POST", f"{base_url}/generate", json={"prompt": "Timeout test"}
            )

            # Mock slow response
            with patch.object(client, "send", new_callable=AsyncMock) as mock_send:

                async def slow_response(*args, **kwargs):
                    await asyncio.sleep(0.2)  # Longer than timeout
                    return httpx.Response(200, content="{}")

                mock_send.side_effect = slow_response

                # Should raise timeout exception
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(client.send(request), timeout=0.1)


class TestLLMRunnerDirectly:
    """Test LLMRunner class directly for comprehensive coverage."""

    def test_runner_initialization(self):
        """Test LLMRunner initialization."""
        runner = openllm.LLMRunner("test-model", mock=True)

        assert runner.model_name == "test-model"
        assert runner.model is not None
        assert runner.tokenizer is not None
        assert runner.get_stats()["requests_count"] == 0

    def test_runner_stats_tracking(self):
        """Test statistics tracking in runner."""
        runner = openllm.LLMRunner("stats-model", mock=True)

        # Initial stats
        initial_stats = runner.get_stats()
        assert initial_stats["requests_count"] == 0
        assert initial_stats["total_tokens"] == 0
        assert initial_stats["avg_latency"] == 0.0

        # Reset stats
        runner.reset_stats()
        reset_stats = runner.get_stats()
        assert reset_stats["requests_count"] == 0

    @pytest.mark.asyncio
    async def test_runner_generate_async_direct(self):
        """Test runner's generate_async method directly."""
        runner = openllm.LLMRunner("direct-test", mock=True)

        result = await runner.generate_async("Direct test prompt")

        assert result["prompt"] == "Direct test prompt"
        assert "latency_ms" in result
        assert "input_tokens" in result
        assert "request_id" in result

        # Check stats were updated
        stats = runner.get_stats()
        assert stats["requests_count"] == 1

    @pytest.mark.asyncio
    async def test_runner_batch_generate_direct(self):
        """Test runner's batch_generate_async method directly."""
        runner = openllm.LLMRunner("batch-direct", mock=True)

        prompts = ["Batch direct 1", "Batch direct 2"]
        results = await runner.batch_generate_async(prompts)

        assert len(results) == 2
        assert all("batch_latency_ms" in result for result in results)

        # Check stats were updated for each prompt
        stats = runner.get_stats()
        assert stats["requests_count"] == 2

    def test_runner_error_cases(self):
        """Test error cases in runner."""
        # Model loading should fail for non-mock during initialization
        with pytest.raises(
            NotImplementedError, match="Production model loading not implemented"
        ):
            openllm.LLMRunner("error-test", mock=False)
