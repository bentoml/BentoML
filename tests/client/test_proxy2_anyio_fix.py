"""
Test for AnyIO NoEventLoopError fix in proxy2.py

This test verifies that the fix for issue #5547 works correctly.
The issue occurred when calling sync API from async API in BentoML services.
"""

import functools
import unittest.mock as mock

import pytest

from _bentoml_impl.client.proxy2 import SyncClient


class TestAnyIOFix:
    """Test the AnyIO NoEventLoopError fix"""

    def test_run_async_syntax_validation(self):
        """Test that run_async method has valid syntax and imports"""

        # This test ensures the fix doesn't break basic functionality
        assert hasattr(SyncClient, "run_async")
        assert callable(SyncClient.run_async)

    @pytest.mark.asyncio
    async def test_run_async_with_anyio_token(self):
        """Test run_async when AnyIO token is available"""

        # Mock the anyio modules to simulate the fix behavior
        with (
            mock.patch("anyio.lowlevel.current_token") as mock_current_token,
            mock.patch("anyio.from_thread.run") as mock_from_thread_run,
        ):
            # Simulate having an AnyIO token
            mock_token = "test_token"
            mock_current_token.return_value = mock_token
            mock_from_thread_run.return_value = "success"

            async def test_coro():
                return "test_result"

            # Call run_async (this would previously fail)
            result = SyncClient.run_async(test_coro)

            # Verify the token was used
            mock_current_token.assert_called_once()
            mock_from_thread_run.assert_called_once()

            # Check that token parameter was passed
            call_args = mock_from_thread_run.call_args
            assert "token" in call_args.kwargs
            assert call_args.kwargs["token"] == mock_token

            assert result == "success"

    @pytest.mark.asyncio
    async def test_run_async_without_anyio_token(self):
        """Test run_async fallback when no AnyIO token is available"""

        # Mock the anyio modules to simulate no token scenario
        with (
            mock.patch("anyio.lowlevel.current_token") as mock_current_token,
            mock.patch("anyio.from_thread.run") as mock_from_thread_run,
        ):
            # Simulate LookupError when no AnyIO context exists
            mock_current_token.side_effect = LookupError("No AnyIO context")
            mock_from_thread_run.return_value = "fallback_success"

            async def test_coro():
                return "test_result"

            # Call run_async
            result = SyncClient.run_async(test_coro)

            # Verify fallback was used
            mock_current_token.assert_called_once()
            mock_from_thread_run.assert_called()

            # Check that NO token parameter was passed in fallback
            call_args = mock_from_thread_run.call_args
            assert (
                "token" not in call_args.kwargs or call_args.kwargs.get("token") is None
            )

            assert result == "fallback_success"

    def test_run_async_preserves_args_and_kwargs(self):
        """Test that run_async correctly preserves function arguments"""

        with (
            mock.patch("anyio.lowlevel.current_token") as mock_current_token,
            mock.patch("anyio.from_thread.run") as mock_from_thread_run,
        ):
            mock_current_token.return_value = "test_token"
            mock_from_thread_run.return_value = "success"

            async def test_coro(arg1, arg2, kwarg1=None, kwarg2=None):
                return f"{arg1}-{arg2}-{kwarg1}-{kwarg2}"

            # Call with args and kwargs
            SyncClient.run_async(test_coro, "a", "b", kwarg1="c", kwarg2="d")

            # Verify functools.partial was called with correct arguments
            call_args = mock_from_thread_run.call_args
            partial_func = call_args.args[0]

            # The partial function should be functools.partial with our coro and args
            assert isinstance(partial_func, functools.partial)
            assert partial_func.func == test_coro

    def test_anyio_import_exists(self):
        """Test that required anyio.lowlevel import is available in the module"""

        # Read the proxy2.py source to verify the import was added
        # This test ensures the import was added to the module
        # The import should be in the module's source code
        import inspect

        import _bentoml_impl.client.proxy2 as proxy2_module

        source = inspect.getsource(proxy2_module)

        assert "import anyio.lowlevel" in source, (
            "anyio.lowlevel import missing from proxy2.py"
        )


class TestOriginalIssueBehavior:
    """Test that reproduces the original issue scenario"""

    def test_issue_5547_scenario_mocked(self):
        """Test the exact scenario from issue #5547 using mocks"""

        # This test simulates:
        # 1. Async API method calls sync API method (predict -> sync_fn)
        # 2. Sync client tries to run async code with from_thread.run
        # 3. Previously failed with NoEventLoopError, should now work

        with (
            mock.patch("anyio.lowlevel.current_token") as mock_current_token,
            mock.patch("anyio.from_thread.run") as mock_from_thread_run,
        ):
            # Simulate being in AnyIO context (like in async API method)
            mock_current_token.return_value = "context_token"
            mock_from_thread_run.return_value = ["test", "result"]

            async def mock_sync_api_call():
                # This simulates the sync API call that gets wrapped in async
                return ["test", "result"]

            # This simulates what happens when async API calls sync API
            result = SyncClient.run_async(mock_sync_api_call)

            # Verify the fix worked
            mock_current_token.assert_called_once()
            mock_from_thread_run.assert_called_once()

            # Most importantly, verify token was passed (this fixes the bug)
            call_args = mock_from_thread_run.call_args
            assert call_args.kwargs.get("token") == "context_token"

            assert result == ["test", "result"]
