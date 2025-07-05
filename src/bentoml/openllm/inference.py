"""
High-level inference functions for OpenLLM integration.

This module provides the main `run` functions that were intended to be tested
in the original design document.
"""

import asyncio
from typing import Any
from typing import Dict
from typing import List

from .runner import LLMRunner

# Global runner cache for model reuse
_runner_cache: Dict[str, LLMRunner] = {}


def get_or_create_runner(model_name: str, mock: bool = False, **kwargs) -> LLMRunner:
    """
    Get or create a runner for the specified model.

    Args:
        model_name: Name of the model
        mock: Whether to use mock model for testing
        **kwargs: Additional runner configuration

    Returns:
        LLMRunner instance for the model
    """
    cache_key = f"{model_name}_{mock}_{hash(frozenset(kwargs.items()))}"

    if cache_key not in _runner_cache:
        runner = LLMRunner(model_name, mock=mock, **kwargs)
        _runner_cache[cache_key] = runner

    return _runner_cache[cache_key]


def run(
    prompt: str,
    model: str = "mock-gpt2",
    max_length: int = 100,
    temperature: float = 1.0,
    mock: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Synchronously run LLM inference on a prompt.

    This is the main synchronous function intended to be tested.

    Args:
        prompt: Input text prompt
        model: Model name to use for inference
        max_length: Maximum tokens to generate
        temperature: Sampling temperature
        mock: Whether to use mock model for testing
        **kwargs: Additional generation parameters

    Returns:
        Generation result dictionary
    """
    runner = get_or_create_runner(model, mock=mock, **kwargs)

    # Run async method in sync context
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running, create new one
        return asyncio.run(
            runner.generate_async(
                prompt, max_length=max_length, temperature=temperature, **kwargs
            )
        )
    else:
        # Event loop already running, run in executor
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                runner.generate_async(
                    prompt, max_length=max_length, temperature=temperature, **kwargs
                ),
            )
            return future.result()


async def run_async(
    prompt: str,
    model: str = "mock-gpt2",
    max_length: int = 100,
    temperature: float = 1.0,
    mock: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Asynchronously run LLM inference on a prompt.

    This is the main async function intended to be tested.

    Args:
        prompt: Input text prompt
        model: Model name to use for inference
        max_length: Maximum tokens to generate
        temperature: Sampling temperature
        mock: Whether to use mock model for testing
        **kwargs: Additional generation parameters

    Returns:
        Generation result dictionary
    """
    runner = get_or_create_runner(model, mock=mock, **kwargs)

    return await runner.generate_async(
        prompt, max_length=max_length, temperature=temperature, **kwargs
    )


async def batch_run_async(
    prompts: List[str],
    model: str = "mock-gpt2",
    max_length: int = 100,
    temperature: float = 1.0,
    mock: bool = False,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Asynchronously run LLM inference on multiple prompts.

    This function processes multiple prompts concurrently.

    Args:
        prompts: List of input prompts
        model: Model name to use for inference
        max_length: Maximum tokens to generate per prompt
        temperature: Sampling temperature
        mock: Whether to use mock model for testing
        **kwargs: Additional generation parameters

    Returns:
        List of generation results
    """
    runner = get_or_create_runner(model, mock=mock, **kwargs)

    return await runner.batch_generate_async(
        prompts, max_length=max_length, temperature=temperature, **kwargs
    )


def clear_cache():
    """Clear the runner cache."""
    global _runner_cache
    _runner_cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return {
        "cached_runners": len(_runner_cache),
        "cache_keys": list(_runner_cache.keys()),
    }
