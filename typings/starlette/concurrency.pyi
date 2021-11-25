

import typing
from typing import Any, AsyncGenerator, Iterator

T = ...

async def run_until_first_complete(
    *args: typing.Tuple[typing.Callable, dict]
) -> None: ...
async def run_in_threadpool(
    func: typing.Callable[..., T], *args: typing.Any, **kwargs: typing.Any
) -> T: ...

class _StopIteration(Exception): ...

async def iterate_in_threadpool(iterator: Iterator) -> AsyncGenerator: ...
