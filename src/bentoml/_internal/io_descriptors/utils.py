from typing import AsyncGenerator
from typing import AsyncIterator


def to_sse(
    data: str, event: str | None = None, id: str | None = None, retry: str | None = None
) -> str:
    sse_data = ""
    if id is not None:
        sse_data += f"id: {id}\n"
    if event is not None:
        sse_data += f"event: {event}\n"
    if retry is not None:
        sse_data += f"retry: {retry}\n"
    sse_data += f"data: {data}\n\n"
    return sse_data


async def from_sse(async_iterator: AsyncIterator[str]) -> AsyncGenerator[str, None]:
    buffer = ""
    async for chunk in async_iterator:
        buffer += chunk
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            yield event
