from __future__ import annotations

import io
import typing as t

import attr

from bentoml.exceptions import BentoMLException

if t.TYPE_CHECKING:

    class SSEArgs(t.TypedDict, total=False):
        data: str
        event: str
        id: str
        retry: int


@attr.s(auto_attribs=True)
class SSE:
    data: str
    event: str | None = None
    id: str | None = None
    retry: int | None = None

    def marshal(self) -> str:
        with io.StringIO() as buffer:
            if self.event:
                if "\n" in self.event:
                    raise BentoMLException("SSE event name should not contain new line")
                buffer.write(f"event: {self.event}\n")
            if self.id is not None:
                buffer.write(f"id: {self.id}\n")
            if self.retry is not None:
                buffer.write(f"retry: {self.retry}\n")
            for line in self.data.split("\n"):
                buffer.write(f"data: {line}\n")
            buffer.write("\n")
            return buffer.getvalue()

    @classmethod
    def _read_sse(cls, buffer: io.StringIO) -> SSE:
        event: SSEArgs = {}
        data_buffer: io.StringIO | None = None
        first_data_line = True

        while True:
            line = buffer.readline()
            if not line:
                break
            if line == "\n":
                break
            if line.startswith("data: "):
                if first_data_line:
                    first_data_line = False
                    event["data"] = line[6:-1]
                else:
                    if data_buffer is None:
                        # only init data_buffer when there is more than one data line
                        data_buffer = io.StringIO()
                        data_buffer.write(event.get("data", ""))
                    data_buffer.write("\n")
                    data_buffer.write(line[6:-1])
            if line.startswith("event: "):
                event["event"] = line[7:].strip()
            if line.startswith("id: "):
                event["id"] = line[4:].strip()
            if line.startswith("retry: "):
                event["retry"] = int(line[7:].strip())

        if data_buffer is not None:
            event["data"] = data_buffer.getvalue()
            data_buffer.close()

        return SSE(**event)

    @classmethod
    async def from_iterator(
        cls,
        async_iterator: t.AsyncIterator[str],
    ) -> t.AsyncGenerator[SSE, None]:
        with io.StringIO() as buffer:
            read_cursor = 0
            last_chunk = ""
            async for chunk in async_iterator:
                buffer.write(chunk)
                if (
                    "\n\n" in chunk
                    or chunk
                    and chunk[0] == "\n"
                    and last_chunk
                    and last_chunk[-1] == "\n"
                ):
                    buffer.seek(read_cursor)
                    yield cls._read_sse(buffer)
                    read_cursor = buffer.tell()
                last_chunk = chunk
