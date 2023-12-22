from typing import AsyncGenerator
from typing import AsyncIterator
from typing import Optional

import attr

from bentoml.exceptions import BentoMLException


@attr.s(auto_attribs=True)
class SSE:
    data: str
    event: Optional[str] = None
    id: Optional[str] = None
    retry: Optional[int] = None

    def marshal(self) -> str:
        sse_data = ""
        if self.id is not None:
            sse_data += f"id: {self.id}\n"
        if self.event is not None:
            sse_data += f"event: {self.event}\n"
        if self.retry is not None:
            sse_data += f"retry: {self.retry}\n"

        # Handle multi-line data
        data_lines = self.data.split("\n")
        for line in data_lines:
            sse_data += f"data: {line}\n"

        # Add an extra newline to mark the end of the message
        sse_data += "\n"
        return sse_data

    @staticmethod
    async def from_iterator(
        async_iterator: AsyncIterator[str],
    ) -> AsyncGenerator["SSE", None]:
        buffer = ""
        async for chunk in async_iterator:
            buffer += chunk
            while "\n\n" in buffer:
                event_text, buffer = buffer.split("\n\n", 1)
                fields = {"data": "", "event": None, "id": None, "retry": None}
                data = []
                for line in event_text.split("\n"):
                    key, _, value = line.partition(": ")
                    if key == "data":
                        data.append(value)
                    if key == "retry":
                        fields[key] = int(value)
                    elif key in fields:
                        fields[key] = value
                    else:
                        raise BentoMLException("Invalid SSE message")
                fields["data"] = "\n".join(data)
                yield SSE(**fields)
