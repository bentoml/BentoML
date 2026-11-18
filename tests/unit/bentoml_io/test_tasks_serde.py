from __future__ import annotations

import pytest
from starlette.responses import Response

from _bentoml_impl.tasks.serde import JSONSerde


@pytest.mark.asyncio
async def test_response_headers_roundtrip_preserves_latin1_bytes():
    serde = JSONSerde()
    response = Response(content=b"ok", status_code=200)
    response.raw_headers.append((b"x-raw-name", b"caf\xe9"))

    data = await serde.serialize_response(response)
    restored = await serde.deserialize_response(data)

    restored_headers = dict(restored.raw_headers)
    assert restored_headers[b"x-raw-name"] == b"caf\xe9"
