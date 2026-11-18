# pylint: disable=unused-argument
from __future__ import annotations

import pandas as pd
import pytest
from starlette.requests import Request

from bentoml.exceptions import BadInput
from bentoml.io import PandasDataFrame
from bentoml.io import PandasSeries


def _make_request(body: bytes, content_type: str) -> Request:
    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.request", "body": b"", "more_body": False}

    sent = False
    scope = {
        "type": "http",
        "method": "POST",
        "headers": [(b"content-type", content_type.encode("latin-1"))],
        "path": "/",
        "query_string": b"",
        "server": ("testserver", 80),
        "scheme": "http",
        "client": ("testclient", 50000),
    }
    return Request(scope, receive=receive)


@pytest.mark.asyncio
async def test_dataframe_from_http_request_valid():
    res = await PandasDataFrame().from_http_request(
        _make_request(b'[{"a": 1}, {"a": 2}]', "application/json")
    )
    assert isinstance(res, pd.DataFrame)
    assert res["a"].tolist() == [1, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, content_type",
    [
        # bentoml/BentoML#5781: bodies the parser rejects must be a 400,
        # not a 500 (ValueError / UnicodeDecodeError escaped uncaught).
        (b"not json", "application/json"),
        (b"\xa9", "application/json"),
        (b"\xff\xfe,,,\n", "text/csv"),
    ],
)
async def test_dataframe_from_http_request_rejected_body_raises_bad_input(
    body: bytes, content_type: str
):
    with pytest.raises(BadInput):
        await PandasDataFrame().from_http_request(_make_request(body, content_type))


@pytest.mark.asyncio
async def test_series_from_http_request_rejected_body_raises_bad_input():
    with pytest.raises(BadInput):
        await PandasSeries().from_http_request(
            _make_request(b"not json", "application/json")
        )
