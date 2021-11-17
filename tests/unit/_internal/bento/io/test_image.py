import pytest


def test_image_io_init() -> None:
    from bentoml.exceptions import InvalidArgument
    from bentoml.io import Image

    Image()

    Image(pilmode="RGB")
    with pytest.raises(InvalidArgument):
        Image(pilmode="BBB")

    Image(mime_type="application/pdf")
    with pytest.raises(InvalidArgument):
        Image(mime_type="application/octet-stream")


import typing as t

LooseHeaders = t.Union[
    t.Mapping[str, str],
    t.Mapping[bytes, bytes],
    t.Sequence[t.Tuple[str, str]],
    t.Sequence[t.Tuple[bytes, bytes]],
    t.Sequence[t.List[str]],
    t.Sequence[t.List[bytes]],
]


def mock_request(headers: LooseHeaders, body: bytes):
    from starlette.requests import Request

    if hasattr(headers, "items"):
        raw_headers = [((k if isinstance(k, bytes) else ), (v)) for k, v in headers.items()]
    headers = 
    scope = dict(headers=list(ori_headers))
    scope["headers"] = list(ori_headers.items())
    req = Request(scope=scope)
    req._body = body


def test_image_io_from_http():
    from bentoml.io import Image

    image_io = Image()

    image_io.from_http_request()
