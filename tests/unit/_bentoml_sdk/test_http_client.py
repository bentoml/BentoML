from __future__ import annotations

from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.responses import Response
from starlette.routing import Route

from _bentoml_impl.client.http import AsyncHTTPClient
from _bentoml_impl.client.http import SyncHTTPClient


@pytest.fixture(
    params=[
        'attachment; filename="subdir/nested/document.pdf"',
        'attachment; filename="subdir\\nested\\document.pdf"',
        'attachment; filename="/var/data/document.pdf"',
        'attachment; filename="C:\\reports\\document.pdf"',
        'attachment; filename="../../document.pdf"',
        "attachment; filename*=utf-8''subdir%2Fnested%2Fdocument.pdf",
        'attachment; filename="document.pdf"',
        None,
    ],
    ids=[
        "relative",
        "backslashes",
        "absolute",
        "drive",
        "parent",
        "extended",
        "plain",
        "absent",
    ],
)
def file_response(request):
    content_disposition = request.param

    async def schema(request):
        return JSONResponse(
            {
                "routes": [
                    {
                        "name": "download",
                        "route": "/download",
                        "input": {"type": "object", "properties": {}},
                        "output": {"type": "file"},
                    }
                ]
            }
        )

    async def download(request):
        return Response(
            b"file contents",
            media_type="application/octet-stream",
            headers=(
                {"Content-Disposition": content_disposition}
                if content_disposition is not None
                else {}
            ),
        )

    app = Starlette(
        routes=[
            Route("/schema.json", schema),
            Route("/download", download, methods=["POST"]),
        ]
    )
    return app, ".pdf" if content_disposition else ""


def test_sync_client_download_discards_filename_directories(file_response):
    app, suffix = file_response
    with SyncHTTPClient("http://localhost", app=app, token="") as client:
        result = client.download()
        assert isinstance(result, Path)
        assert result.suffix == suffix
        if suffix:
            assert result.name.endswith("document.pdf")
            assert "\\" not in result.name
        assert result.read_bytes() == b"file contents"
        result.unlink()


@pytest.mark.asyncio
async def test_async_client_download_discards_filename_directories(file_response):
    app, suffix = file_response
    async with AsyncHTTPClient("http://localhost", app=app, token="") as client:
        result = await client.download()
        assert isinstance(result, Path)
        assert result.suffix == suffix
        if suffix:
            assert result.name.endswith("document.pdf")
            assert "\\" not in result.name
        assert result.read_bytes() == b"file contents"
        result.unlink()
