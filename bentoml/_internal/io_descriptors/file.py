from __future__ import annotations

import io
import typing as t
import logging
from typing import TYPE_CHECKING

from starlette.requests import Request
from multipart.multipart import parse_options_header
from starlette.responses import Response
from starlette.datastructures import UploadFile

from .base import IODescriptor
from ..types import FileLike
from ..utils.http import set_cookies
from ...exceptions import BentoMLException

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..context import InferenceApiContext as Context

    FileKind: t.TypeAlias = t.Literal["binaryio", "textio"]
FileType: t.TypeAlias = t.Union[io.IOBase, t.IO[bytes], FileLike[bytes]]


class File(IODescriptor[FileType]):
    """
    :code:`File` defines API specification for the inputs/outputs of a Service, where either
    inputs will be converted to or outputs will be converted from file-like objects as
    specified in your API function signature.

    Sample implementation of a ViT service:

    .. code-block:: python

        # vit_svc.py
        import bentoml
        from bentoml.io import File

        svc = bentoml.Service("vit-object-detection")

        @svc.api(input=File(), output=File())
        def predict(input_pdf):
            return input_pdf

    Users then can then serve this service with :code:`bentoml serve`:

    .. code-block:: bash

        % bentoml serve ./vit_svc.py:svc --auto-reload

        (Press CTRL+C to quit)
        [INFO] Starting BentoML API server in development mode with auto-reload enabled
        [INFO] Serving BentoML Service "vit-object-detection" defined in "vit_svc.py"
        [INFO] API Server running on http://0.0.0.0:3000

    Users can then send requests to the newly started services with any client:

    .. tabs::

        .. code-tab:: python

            import requests
            requests.post(
                "http://0.0.0.0:3000/predict",
                files = {"upload_file": open('test.pdf', 'rb')},
                headers = {"content-type": "multipart/form-data"}
            ).text


        .. code-tab:: bash

            % curl -H "Content-Type: multipart/form-data" -F 'fileobj=@test.pdf;type=application/pdf' http://0.0.0.0:3000/predict

    Args:
        mime_type (:code:`str`, `optional`, default to :code:`None`):
            Return MIME type of the :code:`starlette.response.Response`, only available
            when used as output descriptor

    Returns:
        :obj:`~bentoml._internal.io_descriptors.IODescriptor`: IO Descriptor that file-like objects.

    """

    _mime_type: str

    def __new__(  # pylint: disable=arguments-differ # returning subclass from new
        cls, kind: FileKind = "binaryio", mime_type: str | None = None
    ) -> File:
        mime_type = mime_type if mime_type is not None else "application/octet-stream"

        if kind == "binaryio":
            res = object.__new__(BytesIOFile)
        else:
            raise ValueError(f"invalid File kind '{kind}'")

        res._mime_type = mime_type
        return res

    def input_type(self) -> t.Type[t.Any]:
        return FileLike[bytes]

    def openapi_schema_type(self) -> dict[str, str]:
        return {"type": "string", "format": "binary"}

    def openapi_request_schema(self) -> dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""
        return {self._mime_type: {"schema": self.openapi_schema_type()}}

    def openapi_responses_schema(self) -> dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""
        return {self._mime_type: {"schema": self.openapi_schema_type()}}

    async def to_http_response(
        self,
        obj: FileType,
        ctx: Context | None = None,
    ):
        if isinstance(obj, bytes):
            body = obj
        else:
            body = obj.read()

        if ctx is not None:
            res = Response(
                body,
                headers=ctx.response.metadata,  # type: ignore (bad starlette types)
                status_code=ctx.response.status_code,
            )
            set_cookies(res, ctx.response.cookies)
        else:
            res = Response(body)
        return res


class BytesIOFile(File):
    async def from_http_request(self, request: Request) -> t.IO[bytes]:
        content_type, _ = parse_options_header(request.headers["content-type"])
        if content_type.decode("utf-8") == "multipart/form-data":
            form = await request.form()
            found_mimes: t.List[str] = []
            val: t.Union[str, UploadFile]
            for val in form.values():  # type: ignore
                if isinstance(val, UploadFile):
                    found_mimes.append(val.content_type)  # type: ignore (bad starlette types)
                    if val.content_type == self._mime_type:  # type: ignore (bad starlette types)
                        res = FileLike[bytes](val.file, val.filename)  # type: ignore (bad starlette types)
                        break
            else:
                if len(found_mimes) == 0:
                    raise BentoMLException("no File found in multipart form")
                else:
                    raise BentoMLException(
                        f"multipart File should have Content-Type '{self._mime_type}', got files with content types {', '.join(found_mimes)}"
                    )
            return res  # type: ignore
        if content_type.decode("utf-8") == self._mime_type:
            body = await request.body()
            return t.cast(t.IO[bytes], FileLike(io.BytesIO(body), "<request body>"))
        raise BentoMLException(
            f"File should have Content-Type '{self._mime_type}' or 'multipart/form-data', got {content_type} instead"
        )
