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
from ..utils import LazyLoader
from ..utils.http import set_cookies
from ...exceptions import BentoMLException
from ..service.openapi import SUCCESS_DESCRIPTION
from ..service.openapi.specification import Schema
from ..service.openapi.specification import Response as OpenAPIResponse
from ..service.openapi.specification import MediaType
from ..service.openapi.specification import RequestBody

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from bentoml.grpc.v1 import service_pb2

    from ..context import InferenceApiContext as Context
    from ..server.grpc.types import BentoServicerContext

    FileKind: t.TypeAlias = t.Literal["binaryio", "textio"]
else:
    service_pb2 = LazyLoader("service_pb2", globals(), "bentoml.grpc.v1.service_pb2")

FileType: t.TypeAlias = t.Union[io.IOBase, t.IO[bytes], FileLike[bytes]]


class File(IODescriptor[FileType], proto_fields=["raw_value"]):
    """
    :obj:`File` defines API specification for the inputs/outputs of a Service, where either
    inputs will be converted to or outputs will be converted from file-like objects as
    specified in your API function signature.

    A sample ViT service:

    .. code-block:: python
       :caption: `service.py`

       from __future__ import annotations

       import io
       from typing import TYPE_CHECKING
       from typing import Any

       import bentoml
       from bentoml.io import File

       if TYPE_CHECKING:
           from numpy.typing import NDArray

       runner = bentoml.tensorflow.get('image-classification:latest').to_runner()

       svc = bentoml.Service("vit-pdf-classifier", runners=[runner])

       @svc.api(input=File(), output=NumpyNdarray(dtype="float32"))
       async def predict(input_pdf: io.BytesIO[Any]) -> NDArray[Any]:
           return await runner.async_run(input_pdf)

    Users then can then serve this service with :code:`bentoml serve`:

    .. code-block:: bash

        % bentoml serve ./service.py:svc --reload

    Users can then send requests to the newly started services with any client:

    .. tab-set::

        .. tab-item:: Bash

            .. code-block:: bash

               % curl -H "Content-Type: multipart/form-data" \\
                      -F 'fileobj=@test.pdf;type=application/pdf' \\
                      http://0.0.0.0:3000/predict

        .. tab-item:: Python

           .. code-block:: python
              :caption: `request.py`

              import requests

              requests.post(
                  "http://0.0.0.0:3000/predict",
                  files = {"upload_file": open('test.pdf', 'rb')},
                  headers = {"content-type": "multipart/form-data"}
              ).text

    Args:
        kind: The kind of file-like object to be used. Currently, the only accepted value is ``binaryio``.
        mime_type: Return MIME type of the :code:`starlette.response.Response`, only available when used as output descriptor

    Returns:
        :obj:`File`: IO Descriptor that represents file-like objects.

    """

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

    def openapi_schema(self) -> Schema:
        return Schema(type="string", format="binary")

    def openapi_components(self) -> dict[str, t.Any] | None:
        pass

    def openapi_request_body(self) -> RequestBody:
        return RequestBody(
            content={self._mime_type: MediaType(schema=self.openapi_schema())},
            required=True,
        )

    def openapi_responses(self) -> OpenAPIResponse:
        return OpenAPIResponse(
            description=SUCCESS_DESCRIPTION,
            content={self._mime_type: MediaType(schema=self.openapi_schema())},
        )

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

    def generate_protobuf(self):
        pass

    async def to_grpc_response(
        self, obj: FileType, context: BentoServicerContext
    ) -> service_pb2.Response:
        if isinstance(obj, bytes):
            body = obj
        else:
            body = obj.read()

        response = service_pb2.Response()
        value = service_pb2.Value()


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

    async def from_grpc_request(
        self, request: service_pb2.Request, context: BentoServicerContext
    ) -> t.IO[bytes]:
        pass
