from __future__ import annotations

import io
import os
import typing as t
import logging
from functools import lru_cache

from starlette.requests import Request
from multipart.multipart import parse_options_header
from starlette.responses import Response
from starlette.datastructures import UploadFile

from .base import IODescriptor
from ..types import FileLike
from ..utils import resolve_user_filepath
from ..utils.http import set_cookies
from ...exceptions import BadInput
from ...exceptions import InvalidArgument
from ...exceptions import BentoMLException
from ...exceptions import MissingDependencyException
from ...grpc.utils import import_generated_stubs
from ..service.openapi import SUCCESS_DESCRIPTION
from ..service.openapi.specification import Schema
from ..service.openapi.specification import MediaType

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from bentoml.grpc.v1 import service_pb2 as pb
    from bentoml.grpc.v1alpha1 import service_pb2 as pb_v1alpha1

    from .base import OpenAPIResponse
    from ..context import ServiceContext as Context

    FileKind: t.TypeAlias = t.Literal["binaryio", "textio"]
else:
    pb, _ = import_generated_stubs("v1")
    pb_v1alpha1, _ = import_generated_stubs("v1alpha1")

FileType = t.Union[io.IOBase, t.IO[bytes], FileLike[bytes]]


class File(
    IODescriptor[FileType], descriptor_id="bentoml.io.File", proto_fields=("file",)
):
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

    def __new__(
        cls, kind: FileKind = "binaryio", mime_type: str | None = None, **kwargs: t.Any
    ) -> File:
        if kind == "binaryio":
            res = super().__new__(BytesIOFile, **kwargs)
        else:
            raise ValueError(f"invalid File kind '{kind}'")
        res._mime_type = mime_type
        return res

    def _from_sample(self, sample: FileType | str) -> FileType:
        """
        Create a :class:`~bentoml._internal.io_descriptors.file.File` IO Descriptor from given inputs.

        Args:
            sample: Given File-like object, or a path to a file.
            kind: The kind of file-like object to be used. Currently, the only accepted value is ``binaryio``.
            mime_type: Optional MIME type for the descriptor. If not provided, ``from_sample``
                       will try to infer the MIME type from the file extension.

        Returns:
            :class:`~bentoml._internal.io_descriptors.file.File`: IODescriptor from given users inputs.

        Example:

        .. code-block:: python
           :caption: `service.py`

           from __future__ import annotations

           import bentoml
           from typing import Any
           from bentoml.io import File

           input_spec = File.from_sample("/path/to/file.pdf")
           @svc.api(input=input_spec, output=File())
           async def predict(input: t.IO[t.Any]) -> t.IO[t.Any]:
               return await runner.async_run(input)

        Raises:
            :class:`MissingDependencyException`: If ``filetype`` is not installed.
            ValueError: If the MIME type cannot be inferred automatically.
            :class:`BadInput`: Any other errors that may occur during the process of figure out the MIME type.
        """
        try:
            import filetype
        except ModuleNotFoundError:
            raise MissingDependencyException(
                "'filetype' is required to use 'from_sample'. Install it with 'pip install bentoml[io-file]'."
            )
        if isinstance(sample, t.IO):
            sample = FileLike[bytes](sample, "<sample>")
        elif isinstance(sample, (str, os.PathLike)):
            p = resolve_user_filepath(sample, ctx=None)
            mime = filetype.guess_mime(p)
            self._mime_type = mime
            with open(p, "rb") as f:
                sample = FileLike[bytes](f, "<sample>")
        return sample

    @classmethod
    def from_spec(cls, spec: dict[str, t.Any]) -> t.Self:
        if "args" not in spec:
            raise InvalidArgument(f"Missing args key in File spec: {spec}")
        return cls(**spec["args"])

    def input_type(self) -> t.Type[t.Any]:
        return FileLike[bytes]

    def openapi_schema(self) -> Schema:
        return Schema(type="string", format="binary")

    def openapi_components(self) -> dict[str, t.Any] | None:
        pass

    def openapi_example(self):
        pass

    def openapi_request_body(self) -> dict[str, t.Any]:
        return {
            "content": {
                "*/*"
                if self._mime_type is None
                else self._mime_type: MediaType(schema=self.openapi_schema())
            },
            "required": True,
            "x-bentoml-io-descriptor": self.to_spec(),
        }

    def openapi_responses(self) -> OpenAPIResponse:
        return {
            "description": SUCCESS_DESCRIPTION,
            "content": {
                "*/*"
                if self._mime_type is None
                else self._mime_type: MediaType(schema=self.openapi_schema())
            },
            "x-bentoml-io-descriptor": self.to_spec(),
        }

    async def to_http_response(self, obj: FileType, ctx: Context | None = None):
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
            res = Response(
                body,
                headers={
                    "content-type": self._mime_type
                    if self._mime_type
                    else "application/octet-stream"
                },
            )
        return res

    async def to_proto(self, obj: FileType) -> pb.File:
        if isinstance(obj, bytes):
            body = obj
        else:
            body = obj.read()

        return pb.File(
            kind="appliction/octet-stream"
            if self._mime_type is None
            else self._mime_type,
            content=body,
        )

    async def to_proto_v1alpha1(self, obj: FileType) -> pb_v1alpha1.File:
        if isinstance(obj, bytes):
            body = obj
        else:
            body = obj.read()

        try:
            kind = mimetype_to_filetype_pb_map()[
                "application/octet-stream"
                if self._mime_type is None
                else self._mime_type
            ]
        except KeyError:
            raise BadInput(
                f"{self._mime_type} doesn't have a corresponding File 'kind'"
            ) from None

        return pb_v1alpha1.File(kind=kind, content=body)

    async def from_proto(self, field: pb.File | bytes) -> FileLike[bytes]:
        raise NotImplementedError

    async def from_http_request(self, request: Request) -> FileLike[bytes]:
        raise NotImplementedError


class BytesIOFile(File, descriptor_id=None):
    def to_spec(self) -> dict[str, t.Any]:
        return {
            "id": super().descriptor_id,
            "args": {
                "kind": "binaryio",
                "mime_type": self._mime_type,
            },
        }

    async def from_http_request(self, request: Request) -> FileLike[bytes]:
        content_type, _ = parse_options_header(request.headers["content-type"])
        if content_type.decode("utf-8") == "multipart/form-data":
            form = await request.form()
            found_mimes: t.List[str] = []
            val: t.Union[str, UploadFile]
            for val in form.values():  # type: ignore
                if isinstance(val, UploadFile):
                    found_mimes.append(val.content_type)  # type: ignore (bad starlette types)
                    if self._mime_type is None or val.content_type == self._mime_type:  # type: ignore (bad starlette types)
                        res = FileLike[bytes](val.file, val.filename)  # type: ignore (bad starlette types)
                        break
            else:
                if len(found_mimes) == 0:
                    raise BentoMLException("no File found in multipart form")
                else:
                    raise BentoMLException(
                        f"The File IO descriptor requires input to be of Content-Type '{self._mime_type}', got files with content types {', '.join(found_mimes)}"
                    )
            return res  # type: ignore
        if self.mime_type is None or content_type.decode("utf-8") == self._mime_type:
            body = await request.body()
            return FileLike[bytes](io.BytesIO(body), "<request body>")
        raise BentoMLException(
            f"File should have Content-Type '{self._mime_type}' or 'multipart/form-data', got {content_type} instead"
        )

    async def from_proto(
        self, field: pb.File | pb_v1alpha1.File | bytes
    ) -> FileLike[bytes]:
        # check if the request message has the correct field
        if isinstance(field, bytes):
            content = field
        elif isinstance(field, pb_v1alpha1.File):
            mapping = filetype_pb_to_mimetype_map()
            if field.kind:
                try:
                    mime_type = mapping[field.kind]
                    if self._mime_type is not None and mime_type != self._mime_type:
                        raise BadInput(
                            f"Inferred mime_type from 'kind' is '{mime_type}', while '{self!r}' is expecting '{self._mime_type}'",
                        )
                except KeyError:
                    raise BadInput(
                        f"{field.kind} is not a valid File kind. Accepted file kind: {[names for names,_ in pb_v1alpha1.File.FileType.items()]}",
                    ) from None
            content = field.content
            if not content:
                raise BadInput("Content is empty!") from None
            return FileLike[bytes](io.BytesIO(content), "<content>")
        else:
            assert isinstance(field, pb.File)
            if (
                field.kind is not None
                and self._mime_type is not None
                and field.kind != self._mime_type
            ):
                raise BadInput(
                    f"MIME type from 'kind' is '{field.kind}', while '{self!r}' is expecting '{self._mime_type}'",
                )
            content = field.content
            if not content:
                raise BadInput("Content is empty!") from None

        return FileLike[bytes](io.BytesIO(content), "<content>")


# v1alpha1 backward compatibility
@lru_cache(maxsize=1)
def filetype_pb_to_mimetype_map() -> dict[pb_v1alpha1.File.FileType.ValueType, str]:
    return {
        pb_v1alpha1.File.FILE_TYPE_CSV: "text/csv",
        pb_v1alpha1.File.FILE_TYPE_PLAINTEXT: "text/plain",
        pb_v1alpha1.File.FILE_TYPE_JSON: "application/json",
        pb_v1alpha1.File.FILE_TYPE_BYTES: "application/octet-stream",
        pb_v1alpha1.File.FILE_TYPE_PDF: "application/pdf",
        pb_v1alpha1.File.FILE_TYPE_PNG: "image/png",
        pb_v1alpha1.File.FILE_TYPE_JPEG: "image/jpeg",
        pb_v1alpha1.File.FILE_TYPE_GIF: "image/gif",
        pb_v1alpha1.File.FILE_TYPE_TIFF: "image/tiff",
        pb_v1alpha1.File.FILE_TYPE_BMP: "image/bmp",
        pb_v1alpha1.File.FILE_TYPE_WEBP: "image/webp",
        pb_v1alpha1.File.FILE_TYPE_SVG: "image/svg+xml",
    }


@lru_cache(maxsize=1)
def mimetype_to_filetype_pb_map() -> dict[str, pb_v1alpha1.File.FileType.ValueType]:
    return {v: k for k, v in filetype_pb_to_mimetype_map().items()}
