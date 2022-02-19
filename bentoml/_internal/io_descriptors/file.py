import typing as t
import logging
from io import BytesIO

from starlette.requests import Request
from multipart.multipart import parse_options_header
from starlette.responses import Response

from .base import IODescriptor
from ..types import FileLike
from ...exceptions import BentoMLException

logger = logging.getLogger(__name__)


class File(IODescriptor[FileLike]):
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

    def __init__(self, mime_type: t.Optional[str] = None):
        self._mime_type = (
            mime_type if mime_type is not None else "application/octet-stream"
        )

    def openapi_schema_type(self) -> t.Dict[str, str]:
        return {"type": "string", "format": "binary"}

    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""
        return {self._mime_type: {"schema": self.openapi_schema_type()}}

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""
        return {self._mime_type: {"schema": self.openapi_schema_type()}}

    async def from_http_request(self, request: Request) -> FileLike:
        content_type, _ = parse_options_header(request.headers["content-type"])
        if content_type.decode("utf-8") == "multipart/form-data":
            form = await request.form()
            f = next(iter(form.values()))
            content = await f.read()
            return FileLike(bytes_=content, name=f.filename)
        if content_type.decode("utf-8") == "application/octet-stream":
            body = await request.body()
            return FileLike(bytes_=body)
        raise BentoMLException(
            f"{self.__class__.__name__} should have Content-Type"
            f" b'application/octet-stream' or b'multipart/form-data',"
            f" got {content_type} instead"
        )

    async def to_http_response(self, obj: t.Union[FileLike, bytes]) -> Response:
        if isinstance(obj, bytes):
            obj = FileLike(bytes_=obj)
        return Response(
            t.cast(BytesIO, obj.stream).getvalue(), media_type=self._mime_type
        )
