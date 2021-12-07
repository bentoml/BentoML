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
    `File` defines API specification for the inputs/outputs of a Service, where either
     inputs will be converted to or outputs will be converted from file-like objects as
     specified in your API function signature.

    .. Toy implementation of a ViT service::
        # vit_svc.py
        import bentoml
        from bentoml.io import File

        svc = bentoml.Service("vit-object-detection")

        @svc.api(input=File(), output=File())
        def predict(input_pdf):
            return input_pdf

    Users then can then serve this service with `bentoml serve`::
        % bentoml serve ./vit_svc.py:svc --auto-reload

        (Press CTRL+C to quit)
        [INFO] Starting BentoML API server in development mode with auto-reload enabled
        [INFO] Serving BentoML Service "vit-object-detection" defined in "vit_svc.py"
        [INFO] API Server running on http://0.0.0.0:5000

    Users can then send a cURL requests like shown in different terminal session with an
     input PDF files::
        % curl -H "Content-Type: multipart/form-data" -F
          'fileobj=@test.pdf;type=application/pdf' http://0.0.0.0:5000/predict

        %PDF-1.7
                  zed 1/L 1959874/O 282/E 157204/N 12/T 1959312/H [ 500 221]>>
        <</DecodeParms<</Columns 5/Predictor 12>>/Filter/FlateDecode/ID[<5901CB390F31F2B4497ED82C610CB725><7DF08A87AED4DE458FEBB8B3350A46FF>]/Index[280 35]/Info 279 0 R/Length 88/Prev 1959313/Root 281 0 R/Size 315/Type/XRef/W[1 3 1]>>stream
        Warning: Binary output can mess up your terminal. Use "--output -" to tell
        Warning: curl to output it to your terminal anyway, or consider "--output
        Warning: <FILE>" to save to a file.

    Returns:
        IO Descriptor that represents file-like objects.
    """  # noqa: LN001

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
