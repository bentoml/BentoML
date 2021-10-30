import inspect
import logging
import shutil
import typing as t
from pathlib import Path
from urllib.parse import urlparse

from starlette.requests import Request
from starlette.responses import FileResponse

from ...exceptions import BentoMLException, InvalidArgument
from ..types import FileLike, PathType
from .base import IODescriptor

logger = logging.getLogger(__name__)

FNAME = "bentoml_output_file"


class File(IODescriptor):
    """
    `File` defines API specification for the inputs/outputs of a Service, where either inputs will be
    converted to or outputs will be converted from file-like objects as specified in your API function signature.

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

    Users can then send a cURL requests like shown in different terminal session with an input PDF files::
        % curl -H "Content-Type: multipart/form-data" -F 'fileobj=@test.pdf;type=application/pdf' http://0.0.0.0:5000/predict

        Warning: Binary output can mess up your terminal. Use "--output -" to tell
        Warning: curl to output it to your terminal anyway, or consider "--output
        Warning: <FILE>" to save to a file.%

    We will then save the given file to your current directory of `vit_svc.py`.

    Args:
        fname (`str`, `optional`, default to `bentoml_output_file`):
            Given filename for Response output. If not specified the the output file will have the same extensions
             with the inputs file.
        output_dir (`PathType`, `optional`):
            Output path to save output files. If not specified, BentoML will use current directory containing your
             service definition. For example, if path to `vit_svc.py` is `/home/users/services/vit_svc.py` then the output
             file will be saved under `/home/users/services`

    Returns:
        IO Descriptor that represents file-like objects.
    """

    def __init__(
        self, fname: t.Optional[str] = None, *, output_dir: t.Optional[PathType] = None
    ):
        if not output_dir:
            output_dir = (
                Path(inspect.getframeinfo(inspect.stack()[1][0]).filename)
                .resolve()
                .parent
            )
        if not fname:
            fname = FNAME
        self._output_dir = output_dir
        self._output_file = Path(output_dir, fname)

    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""

    async def from_http_request(self, request: Request) -> FileLike:
        content_type = request.headers["content-type"].split(";")[0]
        if content_type != "multipart/form-data":
            raise BentoMLException(
                f"{self.__class__.__name__} should have `Content-Type: multipart/form-data`, got {content_type} instead"
            )
        form = await request.form()
        file = form[list(form.keys()).pop()]
        _output_file = self._output_file.with_suffix(Path(file.filename).suffix)
        contents = await file.read()
        return FileLike(bytes_=contents, uri=_output_file.as_uri())

    async def to_http_response(self, obj: t.Union[FileLike, bytes]) -> FileResponse:
        if not any(isinstance(obj, i) for i in [FileLike, bytes]):
            raise InvalidArgument(
                f"Unsupported Image type received: {type(obj)},"
                f" `{self.__class__.__name__}` FileLike objects."
            )
        if isinstance(obj, FileLike):
            resp = obj.bytes_
            _output_file = (
                self._output_file if not obj.uri else Path(urlparse(obj.uri).path)
            )
        else:
            resp = obj
            _output_file = self._output_file

        try:
            with _output_file.open("w+b") as of:
                of.write(resp)
            return FileResponse(path=str(_output_file))
        except Exception as e:  # noqa # pylint: disable=broad-except
            logger.warning(f"Failed to write {str(self._output_file)}, deleting...")
            shutil.rmtree(self._output_file)
            raise e
