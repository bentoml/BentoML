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
        self._output_file = Path(output_dir, fname)
        print(self._output_file)

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
