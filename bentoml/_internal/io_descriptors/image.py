import io
import typing as t

from multipart.multipart import parse_options_header
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

import bentoml._internal.constants as const
from bentoml.exceptions import InvalidArgument

from ..utils import LazyLoader
from .base import IODescriptor

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    import numpy as np
    import PIL
    import PIL.Image
else:
    np = LazyLoader("np", globals(), "numpy")

    # NOTE: pillow-simd only benefits users who want to do preprocessing
    # TODO: add options for users to choose between simd and native mode
    _exc = const.IMPORT_ERROR_MSG.format(
        fwr="Pillow",
        module=f"{__name__}.Image",
        inst="`pip install Pillow`",
    )
    PIL = LazyLoader("PIL", globals(), "PIL", exc_msg=_exc)
    PIL.Image = LazyLoader("PIL.Image", globals(), "PIL.Image", exc_msg=_exc)

DEFAULT_PIL_MODE = "RGB"


class Image(IODescriptor):
    """Image."""

    def __init__(self, pilmode=DEFAULT_PIL_MODE):
        self._pilmode = pilmode

    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""

    async def from_http_request(self, request: Request) -> "PIL.Image":
        form = await request.form()
        contents = await form.upload_file.read()
        return PIL.Image.open(io.BytesIO(contents), mode=self._pilmode)

    @t.overload
    async def to_http_response(self, obj: "np.ndarray") -> StreamingResponse:
        ...

    @t.overload
    async def to_http_response(self, obj: "PIL.Image.Image") -> StreamingResponse:
        ...

    async def to_http_response(
        self, obj: t.Union["np.ndarray", "PIL.Image.Image"]
    ) -> StreamingResponse:
        if not any(isinstance(obj, i) for i in [np.ndarray, PIL.Image.Image]):
            raise InvalidArgument(
                f"Unsupported Image type received: {type(obj)},"
                " `bentoml.io.Image` supports only `np.ndarray` and `PIL.Image`"
            )
        image = PIL.Image.fromarray(obj) if isinstance(obj, np.ndarray) else obj

        # TODO: Support other return types?
        ret = io.BytesIO()
        image.save(ret, "PNG")
        return StreamingResponse(ret, media_type="image/png")
