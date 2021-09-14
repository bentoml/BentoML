import typing as t

from multipart.multipart import parse_options_header
from starlette.requests import Request
from starlette.responses import Response

import bentoml._internal.constants as const

from ..utils import LazyLoader
from .base import IODescriptor

_exc = const.IMPORT_ERROR_MSG.format(
    fwr="Pillow",
    module=f"{__name__}.Image",
    inst="`pip install Pillow`",
)

if t.TYPE_CHECKING:  # pragma: no cover # pylint: disable=unused-import
    import numpy as np
    import PIL
else:
    np = LazyLoader(
        "np",
        globals(),
        "numpy",
        exc_msg="Make sure to install numpy with `pip install numpy`",
    )
    # TODO: use pillow-simd by default instead of pillow for better performance
    PIL = LazyLoader("PIL", globals(), "PIL", exc_msg=_exc)

DEFAULT_PIL_MODE = "RGB"


class Image(IODescriptor):
    def __init__(self, pilmode=DEFAULT_PIL_MODE):
        self.pilmode = pilmode

    def openapi_request_schema(self):
        pass

    def openapi_responses_schema(self):
        pass

    async def from_http_request(
        self, request: Request
    ) -> t.Union["np.ndarray", "PIL.Image"]:
        content_type_header = self.headers.get("Content-Type")
        content_type, _ = parse_options_header(content_type_header)

        if content_type == "application/json":
            json = await request.json()

        else:
            form = await request.form()
            filename = form["upload_file"].filename
            contents = await form["upload_file"].read()

    async def to_http_response(
        self, obj: t.Union["np.ndarray", "PIL.Image"]
    ) -> Response:
        pass
