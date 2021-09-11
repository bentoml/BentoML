import typing as t

import bentoml._internal.constants as const

from ..types import HTTPRequest, HTTPResponse
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

    def from_http_request(
        self, request: HTTPRequest
    ) -> t.Union["np.ndarray", "PIL.Image"]:
        pass

    def to_http_response(self, obj: t.Union["np.ndarray", "PIL.Image"]) -> HTTPResponse:
        pass
