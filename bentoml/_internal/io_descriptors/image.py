import typing as t

import bentoml._internal.constants as const
from bentoml.exceptions import MissingDependencyException

from ..types import HTTPRequest, HTTPResponse
from ..utils import LazyLoader, catch_exceptions
from .base import IODescriptor

_exc = MissingDependencyException(
    const.IMPORT_ERROR_MSG.format(
        fwr="Pillow",
        module=f"{__name__}.Image",
        inst="`pip install Pillow`",
    )
)

if t.TYPE_CHECKING:
    import numpy as np
    import PIL
else:
    np = LazyLoader("np", globals(), "numpy")
    # TODO: use pillow-simd by default instead of pillow for better performance
    PIL = LazyLoader("PIL", globals(), "PIL")

DEFAULT_PIL_MODE = "RGB"


class Image(IODescriptor):
    def __init__(self, pilmode=DEFAULT_PIL_MODE):
        self.pilmode = pilmode

    def openapi_schema(self):
        pass

    @catch_exceptions(catch_exc=ModuleNotFoundError, throw_exc=_exc)
    def from_http_request(
        self, request: HTTPRequest
    ) -> t.Union["np.ndarray", "PIL.Image"]:
        pass

    @catch_exceptions(catch_exc=ModuleNotFoundError, throw_exc=_exc)
    def to_http_response(self, obj: t.Union["np.ndarray", "PIL.Image"]) -> HTTPResponse:
        pass
