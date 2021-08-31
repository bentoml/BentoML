import typing as t

from ..types import HTTPRequest, HTTPResponse
from ..utils.lazy_loader import LazyLoader
from .base import IODescriptor

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

    def from_http_request(self, request: HTTPRequest) -> t.Union[np.ndarray, PIL.Image]:
        pass

    def to_http_response(self, obj: t.Union[np.ndarray, PIL.Image]) -> HTTPResponse:
        pass
