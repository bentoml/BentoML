import typing as t

from ..types import HTTPRequest, HTTPResponse
from ..utils.lazy_loader import LazyLoader
from .base import IODescriptor

if t.TYPE_CHECKING:
    import numpy as np
else:
    np = LazyLoader("np", globals(), "numpy")


class NumpyNdarray(IODescriptor):
    def openapi_schema(self):
        pass

    def from_http_request(self, request: HTTPRequest) -> np.ndarray:
        pass

    def to_http_response(self, obj: np.ndarray) -> HTTPResponse:
        pass
