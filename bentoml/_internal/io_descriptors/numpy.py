from typing import TYPE_CHECKING, Optional, Tuple, Union

from starlette.requests import Request
from starlette.responses import Response

from ..utils.lazy_loader import LazyLoader
from .base import IODescriptor

if TYPE_CHECKING:
    import numpy as np
else:
    np = LazyLoader("np", globals(), "numpy")


class NumpyNdarray(IODescriptor):
    def __init__(
        self,
        dtype: Union["np.dtype", str, None] = None,
        enforce_dtype: bool = False,
        shape: Optional[Tuple] = None,
        enforce_shape: bool = False,
    ):
        if isinstance(dtype, str):
            dtype = np.dtype(dtype)

        self._dtype = dtype
        self._enforce_dtype = enforce_dtype
        self._shape = shape
        self._enforce_shape = enforce_shape

    def openapi_request_schema(self):
        pass

    def openapi_responses_schema(self):
        pass

    async def from_http_request(self, request: Request) -> "np.ndarray":
        pass

    async def to_http_response(self, obj: "np.ndarray") -> Response:
        pass

    @classmethod
    def from_sample(
        cls,
        sample_input: "np.ndarray",
        enforce_dtype=True,
        enforce_shape=True,
    ):
        pass
