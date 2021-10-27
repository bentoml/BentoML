import logging
import typing as t

from starlette.requests import Request
from starlette.responses import Response

from ..utils.lazy_loader import LazyLoader
from .base import IODescriptor
from .json import MIME_TYPE_JSON

if t.TYPE_CHECKING:
    import numpy as np

    _major, _minor = list(map(lambda x: int(x), np.__version__.split(".")[:2]))
    if (_major, _minor) > (1, 20):
        from numpy.typing import ArrayLike, DTypeLike
    else:
        from ..typing_extensions.numpy import ArrayLike, DTypeLike

else:
    np = LazyLoader("np", globals(), "numpy")
    ArrayLike = t.Type["np.ndarray"]
    DTypeLike = t.Union["np.dtype", str]

logger = logging.getLogger(__name__)


class NumpyNdarray(IODescriptor):
    def __init__(
        self,
        dtype: t.Optional[DTypeLike] = None,
        enforce_dtype: bool = False,
        shape: t.Optional[t.Tuple[int, ...]] = None,
        enforce_shape: bool = False,
    ):
        if isinstance(dtype, str):
            dtype = np.dtype(dtype)

        self._dtype = dtype
        self._enforce_dtype = enforce_dtype
        self._shape = shape
        self._enforce_shape = enforce_shape

    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        pass

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        pass

    async def from_http_request(self, request: Request) -> ArrayLike:
        json_obj = await request.json()
        res = np.array(json_obj)
        if self._enforce_dtype:
            if self._dtype is None:
                logger.warning("dtype is None or not yet specified.")
                pass
            else:
                res = res.astype(self._dtype)
        if self._enforce_shape:
            if self._shape is None:
                logger.warning("shape is None or not yet specified.")
                pass
            else:
                res = res.reshape(self._shape)
        return res

    async def to_http_response(self, obj: "np.ndarray") -> Response:
        return Response(obj.tolist(), media_type=MIME_TYPE_JSON)

    @classmethod
    def from_sample(
        cls,
        sample_input: ArrayLike,
        enforce_dtype: t.Optional[bool] = True,
        enforce_shape: t.Optional[bool] = True,
    ) -> None:
        pass
