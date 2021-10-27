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

logger = logging.getLogger(__name__)


class NumpyNdarray(IODescriptor):
    """
    `NumpyNdarray` defines API specification for the inputs/outputs of a Service, where either inputs will be
    converted to or outputs will be converted from type `numpy.ndarray` as specified in your API function signature.

    .. Toy implementation of a sklearn service::
        # sklearn_svc.py
        import numpy as np
        import bentoml.sklearn

        from bentoml import Service
        from bentoml.io import NumpyNdarray

        my_runner = bentoml.sklearn.load_runner("my_sklearn_model:latest")

        svc = Service("IrisClassifierService", runner=[my_runner])

        # Create API function with pre- and post- processing logic
        @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
        def predict(input_array: np.ndarray) -> np.ndarray:
            result = await runner.run(input_array)
            # Define post-processing logic
            return result

    Users then can then serve this service with `bentoml serve`::
        % bentoml serve ./sklearn_svc.py:svc --auto-reload

        (Press CTRL+C to quit)
        [INFO] Starting BentoML API server in development mode with auto-reload enabled
        [INFO] Serving BentoML Service "IrisClassifierService" defined in "sklearn_svc.py"
        [INFO] API Server running on http://0.0.0.0:5000

    Users can then send a cURL requests like shown in different terminal session::
        % curl -X POST -H "application/json" --data "[[5, 4, 3, 2]]" http://0.0.0.0:5000/predict

        {res: [1]}%

    Args:
        dtype (`~bentoml._internal.typing_extensions.numpy.DTypeLike`, `optional`, default to `None`):
            Data Type users wish to convert their inputs/outputs to. Refers to
             https://numpy.org/doc/stable/reference/arrays.dtypes.html for more information
        enforce_dtype (`bool`, `optional`, default to `False`):
            Whether to enforce a certain data type. if `enforce_dtype=True` then `dtype` must
             be specified.
        shape (`Tuple[int, ...]`, `optional`, default to `None`):
            Given shape that an array will be converted to. For example::
                from bentoml.io import NumpyNdarray
                arr = [[1,2,3]]  # shape (1,3)
                inp = NumpyNdarray.from_sample(arr)

                ...
                @svc.api(input=inp(shape=(3,1), enforce_shape=True), output=NumpyNdarray())
                def predict(input_array: np.ndarray) -> np.ndarray:
                    # input_array will have shape (3,1)
                    result = await runner.run(input_array)
        enforce_shape (`bool`, `optional`, default to `False`):
            Whether to enforce a certain shape. If `enforce_shape=True` then `shape` must
             be specified

    Returns:
        IO Descriptor that represents `np.ndarray`.
    """  # noqa

    def __init__(
        self,
        dtype: t.Optional["DTypeLike"] = None,
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
        """Returns OpenAPI schema for incoming requests"""

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""

    async def from_http_request(self, request: Request) -> "ArrayLike":
        """
        Process incoming requests and convert incoming
         objects to `numpy.ndarray`

        Args:
            request (`starlette.requests.Requests`):
                Incoming Requests
        Returns:
            a `numpy.ndarray` object. This can then be used
             inside users defined logics.
        """
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
        """
        Process given objects and convert it to HTTP response.

        Args:
            obj (`np.ndarray`):
                `np.ndarray` that will be serialized to JSON
        Returns:
            HTTP Response of type `starlette.responses.Response`. This can
             be accessed via cURL or any external web traffic.
        """
        return Response(obj.tolist(), media_type=MIME_TYPE_JSON)

    @classmethod
    def from_sample(
        cls,
        sample_input: "ArrayLike",
        enforce_dtype: t.Optional[bool] = True,
        enforce_shape: t.Optional[bool] = True,
    ) -> "NumpyNdarray":
        """Create an IO Descriptor from given inputs."""
