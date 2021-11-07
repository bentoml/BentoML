import json
import logging
import typing as t

from starlette.requests import Request
from starlette.responses import Response

from ...exceptions import InvalidArgument
from ..utils.lazy_loader import LazyLoader
from .base import IODescriptor
from .json import MIME_TYPE_JSON

if t.TYPE_CHECKING:
    import numpy as np

    _major, _minor = list(
        map(lambda x: int(x), np.__version__.split(".")[:2])  # pylint: disable=W0108
    )
    if (_major, _minor) > (1, 20):
        from numpy.typing import ArrayLike, DTypeLike  # pylint: disable=E0611,W0611
    else:
        from ..typing_extensions.numpy import (  # pylint: disable=E0611,W0611
            ArrayLike,
            DTypeLike,
        )
else:
    np = LazyLoader("np", globals(), "numpy")

logger = logging.getLogger(__name__)


class NumpyNdarray(IODescriptor):
    """
    `NumpyNdarray` defines API specification for the inputs/outputs of a Service, where
     either inputs will be converted to or outputs will be converted from type
     `numpy.ndarray` as specified in your API function signature.

    .. Toy implementation of a sklearn service::
        # sklearn_svc.py
        import bentoml
        from bentoml.io import NumpyNdarray
        import bentoml.sklearn

        runner = bentoml.sklearn.load_runner("sklearn_model_clf")

        svc = bentoml.Service("iris-classifier", runners=[runner])

        @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
        def predict(input_arr):
            res = runner.run(input_arr)
            return res

    Users then can then serve this service with `bentoml serve`::
        % bentoml serve ./sklearn_svc.py:svc --auto-reload

        (Press CTRL+C to quit)
        [INFO] Starting BentoML API server in development mode with auto-reload enabled
        [INFO] Serving BentoML Service "iris-classifier" defined in "sklearn_svc.py"
        [INFO] API Server running on http://0.0.0.0:5000

    Users can then send a cURL requests like shown in different terminal session::
        % curl -X POST -H "Content-Type: application/json" --data '[[5,4,3,2]]'
          http://0.0.0.0:5000/predict

        [1]%

    Args:
        dtype (`~bentoml._internal.typing_extensions.numpy.DTypeLike`,
               `optional`, default to `None`):
            Data Type users wish to convert their inputs/outputs to. Refers to
             https://numpy.org/doc/stable/reference/arrays.dtypes.html for more
             information
        enforce_dtype (`bool`, `optional`, default to `False`):
            Whether to enforce a certain data type. if `enforce_dtype=True` then `dtype`
             must be specified.
        shape (`Tuple[int, ...]`, `optional`, default to `None`):
            Given shape that an array will be converted to. For example::
                from bentoml.io import NumpyNdarray
                arr = [[1,2,3]]  # shape (1,3)
                inp = NumpyNdarray.from_sample(arr)

                ...
                @svc.api(input=inp(shape=(3,1), enforce_shape=True),
                         output=NumpyNdarray())
                def predict(input_array: np.ndarray) -> np.ndarray:
                    # input_array will have shape (3,1)
                    result = await runner.run(input_array)
        enforce_shape (`bool`, `optional`, default to `False`):
            Whether to enforce a certain shape. If `enforce_shape=True` then `shape`
             must be specified

    Returns:
        IO Descriptor that represents `np.ndarray`.
    """

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
        obj = await request.json()
        res = np.array(obj)
        if self._enforce_dtype:
            if self._dtype is None:
                logger.warning("dtype is None or not yet specified.")
            else:
                res = res.astype(self._dtype)
        if self._enforce_shape:
            if self._shape is None:
                logger.warning("shape is None or not yet specified.")
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
        if not isinstance(obj, np.ndarray):
            raise InvalidArgument(
                f"return object is not of type `np.ndarray`, got type {type(obj)} instead"
            )
        return Response(content=json.dumps(obj.tolist()), media_type=MIME_TYPE_JSON)

    @classmethod
    def from_sample(
        cls,
        sample_input: "np.ndarray",
        enforce_dtype: bool = True,
        enforce_shape: bool = True,
    ) -> "NumpyNdarray":
        """
        Create a NumpyNdarray IO Descriptor from given inputs.

        Args:
            sample_input (`~bentoml._internal.typing_extensions.numpy.ArrayLike`):
                Given inputs
            enforce_dtype (`bool`, `optional`, default to `True`):
                Enforce a certain data type. `dtype` must be specified at function
                 signature. If you don't want to enforce a specific dtype then change
                 `enforce_dtype=False`.
            enforce_shape (`bool`, `optional`, default to `False`):
                Enforce a certain shape. `shape` must be specified at function
                 signature. If you don't want to enforce a specific shape then change
                 `enforce_shape=False`.

        Returns:
            `NumpyNdarray` IODescriptor from given users inputs.

        Examples::
            import numpy as np
            from bentoml.io import NumpyNdarray
            arr = [[1,2,3]]
            inp = NumpyNdarray.from_sample(arr)

            ...
            @svc.api(input=inp, output=NumpyNdarray())
            def predict() -> np.ndarray:...
        """
        return cls(
            dtype=sample_input.dtype,
            shape=sample_input.shape,
            enforce_dtype=enforce_dtype,
            enforce_shape=enforce_shape,
        )
