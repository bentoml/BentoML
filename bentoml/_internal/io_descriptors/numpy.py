from __future__ import annotations

import json
import typing as t
import logging
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

from .base import IODescriptor
from .json import MIME_TYPE_JSON
from ..types import LazyType
from ..utils.http import set_cookies
from ...exceptions import BadInput
from ...exceptions import InternalServerError

if TYPE_CHECKING:
    import numpy as np

    from .. import external_typing as ext
    from ..context import InferenceApiContext as Context

logger = logging.getLogger(__name__)


def _is_matched_shape(
    left: t.Optional[t.Tuple[int, ...]],
    right: t.Optional[t.Tuple[int, ...]],
) -> bool:  # pragma: no cover
    if (left is None) or (right is None):
        return False

    if len(left) != len(right):
        return False

    for i, j in zip(left, right):
        if i == -1 or j == -1:
            continue
        if i == j:
            continue
        return False
    return True


class NumpyNdarray(IODescriptor["ext.NpNDArray"]):
    """
    :code:`NumpyNdarray` defines API specification for the inputs/outputs of a Service, where
    either inputs will be converted to or outputs will be converted from type
    :code:`numpy.ndarray` as specified in your API function signature.

    Sample implementation of a sklearn service:

    .. code-block:: python

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

    Users then can then serve this service with :code:`bentoml serve`:

    .. code-block:: bash

        % bentoml serve ./sklearn_svc.py:svc --auto-reload

        (Press CTRL+C to quit)
        [INFO] Starting BentoML API server in development mode with auto-reload enabled
        [INFO] Serving BentoML Service "iris-classifier" defined in "sklearn_svc.py"
        [INFO] API Server running on http://0.0.0.0:3000

    Users can then send requests to the newly started services with any client:

    .. tabs::

    .. code-block:: bash

        % curl -X POST -H "Content-Type: application/json" --data '[[5,4,3,2]]' http://0.0.0.0:3000/predict

        [1]%

    Args:
        dtype (:code:`numpy.typings.DTypeLike`, `optional`, default to :code:`None`):
            Data Type users wish to convert their inputs/outputs to. Refers to `arrays dtypes <https://numpy.org/doc/stable/reference/arrays.dtypes.html>`_ for more information.
        enforce_dtype (:code:`bool`, `optional`, default to :code:`False`):
            Whether to enforce a certain data type. if :code:`enforce_dtype=True` then :code:`dtype` must be specified.
        shape (:code:`Tuple[int, ...]`, `optional`, default to :code:`None`):
            Given shape that an array will be converted to. For example:

            .. code-block:: python

                from bentoml.io import NumpyNdarray

                @svc.api(input=NumpyNdarray(shape=(3,1), enforce_shape=True), output=NumpyNdarray())
                def predict(input_array: np.ndarray) -> np.ndarray:
                    # input_array will have shape (3,1)
                    result = await runner.run(input_array)

        enforce_shape (:code:`bool`, `optional`, default to :code:`False`):
            Whether to enforce a certain shape. If `enforce_shape=True` then `shape`
            must be specified

    Returns:
        :obj:`~bentoml._internal.io_descriptors.IODescriptor`: IO Descriptor that :code:`np.ndarray`.
    """

    def __init__(
        self,
        dtype: t.Optional[t.Union[str, "np.dtype[t.Any]"]] = None,
        enforce_dtype: bool = False,
        shape: t.Optional[t.Tuple[int, ...]] = None,
        enforce_shape: bool = False,
    ):
        import numpy as np

        if isinstance(dtype, str):
            dtype = np.dtype(dtype)

        self._dtype = dtype
        self._shape = shape
        self._enforce_dtype = enforce_dtype
        self._enforce_shape = enforce_shape

    def _infer_types(self) -> str:  # pragma: no cover
        if self._dtype is not None:
            name = self._dtype.name
            if name.startswith("int") or name.startswith("uint"):
                var_type = "integer"
            elif name.startswith("float") or name.startswith("complex"):
                var_type = "number"
            else:
                var_type = "object"
        else:
            var_type = "object"
        return var_type

    def _items_schema(self) -> t.Dict[str, t.Any]:
        if self._shape is not None:
            if len(self._shape) > 1:
                return {"type": "array", "items": {"type": self._infer_types()}}
            return {"type": self._infer_types()}
        return {}

    def input_type(self) -> LazyType["ext.NpNDArray"]:
        return LazyType("numpy", "ndarray")

    def openapi_schema_type(self) -> t.Dict[str, t.Any]:
        return {"type": "array", "items": self._items_schema()}

    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""
        return {MIME_TYPE_JSON: {"schema": self.openapi_schema_type()}}

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""
        return {MIME_TYPE_JSON: {"schema": self.openapi_schema_type()}}

    def _verify_ndarray(
        self,
        obj: "ext.NpNDArray",
        exception_cls: t.Type[Exception] = BadInput,
    ) -> "ext.NpNDArray":
        if self._dtype is not None and self._dtype != obj.dtype:
            if self._enforce_dtype:
                raise exception_cls(
                    f"{self.__class__.__name__}: enforced dtype mismatch"
                )
            try:
                obj = obj.astype(self._dtype)  # type: ignore
            except ValueError as e:
                logger.warning(f"{self.__class__.__name__}: {e}")

        if self._shape is not None and not _is_matched_shape(self._shape, obj.shape):
            if self._enforce_shape:
                raise exception_cls(
                    f"{self.__class__.__name__}: enforced shape mismatch"
                )
            try:
                obj = obj.reshape(self._shape)
            except ValueError as e:
                logger.warning(f"{self.__class__.__name__}: {e}")
        return obj

    async def from_http_request(self, request: Request) -> "ext.NpNDArray":
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
        import numpy as np

        obj = await request.json()
        res: "ext.NpNDArray"
        try:
            res = np.array(obj, dtype=self._dtype)  # type: ignore[arg-type]
        except ValueError:
            res = np.array(obj)  # type: ignore[arg-type]
        res = self._verify_ndarray(res, BadInput)
        return res

    async def to_http_response(self, obj: ext.NpNDArray, ctx: Context | None = None):
        """
        Process given objects and convert it to HTTP response.

        Args:
            obj (`np.ndarray`):
                `np.ndarray` that will be serialized to JSON
        Returns:
            HTTP Response of type `starlette.responses.Response`. This can
             be accessed via cURL or any external web traffic.
        """
        obj = self._verify_ndarray(obj, InternalServerError)
        if ctx is not None:
            res = Response(
                json.dumps(obj.tolist()),
                media_type=MIME_TYPE_JSON,
                headers=ctx.response.metadata,  # type: ignore (bad starlette types)
                status_code=ctx.response.status_code,
            )
            set_cookies(res, ctx.response.cookies)
            return res
        else:
            return Response(json.dumps(obj.tolist()), media_type=MIME_TYPE_JSON)

    def handle_tuple(self, proto_tuple):
        """
        Convert given protobuf tuple to a tuple list
        """
        import io_descriptors_pb2
        from google.protobuf.duration_pb2 import Duration
        from google.protobuf.timestamp_pb2 import Timestamp

        tuple_arr = [i for i in getattr(proto_tuple, "value_")]

        if not tuple_arr:
            raise ValueError("Provided tuple is either empty or invalid.")

        return_arr = []

        for i in range(len(tuple_arr)):
            val = getattr(tuple_arr[i], tuple_arr[i].WhichOneof("dtype"))

            if not val:
                raise ValueError("Provided protobuf tuple is missing a value.")

            if tuple_arr[i].WhichOneof("dtype") == "timestamp_":
                val = Timestamp.ToDatetime(val)
            elif tuple_arr[i].WhichOneof("dtype") == "duration_":
                val = Duration.ToTimedelta(val)

            if isinstance(val, io_descriptors_pb2.NumpyNdarray):
                val = self.proto_to_arr(val)
            elif isinstance(val, io_descriptors_pb2.Tuple):
                val = self.handle_tuple(val)
            return_arr.append(val)

        return return_arr

    def proto_to_arr(self, proto_arr):
        """
        Convert given protobuf array to python list
        """
        import io_descriptors_pb2
        from google.protobuf.duration_pb2 import Duration
        from google.protobuf.timestamp_pb2 import Timestamp

        return_arr = [i for i in getattr(proto_arr, proto_arr.dtype)]

        if proto_arr.dtype == "timestamp_":
            return_arr = [Timestamp.ToDatetime(dt) for dt in return_arr]
        elif proto_arr.dtype == "duration_":
            return_arr = [Duration.ToTimedelta(td) for td in return_arr]

        if not return_arr:
            raise ValueError("Provided array is either empty or invalid")

        for i in range(len(return_arr)):
            if isinstance(return_arr[i], io_descriptors_pb2.NumpyNdarray):
                return_arr[i] = self.proto_to_arr(return_arr[i])
            elif isinstance(return_arr[i], io_descriptors_pb2.Tuple):
                return_arr[i] = self.handle_tuple(return_arr[i])

        return return_arr

    async def from_grpc_request(self, request, context):
        """
        Process incoming protobuf request and convert it to `numpy.ndarray`

        Args:
            request (`io_descriptors_pb2.NumpyNdarray`):
                Incoming Requests
        Returns:
            a `numpy.ndarray` object. This can then be used
             inside users defined logics.
        """
        import numpy as np

        res: "ext.NpNDArray"
        try:
            res = np.array(self.proto_to_arr(request), dtype=self._dtype)  # type: ignore[arg-type]
        except ValueError:
            res = np.array(self.proto_to_arr(request))  # type: ignore[arg-type]
        res = self._verify_ndarray(res, BadInput)
        return res

    def is_supported(self, data):
        """
        Checks if the given type is within `supported_datatypes` dictionary
        """
        import datetime

        import numpy as np

        supported_datatypes = {
            np.int32: "sint32_",
            np.int64: "sint64_",
            np.uint32: "uint32_",
            np.uint64: "uint64_",
            np.float32: "float_",
            np.float64: "double_",
            np.bool_: "bool_",
            np.bytes_: "bytes_",
            np.str_: "string_",
            np.ndarray: "array_",
            np.datetime64: "timestamp_",
            np.timedelta64: "duration_"
            # TODO : complex types, lower byte integers(8,16)
        }

        found_dtype = ""
        for key in supported_datatypes:
            if np.dtype(type(data)) == key:
                found_dtype = supported_datatypes[key]
                if found_dtype == "array_":
                    if isinstance(data, datetime.datetime) or isinstance(
                        data, datetime.date
                    ):
                        found_dtype = "timestamp_"
                    elif isinstance(data, datetime.timedelta):
                        found_dtype = "duration_"
                break

        return found_dtype

    def create_tuple_proto(self, tuple):
        """
        Convert given tuple list or tuple array to protobuf
        """
        import datetime

        import numpy as np
        import io_descriptors_pb2
        from google.protobuf.duration_pb2 import Duration
        from google.protobuf.timestamp_pb2 import Timestamp

        if len(tuple) == 0:
            raise ValueError("Provided tuple is either empty or invalid.")

        tuple_arr = []
        for item in tuple:
            dtype = self.is_supported(item)

            if not dtype:
                raise ValueError(
                    f'Invalid datatype "{type(item).__name__}" within tuple.'
                )
            elif dtype == "timestamp_":
                if isinstance(item, np.datetime64):
                    item = item.astype(datetime.datetime)
                if isinstance(item, datetime.date):
                    item = datetime.datetime(item.year, item.month, item.day)
                t = Timestamp()
                t.FromDatetime(item)
                tuple_arr.append(io_descriptors_pb2.Value(**{"timestamp_": t}))
            elif dtype == "duration_":
                if isinstance(item, np.timedelta64):
                    item = item.astype(datetime.timedelta)
                d = Duration()
                d.FromTimedelta(item)
                tuple_arr.append(io_descriptors_pb2.Value(**{"duration_": d}))
            elif dtype == "array_":
                if not all(isinstance(x, type(item[0])) for x in item):
                    val = self.create_tuple_proto(item)
                    tuple_arr.append(io_descriptors_pb2.Value(tuple_=val))
                else:
                    val = self.arr_to_proto(item)
                    tuple_arr.append(io_descriptors_pb2.Value(array_=val))
            else:
                tuple_arr.append(io_descriptors_pb2.Value(**{f"{dtype}": item}))

        return io_descriptors_pb2.Tuple(value_=tuple_arr)

    def arr_to_proto(self, arr):
        """
        Convert given list or array to protobuf
        """
        import datetime

        import numpy as np
        import io_descriptors_pb2
        from google.protobuf.duration_pb2 import Duration
        from google.protobuf.timestamp_pb2 import Timestamp

        if len(arr) == 0:
            raise ValueError("Provided array is either empty or invalid.")
        if not all(isinstance(x, type(arr[0])) for x in arr):
            raise ValueError("Entered tuple, expecting array.")

        dtype = self.is_supported(arr[0])
        if not dtype:
            raise ValueError("Dtype is not supported.")

        if dtype == "timestamp_":
            timestamp_arr = []
            for dt in arr:
                if isinstance(dt, np.datetime64):
                    dt = dt.astype(datetime.datetime)
                if isinstance(dt, datetime.date):
                    dt = datetime.datetime(dt.year, dt.month, dt.day)
                t = Timestamp()
                t.FromDatetime(dt)
                timestamp_arr.append(t)
            return io_descriptors_pb2.NumpyNdarray(
                dtype="timestamp_", timestamp_=timestamp_arr
            )
        elif dtype == "duration_":
            duration_arr = []
            for td in arr:
                if isinstance(td, np.timedelta64):
                    td = td.astype(datetime.timedelta)
                d = Duration()
                d.FromTimedelta(td)
                duration_arr.append(d)
            return io_descriptors_pb2.NumpyNdarray(
                dtype="duration_", duration_=duration_arr
            )
        elif dtype != "array_":
            return io_descriptors_pb2.NumpyNdarray(**{"dtype": dtype, f"{dtype}": arr})

        return_arr = []
        is_tuple = False
        for i in range(len(arr)):
            if not all(isinstance(x, type(arr[i][0])) for x in arr[i]):
                is_tuple = True
                val = self.create_tuple_proto(arr[i])
            else:
                val = self.arr_to_proto(arr[i])
            return_arr.append(val)

        if is_tuple:
            return_arr = io_descriptors_pb2.NumpyNdarray(
                dtype="tuple_", tuple_=return_arr
            )
        else:
            return_arr = io_descriptors_pb2.NumpyNdarray(
                dtype="array_", array_=return_arr
            )

        return return_arr

    async def to_grpc_response(self, obj):
        """
        Process given objects and convert it to grpc protobuf response.

        Args:
            obj (`np.ndarray`):
                `np.ndarray` that will be serialized to protobuf
        Returns:
            `io_descriptor_pb2.NumpyNdarray`:
                Protobuf representation of given `np.ndarray`
        """
        obj = self._verify_ndarray(obj, InternalServerError)
        return self.arr_to_proto(obj)

    @classmethod
    def from_sample(
        cls,
        sample_input: "ext.NpNDArray",
        enforce_dtype: bool = True,
        enforce_shape: bool = True,
    ) -> "NumpyNdarray":
        """
        Create a NumpyNdarray IO Descriptor from given inputs.

        Args:
            sample_input (:code:`np.ndarray`): Given sample np.ndarray data
            enforce_dtype (;code:`bool`, `optional`, default to :code:`True`):
                Enforce a certain data type. :code:`dtype` must be specified at function
                signature. If you don't want to enforce a specific dtype then change
                :code:`enforce_dtype=False`.
            enforce_shape (:code:`bool`, `optional`, default to :code:`False`):
                Enforce a certain shape. :code:`shape` must be specified at function
                signature. If you don't want to enforce a specific shape then change
                :code:`enforce_shape=False`.

        Returns:
            :obj:`~bentoml._internal.io_descriptors.NumpyNdarray`: :code:`NumpyNdarray` IODescriptor from given users inputs.

        Example:

        .. code-block:: python

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
