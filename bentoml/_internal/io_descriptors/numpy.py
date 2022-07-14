from __future__ import annotations

import json
import typing as t
import logging
import datetime
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

from .base import IODescriptor
from .json import MIME_TYPE_JSON
from ..types import LazyType
from ..utils.http import set_cookies
from ...exceptions import BadInput
from ...exceptions import BentoMLException
from ...exceptions import InternalServerError
from ..service.openapi import SUCCESS_DESCRIPTION
from ..utils.lazy_loader import LazyLoader
from ..service.openapi.specification import Schema
from ..service.openapi.specification import Response as OpenAPIResponse
from ..service.openapi.specification import MediaType
from ..service.openapi.specification import RequestBody

if TYPE_CHECKING:
    import numpy as np

    from .. import external_typing as ext
    from ..context import InferenceApiContext as Context
else:
    np = LazyLoader("np", globals(), "numpy")

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
    :obj:`NumpyNdarray` defines API specification for the inputs/outputs of a Service, where
    either inputs will be converted to or outputs will be converted from type
    :code:`numpy.ndarray` as specified in your API function signature.

    A sample service implementation:

    .. code-block:: python
       :caption: `service.py`

       from __future__ import annotations

       from typing import TYPE_CHECKING, Any

       import bentoml
       from bentoml.io import NumpyNdarray

       if TYPE_CHECKING:
           from numpy.typing import NDArray

       runner = bentoml.sklearn.get("sklearn_model_clf").to_runner()

       svc = bentoml.Service("iris-classifier", runners=[runner])

       @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
       def predict(input_arr: NDArray[Any]) -> NDArray[Any]:
           return runner.run(input_arr)

    Users then can then serve this service with :code:`bentoml serve`:

    .. code-block:: bash

        % bentoml serve ./service.py:svc --reload

    Users can then send requests to the newly started services with any client:

    .. tab-set::

        .. tab-item:: Bash

            .. code-block:: bash

                % curl -X POST -H "Content-Type: application/json" \\
                        --data '[[5,4,3,2]]' http://0.0.0.0:3000/predict

                # [1]%

        .. tab-item:: Python

            .. code-block:: python
               :caption: `request.py`

                import requests

                requests.post(
                    "http://0.0.0.0:3000/predict",
                    headers={"content-type": "application/json"},
                    data='[{"0":5,"1":4,"2":3,"3":2}]'
                ).text

    Args:
        dtype: Data type users wish to convert their inputs/outputs to. Refers to `arrays dtypes <https://numpy.org/doc/stable/reference/arrays.dtypes.html>`_ for more information.
        enforce_dtype: Whether to enforce a certain data type. if :code:`enforce_dtype=True` then :code:`dtype` must be specified.
        shape: Given shape that an array will be converted to. For example:

               .. code-block:: python
                  :caption: `service.py`

                  from bentoml.io import NumpyNdarray

                  @svc.api(input=NumpyNdarray(shape=(2,2), enforce_shape=False), output=NumpyNdarray())
                  async def predict(input_array: np.ndarray) -> np.ndarray:
                      # input_array will be reshaped to (2,2)
                      result = await runner.run(input_array)

               When ``enforce_shape=True`` is provided, BentoML will raise an exception if the input array received does not match the `shape` provided.
        enforce_shape: Whether to enforce a certain shape. If ``enforce_shape=True`` then ``shape`` must be specified.

    Returns:
        :obj:`~bentoml._internal.io_descriptors.IODescriptor`: IO Descriptor that represents a :code:`np.ndarray`.
    """

    def __init__(
        self,
        dtype: str | ext.NpDTypeLike | None = None,
        enforce_dtype: bool = False,
        shape: tuple[int, ...] | None = None,
        enforce_shape: bool = False,
    ):
        if dtype and not isinstance(dtype, np.dtype):
            # Convert from primitive type or type string, e.g.:
            # np.dtype(float)
            # np.dtype("float64")
            try:
                dtype = np.dtype(dtype)
            except TypeError as e:
                raise BentoMLException(
                    f'NumpyNdarray: Invalid dtype "{dtype}": {e}'
                ) from e

        self._dtype = dtype
        self._shape = shape
        self._enforce_dtype = enforce_dtype
        self._enforce_shape = enforce_shape

        self._sample_input = None

    def _openapi_types(self) -> str:
        # convert numpy dtypes to openapi compatible types.
        var_type = "integer"
        if self._dtype:
            name: str = self._dtype.name
            if name.startswith("float") or name.startswith("complex"):
                var_type = "number"
        return var_type

    def input_type(self) -> LazyType[ext.NpNDArray]:
        return LazyType("numpy", "ndarray")

    @property
    def sample_input(self) -> ext.NpNDArray | None:
        return self._sample_input

    @sample_input.setter
    def sample_input(self, value: ext.NpNDArray) -> None:
        self._sample_input = value

    def openapi_schema(self) -> Schema:
        # Note that we are yet provide
        # supports schemas for arrays that is > 2D.
        items = Schema(type=self._openapi_types())
        if self._shape and len(self._shape) > 1:
            items = Schema(type="array", items=Schema(type=self._openapi_types()))

        return Schema(type="array", items=items, nullable=True)

    def openapi_components(self) -> dict[str, t.Any] | None:
        pass

    def openapi_example(self) -> t.Any:
        if self.sample_input is not None:
            if isinstance(self.sample_input, np.generic):
                raise BadInput("NumpyNdarray: sample_input must be a numpy array.")
            return self.sample_input.tolist()
        return

    def openapi_request_body(self) -> RequestBody:
        return RequestBody(
            content={
                self._mime_type: MediaType(
                    schema=self.openapi_schema(), example=self.openapi_example()
                )
            },
            required=True,
        )

    def openapi_responses(self) -> OpenAPIResponse:
        return OpenAPIResponse(
            description=SUCCESS_DESCRIPTION,
            content={
                self._mime_type: MediaType(
                    schema=self.openapi_schema(), example=self.openapi_example()
                )
            },
        )

    def _verify_ndarray(
        self, obj: ext.NpNDArray, exception_cls: t.Type[Exception] = BadInput
    ) -> ext.NpNDArray:
        if self._dtype is not None and self._dtype != obj.dtype:
            # ‘same_kind’ means only safe casts or casts within a kind, like float64
            # to float32, are allowed.
            if np.can_cast(obj.dtype, self._dtype, casting="same_kind"):
                obj = obj.astype(self._dtype, casting="same_kind")  # type: ignore
            else:
                msg = f'{self.__class__.__name__}: Expecting ndarray of dtype "{self._dtype}", but "{obj.dtype}" was received.'
                if self._enforce_dtype:
                    raise exception_cls(msg)
                else:
                    logger.debug(msg)

        if self._shape is not None and not _is_matched_shape(self._shape, obj.shape):
            msg = f'{self.__class__.__name__}: Expecting ndarray of shape "{self._shape}", but "{obj.shape}" was received.'
            if self._enforce_shape:
                raise exception_cls(msg)
            try:
                obj = obj.reshape(self._shape)
            except ValueError as e:
                logger.debug(f"{msg} Failed to reshape: {e}.")

        return obj

    async def from_http_request(self, request: Request) -> "ext.NpNDArray":
        """
        Process incoming requests and convert incoming objects to ``numpy.ndarray``.

        Args:
            request: Incoming Requests

        Returns:
            a ``numpy.ndarray`` object. This can then be used
             inside users defined logics.
        """
        obj = await request.json()
        try:
            res = np.array(obj, dtype=self._dtype)
        except ValueError:
            res = np.array(obj)
        return self._verify_ndarray(res)

    async def to_http_response(self, obj: ext.NpNDArray, ctx: Context | None = None):
        """
        Process given objects and convert it to HTTP response.

        Args:
            obj: ``np.ndarray`` that will be serialized to JSON
            ctx: ``Context`` object that contains information about the request.

        Returns:
            HTTP Response of type ``starlette.responses.Response``. This can
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
        from google.protobuf.duration_pb2 import Duration
        from google.protobuf.timestamp_pb2 import Timestamp

        from bentoml.protos import payload_pb2

        tuple_arr = [i for i in getattr(proto_tuple, "value_")]

        if not tuple_arr:
            raise ValueError("Provided tuple is either empty or invalid.")

        return_arr = []

        for item in tuple_arr:
            val = getattr(item, item.WhichOneof("dtype"))

            if not val:
                raise ValueError("Provided protobuf tuple is missing a value.")

            if item.WhichOneof("dtype") == "timestamp_":
                val = Timestamp.ToDatetime(val)
            elif item.WhichOneof("dtype") == "duration_":
                val = Duration.ToTimedelta(val)

            if isinstance(val, payload_pb2.Array):
                val = self.proto_to_arr(val)
            elif isinstance(val, payload_pb2.Tuple):
                val = self.handle_tuple(val)
            return_arr.append(val)

        return return_arr

    def proto_to_arr(self, proto_arr):
        """
        Convert given protobuf array to python list
        """
        from google.protobuf.duration_pb2 import Duration
        from google.protobuf.timestamp_pb2 import Timestamp

        from bentoml.protos import payload_pb2

        return_arr = [i for i in getattr(proto_arr, proto_arr.dtype)]

        if proto_arr.dtype == "timestamp_":
            return_arr = [Timestamp.ToDatetime(dt) for dt in return_arr]
        elif proto_arr.dtype == "duration_":
            return_arr = [Duration.ToTimedelta(td) for td in return_arr]

        if not return_arr:
            raise ValueError("Provided array is either empty or invalid")

        for i, item in enumerate(return_arr):
            if isinstance(item, payload_pb2.Array):
                return_arr[i] = self.proto_to_arr(item)
            elif isinstance(item, payload_pb2.Tuple):
                return_arr[i] = self.handle_tuple(item)

        return return_arr

    async def from_grpc_request(self, request, context):
        """
        Process incoming protobuf request and convert it to `numpy.ndarray`

        Args:
            request (`payload_pb2.Array`):
                Incoming Requests
        Returns:
            a `numpy.ndarray` object. This can then be used
             inside users defined logics.
        """
        import numpy as np

        res: "ext.NpNDArray"
        try:
            res = np.array(self.proto_to_arr(request), dtype=self._dtype)
        except ValueError:
            res = np.array(self.proto_to_arr(request))
        res = self._verify_ndarray(res, BadInput)
        return res

    def is_supported(self, data):
        """
        Checks if the given type is within `supported_datatypes` dictionary
        """
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
        import numpy as np
        from google.protobuf.duration_pb2 import Duration
        from google.protobuf.timestamp_pb2 import Timestamp

        from bentoml.protos import payload_pb2

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
                tuple_arr.append(
                    payload_pb2.Value(**{"timestamp_": Timestamp().FromDatetime(item)})
                )
            elif dtype == "duration_":
                if isinstance(item, np.timedelta64):
                    item = item.astype(datetime.timedelta)
                tuple_arr.append(
                    payload_pb2.Value(**{"duration_": Duration().FromTimedelta(item)})
                )
            elif dtype == "array_":
                if not all(isinstance(x, type(item[0])) for x in item):
                    val = self.create_tuple_proto(item)
                    tuple_arr.append(payload_pb2.Value(tuple_=val))
                else:
                    val = self.arr_to_proto(item)
                    tuple_arr.append(payload_pb2.Value(array_=val))
            else:
                tuple_arr.append(payload_pb2.Value(**{f"{dtype}": item}))

        return payload_pb2.Tuple(value_=tuple_arr)

    def arr_to_proto(self, arr):
        """
        Convert given list or array to protobuf
        """
        import numpy as np
        from google.protobuf.duration_pb2 import Duration
        from google.protobuf.timestamp_pb2 import Timestamp

        from bentoml.protos import payload_pb2

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
                timestamp_arr.append(Timestamp().FromDatetime(dt))
            return payload_pb2.Array(dtype="timestamp_", timestamp_=timestamp_arr)
        elif dtype == "duration_":
            duration_arr = []
            for td in arr:
                if isinstance(td, np.timedelta64):
                    td = td.astype(datetime.timedelta)
                duration_arr.append(Duration().FromTimedelta(td))
            return payload_pb2.Array(dtype="duration_", duration_=duration_arr)
        elif dtype != "array_":
            return payload_pb2.Array(**{"dtype": dtype, f"{dtype}": arr})

        return_arr = []
        is_tuple = False
        for item in arr:
            if not all(isinstance(x, type(item[0])) for x in item):
                is_tuple = True
                val = self.create_tuple_proto(item)
            else:
                val = self.arr_to_proto(item)
            return_arr.append(val)

        if is_tuple:
            return_arr = payload_pb2.Array(dtype="tuple_", tuple_=return_arr)
        else:
            return_arr = payload_pb2.Array(dtype="array_", array_=return_arr)

        return return_arr

    async def to_grpc_response(self, obj):
        """
        Process given objects and convert it to grpc protobuf response.

        Args:
            obj (`np.ndarray`):
                `np.ndarray` that will be serialized to protobuf
        Returns:
            `io_descriptor_pb2.Array`:
                Protobuf representation of given `np.ndarray`
        """
        obj = self._verify_ndarray(obj, InternalServerError)
        return self.arr_to_proto(obj)

    @classmethod
    def from_sample(
        cls,
        sample_input: ext.NpNDArray,
        enforce_dtype: bool = True,
        enforce_shape: bool = True,
    ) -> NumpyNdarray:
        """
        Create a :obj:`NumpyNdarray` IO Descriptor from given inputs.

        Args:
            sample_input: Given sample ``np.ndarray`` data
            enforce_dtype: Enforce a certain data type. :code:`dtype` must be specified at function
                           signature. If you don't want to enforce a specific dtype then change
                           :code:`enforce_dtype=False`.
            enforce_shape: Enforce a certain shape. :code:`shape` must be specified at function
                           signature. If you don't want to enforce a specific shape then change
                           :code:`enforce_shape=False`.

        Returns:
            :obj:`NumpyNdarray`: :code:`NumpyNdarray` IODescriptor from given users inputs.

        Example:

        .. code-block:: python
           :caption: `service.py`

           from __future__ import annotations

           from typing import TYPE_CHECKING, Any

           import bentoml
           from bentoml.io import NumpyNdarray

           import numpy as np

           if TYPE_CHECKING:
               from numpy.typing import NDArray

           input_spec = NumpyNdarray.from_sample(np.array([[1,2,3]]))

           @svc.api(input=input_spec, output=NumpyNdarray())
           async def predict(input: NDArray[np.int16]) -> NDArray[Any]:
               return await runner.async_run(input)
        """
        if isinstance(sample_input, np.generic):
            raise BentoMLException(
                "NumpyNdarray.from_sample() expects a numpy.array, not numpy.generic."
            )

        inst = cls(
            dtype=sample_input.dtype,
            shape=sample_input.shape,
            enforce_dtype=enforce_dtype,
            enforce_shape=enforce_shape,
        )
        inst.sample_input = sample_input

        return inst
