from __future__ import annotations

import json
import typing as t
import logging
from functools import lru_cache

from starlette.requests import Request
from starlette.responses import Response

from .base import IODescriptor
from ..types import LazyType
from ..utils import LazyLoader
from ..utils.http import set_cookies
from ...exceptions import BadInput
from ...exceptions import InvalidArgument
from ...exceptions import UnprocessableEntity
from ...grpc.utils import import_generated_stubs
from ...grpc.utils import LATEST_PROTOCOL_VERSION
from ..service.openapi import SUCCESS_DESCRIPTION
from ..service.openapi.specification import Schema
from ..service.openapi.specification import MediaType

if t.TYPE_CHECKING:
    import numpy as np
    import pyarrow
    import pyspark.sql.types
    from google.protobuf import message as _message

    from bentoml.grpc.v1 import service_pb2 as pb
    from bentoml.grpc.v1alpha1 import service_pb2 as pb_v1alpha1

    from .. import external_typing as ext
    from .base import OpenAPIResponse
    from ..context import InferenceApiContext as Context
else:
    pb, _ = import_generated_stubs("v1")
    pb_v1alpha1, _ = import_generated_stubs("v1alpha1")
    np = LazyLoader("np", globals(), "numpy")

logger = logging.getLogger(__name__)


# TODO: support the following types for for protobuf message:
# - support complex64, complex128, object and struct types
# - BFLOAT16, QINT32, QINT16, QUINT16, QINT8, QUINT8
#
# For int16, uint16, int8, uint8 -> specify types in NumpyNdarray + using int_values.
#
# For bfloat16, half (float16) -> specify types in NumpyNdarray + using float_values.
#
# for string_values, use <U for np.dtype instead of S (zero-terminated bytes).
FIELDPB_TO_NPDTYPE_NAME_MAP = {
    "bool_values": "bool",
    "float_values": "float32",
    "string_values": "<U",
    "double_values": "float64",
    "int32_values": "int32",
    "int64_values": "int64",
    "uint32_values": "uint32",
    "uint64_values": "uint64",
}


@t.overload
def dtypepb_to_npdtype_map(
    version: t.Literal["v1"] = ...,
) -> dict[pb.NDArray.DType.ValueType, ext.NpDTypeLike]:
    ...


@t.overload
def dtypepb_to_npdtype_map(
    version: t.Literal["v1alpha1"] = ...,
) -> dict[pb_v1alpha1.NDArray.DType.ValueType, ext.NpDTypeLike]:
    ...


@lru_cache(maxsize=2)
def dtypepb_to_npdtype_map(
    version: str = LATEST_PROTOCOL_VERSION,
) -> dict[int, ext.NpDTypeLike]:
    pb, _ = import_generated_stubs(version)
    # pb.NDArray.Dtype -> np.dtype
    return {
        pb.NDArray.DTYPE_FLOAT: np.dtype("float32"),
        pb.NDArray.DTYPE_DOUBLE: np.dtype("double"),
        pb.NDArray.DTYPE_INT32: np.dtype("int32"),
        pb.NDArray.DTYPE_INT64: np.dtype("int64"),
        pb.NDArray.DTYPE_UINT32: np.dtype("uint32"),
        pb.NDArray.DTYPE_UINT64: np.dtype("uint64"),
        pb.NDArray.DTYPE_BOOL: np.dtype("bool"),
        pb.NDArray.DTYPE_STRING: np.dtype("<U"),
    }


@t.overload
def dtypepb_to_fieldpb_map(
    version: t.Literal["v1"] = ...,
) -> dict[pb.NDArray.DType.ValueType, str]:
    ...


@t.overload
def dtypepb_to_fieldpb_map(
    version: t.Literal["v1alpha1"] = ...,
) -> dict[pb_v1alpha1.NDArray.DType.ValueType, str]:
    ...


@lru_cache(maxsize=2)
def dtypepb_to_fieldpb_map(version: str = LATEST_PROTOCOL_VERSION) -> dict[int, str]:
    return {
        k: npdtype_to_fieldpb_map()[v]
        for k, v in dtypepb_to_npdtype_map(version).items()
    }


@lru_cache(maxsize=1)
def fieldpb_to_npdtype_map() -> dict[str, ext.NpDTypeLike]:
    # str -> np.dtype
    return {k: np.dtype(v) for k, v in FIELDPB_TO_NPDTYPE_NAME_MAP.items()}


@t.overload
def npdtype_to_dtypepb_map(
    version: t.Literal["v1"] = ...,
) -> dict[ext.NpDTypeLike, pb.NDArray.DType.ValueType]:
    ...


@t.overload
def npdtype_to_dtypepb_map(
    version: t.Literal["v1alpha1"] = ...,
) -> dict[ext.NpDTypeLike, pb_v1alpha1.NDArray.DType.ValueType]:
    ...


@lru_cache(maxsize=2)
def npdtype_to_dtypepb_map(
    version: str = LATEST_PROTOCOL_VERSION,
) -> dict[ext.NpDTypeLike, int]:
    # np.dtype -> pb.NDArray.Dtype
    return {v: k for k, v in dtypepb_to_npdtype_map(version).items()}


@lru_cache(maxsize=1)
def npdtype_to_fieldpb_map() -> dict[ext.NpDTypeLike, str]:
    # np.dtype -> str
    return {v: k for k, v in fieldpb_to_npdtype_map().items()}


def _is_matched_shape(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
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


# TODO: when updating docs, add examples with gRPCurl
class NumpyNdarray(
    IODescriptor["ext.NpNDArray"],
    descriptor_id="bentoml.io.NumpyNdarray",
    proto_fields=("ndarray",),
):
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
        dtype: Data type users wish to convert their inputs/outputs to. Refer to `arrays dtypes <https://numpy.org/doc/stable/reference/arrays.dtypes.html>`_ for more information.
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

               .. dropdown:: About the behaviour of ``shape``
                  :icon: triangle-down

                  If specified, then both :meth:`bentoml.io.NumpyNdarray.from_http_request` and :meth:`bentoml.io.NumpyNdarray.from_proto`
                  will reshape the input array before sending it to the API function.
        enforce_shape: Whether to enforce a certain shape. If ``enforce_shape=True`` then ``shape`` must be specified.

    Returns:
        :obj:`~bentoml._internal.io_descriptors.IODescriptor`: IO Descriptor that represents a :code:`np.ndarray`.
    """

    _mime_type = "application/json"

    def __init__(
        self,
        dtype: str | ext.NpDTypeLike | None = None,
        enforce_dtype: bool = False,
        shape: tuple[int, ...] | None = None,
        enforce_shape: bool = False,
    ):
        if dtype and not isinstance(dtype, np.dtype):
            # Convert from primitive type or type string, e.g.: np.dtype(float) or np.dtype("float64")
            try:
                dtype = np.dtype(dtype)
            except TypeError as e:
                raise UnprocessableEntity(f'Invalid dtype "{dtype}": {e}') from None

        self._dtype = dtype
        self._shape = shape
        self._enforce_dtype = enforce_dtype
        self._enforce_shape = enforce_shape

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

    def to_spec(self) -> dict[str, t.Any]:
        return {
            "id": self.descriptor_id,
            "args": {
                "dtype": None if self._dtype is None else self._dtype.name,
                "shape": self._shape,
                "enforce_dtype": self._enforce_dtype,
                "enforce_shape": self._enforce_shape,
            },
        }

    @classmethod
    def from_spec(cls, spec: dict[str, t.Any]) -> t.Self:
        if "args" not in spec:
            raise InvalidArgument(f"Missing args key in NumpyNdarray spec: {spec}")
        res = NumpyNdarray(**spec["args"])
        return res

    def openapi_schema(self) -> Schema:
        # Note that we are yet provide
        # supports schemas for arrays that is > 2D.
        items = Schema(type=self._openapi_types())
        if self._shape and len(self._shape) > 1:
            items = Schema(type="array", items=Schema(type=self._openapi_types()))

        return Schema(type="array", items=items, nullable=True)

    def openapi_components(self) -> dict[str, t.Any] | None:
        pass

    def openapi_example(self):
        if self.sample is not None:
            if isinstance(self.sample, np.generic):
                raise BadInput("NumpyNdarray: sample must be a numpy array.") from None
            return self.sample.tolist()

    def openapi_request_body(self) -> dict[str, t.Any]:
        return {
            "content": {
                self._mime_type: MediaType(
                    schema=self.openapi_schema(), example=self.openapi_example()
                )
            },
            "required": True,
            "x-bentoml-io-descriptor": self.to_spec(),
        }

    def openapi_responses(self) -> OpenAPIResponse:
        return {
            "description": SUCCESS_DESCRIPTION,
            "content": {
                self._mime_type: MediaType(
                    schema=self.openapi_schema(), example=self.openapi_example()
                )
            },
            "x-bentoml-io-descriptor": self.to_spec(),
        }

    def validate_array(
        self, arr: ext.NpNDArray, exception_cls: t.Type[Exception] = BadInput
    ) -> ext.NpNDArray:
        if self._dtype is not None and self._dtype != arr.dtype:
            # ‘same_kind’ means only safe casts or casts within a kind, like float64
            # to float32, are allowed.
            if np.can_cast(arr.dtype, self._dtype, casting="same_kind"):
                arr = arr.astype(self._dtype, casting="same_kind")  # type: ignore
            else:
                msg = '%s: Expecting ndarray of dtype "%s", but "%s" was received.'
                if self._enforce_dtype:
                    raise exception_cls(
                        msg % (self.__class__.__name__, self._dtype, arr.dtype)
                    ) from None
                else:
                    logger.debug(msg, self.__class__.__name__, self._dtype, arr.dtype)

        if self._shape is not None and not _is_matched_shape(self._shape, arr.shape):
            msg = '%s: Expecting ndarray of shape "%s", but "%s" was received.'
            if self._enforce_shape:
                raise exception_cls(
                    msg % (self.__class__.__name__, self._shape, arr.shape)
                ) from None
            try:
                arr = arr.reshape(self._shape)
            except ValueError as e:
                logger.debug(
                    msg + "Failed to reshape: %s",
                    self.__class__.__name__,
                    self._shape,
                    arr.shape,
                    e,
                )

        return arr

    async def from_http_request(self, request: Request) -> ext.NpNDArray:
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

        return self.validate_array(res)

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
        obj = self.validate_array(obj)

        if ctx is not None:
            res = Response(
                json.dumps(obj.tolist()),
                media_type=self._mime_type,
                headers=ctx.response.metadata,  # type: ignore (bad starlette types)
                status_code=ctx.response.status_code,
            )
            set_cookies(res, ctx.response.cookies)
            return res
        else:
            return Response(json.dumps(obj.tolist()), media_type=self._mime_type)

    def _from_sample(self, sample: ext.NpNDArray | t.Sequence[t.Any]) -> ext.NpNDArray:
        """
        Create a :class:`~bentoml._internal.io_descriptors.numpy.NumpyNdarray` IO Descriptor from given inputs.

        Args:
            sample: Given sample ``np.ndarray`` data. It also accepts a sequence-like data type that
                    can be converted to ``np.ndarray``.
            enforce_dtype: Enforce a certain data type. :code:`dtype` must be specified at function
                           signature. If you don't want to enforce a specific dtype then change
                           :code:`enforce_dtype=False`.
            enforce_shape: Enforce a certain shape. :code:`shape` must be specified at function
                           signature. If you don't want to enforce a specific shape then change
                           :code:`enforce_shape=False`.

        Returns:
            :class:`~bentoml._internal.io_descriptors.numpy.NumpyNdarray`: IODescriptor from given users inputs.

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

        Raises:
            :class:`BadInput`: If given sample is a type ``numpy.generic``. This exception
                                       will also be raised if we failed to create a ``np.ndarray``
                                       from given sample.
        """
        if isinstance(sample, np.generic):
            raise BadInput(
                "'NumpyNdarray.from_sample()' expects a 'numpy.array', not 'numpy.generic'."
            ) from None
        try:
            if not isinstance(sample, np.ndarray):
                sample = np.array(sample)
        except ValueError:
            raise BadInput(f"Given sample ({sample}) is not a numpy ND-array") from None
        if self._dtype is None:
            self._dtype = sample.dtype
        if self._shape is None:
            self._shape = sample.shape
        return sample

    async def from_proto(self, field: pb.NDArray | bytes) -> ext.NpNDArray:
        """
        Process incoming protobuf request and convert it to ``numpy.ndarray``

        Args:
            request: Incoming RPC request message.
            context: grpc.ServicerContext

        Returns:
            ``numpy.ndarray``: A ``np.array`` constructed from given protobuf message.

        .. seealso::

            :ref:`Protobuf representation of np.ndarray <guides/grpc:Array representation via \\`\\`NDArray\\`\\`>`

        .. note::

           Currently, we support ``pb.NDArray`` and ``serialized_bytes`` as valid inputs.
           ``serialized_bytes`` will be prioritised over ``pb.NDArray`` if both are provided.
           Serialized bytes has a specialized bytes representation and should not be used by users directly.
        """
        if isinstance(field, bytes):
            # We will be using ``np.frombuffer`` to deserialize the bytes.
            # This means that we need to ensure that ``dtype`` are provided to the IO descriptor
            #
            # ```python
            # from __future__ import annotations
            #
            # import numpy as np
            #
            # @svc.api(input=NumpyNdarray(dtype=np.float16), output=NumpyNdarray())
            # def predict(input: np.ndarray):
            #     ... # input will be serialized with np.frombuffer, and hence dtype is required
            # ```
            if not self._dtype:
                raise BadInput(
                    "'serialized_bytes' requires specifying 'dtype'."
                ) from None

            dtype: ext.NpDTypeLike = self._dtype
            array = np.frombuffer(field, dtype=self._dtype)
        else:
            if isinstance(field, pb_v1alpha1.NDArray):
                version = "v1alpha1"
            elif isinstance(field, pb.NDArray):
                version = "v1"
            else:
                raise BadInput(
                    f"Expected 'pb.NDArray' or 'bytes' but received {type(field)}"
                ) from None

            # The behaviour of dtype are as follows:
            # - if not provided:
            #     * All of the fields are empty, then we return a ``np.empty``.
            #     * We will loop through all of the provided fields, and only allows one field per message.
            #       If here are more than two fields (i.e. ``string_values`` and ``float_values``), then we will raise an error, as we don't know how to deserialize the data.
            # - if provided:
            #     * We will use the provided dtype-to-field maps to get the data from the given message.
            if field.dtype == pb.NDArray.DTYPE_UNSPECIFIED:
                dtype = None
            else:
                try:
                    dtype = dtypepb_to_npdtype_map(version=version)[field.dtype]
                except KeyError:
                    raise BadInput(f"{field.dtype} is invalid.") from None
            if dtype is not None:
                values_array = getattr(
                    field, dtypepb_to_fieldpb_map(version=version)[field.dtype]
                )
            else:
                fieldpb = [
                    f.name for f, _ in field.ListFields() if f.name.endswith("_values")
                ]
                if len(fieldpb) == 0:
                    # input message doesn't have any fields.
                    return np.empty(shape=field.shape or 0)
                elif len(fieldpb) > 1:
                    # when there are more than two values provided in the proto.
                    raise BadInput(
                        f"Array contents can only be one of given values key. Use one of '{fieldpb}' instead.",
                    ) from None

                dtype: ext.NpDTypeLike = fieldpb_to_npdtype_map()[fieldpb[0]]
                values_array = getattr(field, fieldpb[0])
            try:
                array = np.array(values_array, dtype=dtype)
            except ValueError:
                array = np.array(values_array)

            # We will try to reshape the array if ``shape`` is provided.
            # Note that all of the logics here are handled in-place, meaning that we will ensure
            # not to create new copies of given initialized array.
            if field.shape:
                array = np.reshape(array, field.shape)

        # We will try to run validation process before sending this of to the user.
        return self.validate_array(array)

    @t.overload
    async def _to_proto_impl(
        self, obj: ext.NpNDArray, *, version: t.Literal["v1"]
    ) -> pb.NDArray:
        ...

    @t.overload
    async def _to_proto_impl(
        self, obj: ext.NpNDArray, *, version: t.Literal["v1alpha1"]
    ) -> pb_v1alpha1.NDArray:
        ...

    async def _to_proto_impl(
        self, obj: ext.NpNDArray, *, version: str
    ) -> _message.Message:
        """
        Process given objects and convert it to grpc protobuf response.

        Args:
            obj: ``np.array`` that will be serialized to protobuf.
        Returns:
            ``pb.NDArray``:
                Protobuf representation of given ``np.ndarray``
        """
        try:
            obj = self.validate_array(obj)
        except BadInput as e:
            raise e from None

        pb, _ = import_generated_stubs(version)

        try:
            fieldpb = npdtype_to_fieldpb_map()[obj.dtype]
            dtypepb = npdtype_to_dtypepb_map(version=version)[obj.dtype]
            return pb.NDArray(
                dtype=dtypepb,
                shape=tuple(obj.shape),
                **{fieldpb: obj.ravel().tolist()},
            )
        except KeyError:
            raise BadInput(
                f"Unsupported dtype '{obj.dtype}' for response message.",
            ) from None

    def from_arrow(self, batch: pyarrow.RecordBatch) -> ext.NpNDArray:
        df = batch.to_pandas()
        if not LazyType("pandas", "DataFrame").isinstance(df):
            raise InvalidArgument("Unable to convert input into numpy ndarray.")
        return df.to_numpy()

    def to_arrow(self, arr: ext.NpNDArray) -> pyarrow.RecordBatch:
        import pyarrow

        return pyarrow.RecordBatch.from_arrays([pyarrow.array(arr)], names=["output"])

    def spark_schema(self) -> pyspark.sql.types.StructType:
        from pyspark.sql.types import StructType
        from pyspark.sql.types import StructField
        from pyspark.pandas.typedef import as_spark_type

        if self._dtype is None:
            raise InvalidArgument(
                "Cannot perform batch inference with a numpy output without a known dtype; please provide a dtype."
            )
        if self._shape is None:
            raise InvalidArgument(
                "Cannot perform batch inference with a numpy output without a known shape; please provide a shape."
            )
        if len(self._shape) != 1:
            raise InvalidArgument(
                "Cannot perform batch inference with a multidimensional numpy ndarray output; consider using pandas DataFrames instead."
            )

        try:
            out_spark_type = as_spark_type(self._dtype)
        except TypeError:
            raise InvalidArgument(
                f"dtype {self._dtype} is not supported for batch inference."
            )
        return StructType([StructField("out", out_spark_type, nullable=False)])

    async def to_proto(self, obj: ext.NpNDArray) -> pb.NDArray:
        return await self._to_proto_impl(obj, version="v1")

    async def to_proto_v1alpha1(self, obj: ext.NpNDArray) -> pb_v1alpha1.NDArray:
        return await self._to_proto_impl(obj, version="v1alpha1")
