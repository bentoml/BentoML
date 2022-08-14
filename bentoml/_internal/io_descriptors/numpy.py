from __future__ import annotations

import json
import typing as t
import logging
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

from .base import IODescriptor
from ..types import LazyType
from ..utils import LazyLoader
from ..utils.http import set_cookies
from ...exceptions import BadInput
from ...exceptions import BentoMLException
from ...exceptions import InternalServerError
from ...exceptions import UnprocessableEntity
from ..service.openapi import SUCCESS_DESCRIPTION
from ..service.openapi.specification import Schema
from ..service.openapi.specification import Response as OpenAPIResponse
from ..service.openapi.specification import MediaType
from ..service.openapi.specification import RequestBody

if TYPE_CHECKING:
    import numpy as np

    from bentoml.grpc.v1 import service_pb2 as pb
    from bentoml.grpc.types import BentoServicerContext

    from .. import external_typing as ext
    from ..context import InferenceApiContext as Context
else:
    np = LazyLoader("np", globals(), "numpy")
    pb = LazyLoader("pb", globals(), "bentoml.grpc.v1.service_pb2")

logger = logging.getLogger(__name__)


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
class NumpyNdarray(IODescriptor["ext.NpNDArray"], proto_field="ndarray"):
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
        bytesorder: t.Literal["C", "F", "A", None] = None,
    ):
        if dtype and not isinstance(dtype, np.dtype):
            # Convert from primitive type or type string, e.g.: np.dtype(float) or np.dtype("float64")
            try:
                dtype = np.dtype(dtype)
            except TypeError as e:
                raise UnprocessableEntity(f'Invalid dtype "{dtype}": {e}') from e

        self._dtype = dtype
        self._shape = shape
        self._enforce_dtype = enforce_dtype
        self._enforce_shape = enforce_shape

        self._sample_input = None

        if bytesorder and bytesorder not in ["C", "F", "A"]:
            raise BadInput(
                f"'bytesorder' must be one of ['C', 'F', 'A'], got {bytesorder} instead."
            )
        if not bytesorder:
            # https://numpy.org/doc/stable/user/basics.byteswapping.html#introduction-to-byte-ordering-and-ndarrays
            bytesorder = "C"  # default from numpy (C-order)

        self._bytesorder: t.Literal["C", "F", "A"] = bytesorder

        if self._enforce_dtype and not self._dtype:
            raise UnprocessableEntity(
                "'dtype' must be specified when 'enforce_dtype=True'"
            )
        if self._enforce_shape and not self._shape:
            raise UnprocessableEntity(
                "'shape' must be specified when 'enforce_shape=True'"
            )

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

    def validate_array(
        self,
        arr: ext.NpNDArray,
        dtype: ext.NpDTypeLike | None = None,
        shape: tuple[int, ...] | None = None,
        exception_cls: t.Type[Exception] = BadInput,
    ) -> ext.NpNDArray:
        if dtype is not None and dtype != arr.dtype:
            # ‘same_kind’ means only safe casts or casts within a kind, like float64
            # to float32, are allowed.
            if np.can_cast(arr.dtype, dtype, casting="same_kind"):
                arr = arr.astype(dtype, casting="same_kind")  # type: ignore
            else:
                msg = f'{self.__class__.__name__}: Expecting ndarray of dtype "{dtype}", but "{arr.dtype}" was received.'
                if self._enforce_dtype:
                    raise exception_cls(msg)
                else:
                    logger.debug(msg)

        if shape is not None and not _is_matched_shape(shape, arr.shape):
            msg = f'{self.__class__.__name__}: Expecting ndarray of shape "{shape}", but "{arr.shape}" was received.'
            if self._enforce_shape:
                raise exception_cls(msg)
            try:
                arr = arr.reshape(shape)
            except ValueError as e:
                logger.debug(f"{msg} Failed to reshape: {e}.")

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
        return self.validate_array(res, dtype=self._dtype, shape=self._shape)

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
        obj = self.validate_array(
            obj, dtype=self._dtype, shape=self._shape, exception_cls=InternalServerError
        )
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
                "'NumpyNdarray.from_sample()' expects a 'numpy.array', not 'numpy.generic'."
            )

        klass = cls(
            dtype=sample_input.dtype,
            shape=sample_input.shape,
            enforce_dtype=enforce_dtype,
            enforce_shape=enforce_shape,
        )
        klass.sample_input = sample_input

        return klass

    async def from_grpc_request(
        self, request: pb.Request, context: BentoServicerContext
    ) -> ext.NpNDArray:
        """
        Process incoming protobuf request and convert it to ``numpy.ndarray``

        Args:
            request: Incoming RPC request message.
            context: grpc.ServicerContext

        Returns:
            a ``numpy.ndarray`` object. This can then be used
             inside users defined logics.
        """
        from bentoml.grpc.utils import get_field
        from bentoml.grpc.utils import raise_grpc_exception
        from bentoml.grpc.utils import validate_content_type
        from bentoml.grpc.utils.mapping import dtypepb_to_fieldpb_map
        from bentoml.grpc.utils.mapping import dtypepb_to_npdtype_map
        from bentoml.grpc.utils.mapping import fieldpb_to_npdtype_map
        from bentoml.grpc.utils.mapping import npdtype_to_fieldpb_map

        # validate gRPC content type if content type is specified
        validate_content_type(context, self)

        field = get_field(request, self)

        # The logic behind processing shape is as follows:
        # 1. If 'shape' is specified in the request message, then:
        #    - if 'shape' is specified inside IO Descriptor, we will omit this value unless 'enforce_shape' is True.
        #    - Otherwise, use 'shape' from message
        #
        # 2. If the 'shape' is not specified inside the request message, then:
        #    - If 'shape' is not defined at the IO descriptor level, 'shape' value will be omitted unless 'enforce_shape' is True.
        #    - Otherwise, we will use shape from the IO descriptor.
        if self._enforce_shape:
            logger.debug(
                f"'enforce_shape={self._enforce_shape}', using '{self._shape}'..."
            )
            shape = self._shape
        else:
            shapepb = field.shape  # this can be an empty list ([])
            if shapepb:
                shape = tuple(shapepb)
                if self._shape:
                    logger.warning(
                        f"'shape={self._shape}' is ignored when 'shape' is specified in request message. Using '{shape}' from request message..."
                    )
            else:
                shape = self._shape

        # The same logic for 'shape' applies to 'dtype' with the additional:
        #
        # 2a. If 'dtype' message is not specified, then we will infer given numpy dtype from the entry.
        if self._enforce_dtype:
            logger.debug(
                f"'enforce_dtype={self._enforce_dtype}', using '{self._dtype}'..."
            )
            dtype = self._dtype
            field_values = npdtype_to_fieldpb_map()[dtype]

            values_arr = getattr(field, field_values, None)
            if not values_arr:
                raise_grpc_exception(
                    f"field '{field_values}' (required for '{dtype}') is missing.",
                    context=context,
                    exception_cls=UnprocessableEntity,
                )
        else:
            dtypepb = field.dtype
            if dtypepb:
                if dtypepb == pb.NDArray.DTYPE_UNSPECIFIED:
                    dtype = None
                else:
                    dtype = dtypepb_to_npdtype_map()[dtypepb]

                if self._dtype:
                    logger.warning(
                        f"'dtype={self._dtype}' is ignored when 'dtype' is specified in request message. Using '{dtype}' from request message..."
                    )

                values_arr = getattr(field, dtypepb_to_fieldpb_map()[dtypepb])

                if not dtype:
                    # This is the case where 'dtype' proto uses DTYPE_UNSPECIFIED
                    return np.empty(shape=shape or 0)
            else:
                fieldpb = [
                    f.name for f, _ in field.ListFields() if f.name.endswith("_values")
                ]
                if len(fieldpb) != 1:
                    raise_grpc_exception(
                        f"Array contents can only be one of given values key. Use one of {fieldpb} instead.",
                        context=context,
                        exception_cls=BadInput,
                    )
                dtype = fieldpb_to_npdtype_map()[fieldpb[0]]
                values_arr = getattr(field, fieldpb[0])

        if shape and dtype in [np.float32, np.double, np.bool_]:
            buffer = field.SerializeToString()
            num_entries = np.prod(shape)
            return np.frombuffer(
                memoryview(buffer[-(dtype.itemsize * num_entries) :]),
                dtype=dtype,
                count=num_entries,
                offset=0,
            ).reshape(shape)
        else:
            try:
                res = np.array(values_arr, dtype=dtype)
            except ValueError:
                res = np.array(values_arr)

            return self.validate_array(res, dtype=dtype, shape=shape)

    async def to_grpc_response(
        self, obj: ext.NpNDArray, context: BentoServicerContext
    ) -> pb.Response:
        """
        Process given objects and convert it to grpc protobuf response.

        Args:
            obj: `np.ndarray` that will be serialized to protobuf
            context: grpc.aio.ServicerContext from grpc.aio.Server
        Returns:
            `io_descriptor_pb2.Array`:
                Protobuf representation of given `np.ndarray`
        """
        from bentoml.grpc.utils import raise_grpc_exception
        from bentoml.grpc.utils.mapping import npdtype_to_dtypepb_map
        from bentoml.grpc.utils.mapping import npdtype_to_fieldpb_map

        try:
            obj = self.validate_array(
                obj,
                dtype=self._dtype,
                shape=self._shape,
                exception_cls=InternalServerError,
            )
        except InternalServerError as e:
            raise_grpc_exception(e.message, context=context, exception_cls=e.__class__)

        context.set_trailing_metadata((("content-type", self.grpc_content_type),))

        try:
            fieldpb = npdtype_to_fieldpb_map()[obj.dtype]
            dtypepb = npdtype_to_dtypepb_map()[obj.dtype]
            return pb.Response(
                ndarray=pb.NDArray(
                    dtype=dtypepb,
                    shape=tuple(obj.shape),
                    **{fieldpb: obj.ravel(order=self._bytesorder).tolist()},
                )
            )
        except KeyError:
            raise_grpc_exception(
                f"Unsupported dtype '{obj.dtype}' for response message.",
                context=context,
            )
