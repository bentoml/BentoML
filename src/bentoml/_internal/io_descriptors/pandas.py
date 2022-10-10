from __future__ import annotations

import io
import typing as t
import logging
import functools
from enum import Enum
from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

from starlette.requests import Request
from starlette.responses import Response

from .base import IODescriptor
from ..types import LazyType
from ..utils.pkg import find_spec
from ..utils.http import set_cookies
from ...exceptions import BadInput
from ...exceptions import InvalidArgument
from ...exceptions import UnprocessableEntity
from ...exceptions import MissingDependencyException
from ..service.openapi import SUCCESS_DESCRIPTION
from ..utils.lazy_loader import LazyLoader
from ..service.openapi.specification import Schema
from ..service.openapi.specification import Response as OpenAPIResponse
from ..service.openapi.specification import MediaType
from ..service.openapi.specification import RequestBody

if TYPE_CHECKING:
    import pandas as pd

    from bentoml.grpc.v1alpha1 import service_pb2 as pb

    from .. import external_typing as ext
    from ..context import InferenceApiContext as Context

else:
    from bentoml.grpc.utils import import_generated_stubs

    pb, _ = import_generated_stubs()
    pd = LazyLoader(
        "pd",
        globals(),
        "pandas",
        exc_msg='pandas" is required to use PandasDataFrame or PandasSeries. Install with "pip install -U pandas"',
    )

logger = logging.getLogger(__name__)


# Check for parquet support
@functools.lru_cache(maxsize=1)
def get_parquet_engine() -> str:
    if find_spec("pyarrow") is not None:
        return "pyarrow"
    elif find_spec("fastparquet") is not None:
        return "fastparquet"
    else:
        logger.warning(
            "Neither pyarrow nor fastparquet packages found. Parquet de/serialization will not be available."
        )
        raise MissingDependencyException(
            "Parquet serialization is not available. Try installing pyarrow or fastparquet first."
        )


def _openapi_types(item: str) -> str:  # pragma: no cover
    # convert pandas types to OpenAPI types
    if item.startswith("int"):
        return "integer"
    elif item.startswith("float") or item.startswith("double"):
        return "number"
    elif item.startswith("str") or item.startswith("date"):
        return "string"
    elif item.startswith("bool"):
        return "boolean"
    else:
        return "object"


def _dataframe_openapi_schema(
    dtype: bool | ext.PdDTypeArg | None, orient: ext.DataFrameOrient = None
) -> Schema:  # pragma: no cover
    if isinstance(dtype, dict):
        if orient == "records":
            return Schema(
                type="array",
                items=Schema(
                    type="object",
                    properties={
                        k: Schema(type=_openapi_types(v)) for k, v in dtype.items()
                    },
                ),
            )
        if orient == "index":
            return Schema(
                type="object",
                additionalProperties=Schema(
                    type="object",
                    properties={
                        k: Schema(type=_openapi_types(v)) for k, v in dtype.items()
                    },
                ),
            )
        if orient == "columns":
            return Schema(
                type="object",
                properties={
                    k: Schema(
                        type="object",
                        additionalProperties=Schema(type=_openapi_types(v)),
                    )
                    for k, v in dtype.items()
                },
            )

        return Schema(
            type="object",
            properties={
                k: Schema(type="array", items=Schema(type=_openapi_types(v)))
                for k, v in dtype.items()
            },
        )
    else:
        return Schema(type="object")


def _series_openapi_schema(
    dtype: bool | ext.PdDTypeArg | None, orient: ext.SeriesOrient = None
) -> Schema:  # pragma: no cover
    if isinstance(dtype, str):
        if orient in ["index", "values"]:
            return Schema(
                type="object", additionalProperties=Schema(type=_openapi_types(dtype))
            )
        if orient in ["records", "columns"]:
            return Schema(type="array", items=Schema(type=_openapi_types(dtype)))
    return Schema(type="object")


class SerializationFormat(Enum):
    JSON = "application/json"
    PARQUET = "application/octet-stream"
    CSV = "text/csv"

    def __init__(self, mime_type: str):
        self.mime_type = mime_type


def _infer_serialization_format_from_request(
    request: Request, default_format: SerializationFormat
) -> SerializationFormat:
    """Determine the serialization format from the request's headers['content-type']"""

    content_type = request.headers.get("content-type")

    if content_type == "application/json":
        return SerializationFormat.JSON
    elif content_type == "application/octet-stream":
        return SerializationFormat.PARQUET
    elif content_type == "text/csv":
        return SerializationFormat.CSV
    elif content_type:
        logger.debug(
            f"Unknown content-type ('{content_type}'), falling back to '{default_format}' serialization format.",
        )
        return default_format
    else:
        logger.debug(
            f"Content-type not specified, falling back to '{default_format}' serialization format.",
        )
        return default_format


def _validate_serialization_format(serialization_format: SerializationFormat):
    if (
        serialization_format is SerializationFormat.PARQUET
        and get_parquet_engine() is None
    ):
        raise MissingDependencyException(
            "Parquet serialization is not available. Try installing pyarrow or fastparquet first."
        )


class PandasDataFrame(IODescriptor["ext.PdDataFrame"]):
    """
    :obj:`PandasDataFrame` defines API specification for the inputs/outputs of a Service,
    where either inputs will be converted to or outputs will be converted from type
    :code:`pd.DataFrame` as specified in your API function signature.

    A sample service implementation:

    .. code-block:: python
       :caption: `service.py`

       from __future__ import annotations

       import bentoml
       import pandas as pd
       import numpy as np
       from bentoml.io import PandasDataFrame

       input_spec = PandasDataFrame.from_sample(pd.DataFrame(np.array([[5,4,3,2]])))

       runner = bentoml.sklearn.get("sklearn_model_clf").to_runner()

       svc = bentoml.Service("iris-classifier", runners=[runner])

       @svc.api(input=input_spec, output=PandasDataFrame())
       def predict(input_arr):
           res = runner.run(input_arr)
           return pd.DataFrame(res)

    Users then can then serve this service with :code:`bentoml serve`:

    .. code-block:: bash

        % bentoml serve ./service.py:svc --reload

    Users can then send requests to the newly started services with any client:

    .. tab-set::

        .. tab-item:: Bash

            .. code-block:: bash

                % curl -X POST -H "Content-Type: application/json" \\
                        --data '[{"0":5,"1":4,"2":3,"3":2}]' http://0.0.0.0:3000/predict

                # [{"0": 1}]%

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
        orient: Indication of expected JSON string format. Compatible JSON strings can be
                produced by :func:`pandas.io.json.to_json()` with a corresponding orient value.
                Possible orients are:

                - :obj:`split` - :code:`dict[str, Any]` ↦ {``idx`` ↠ ``[idx]``, ``columns`` ↠ ``[columns]``, ``data`` ↠ ``[values]``}
                - :obj:`records` - :code:`list[Any]` ↦ [{``column`` ↠ ``value``}, ..., {``column`` ↠ ``value``}]
                - :obj:`index` - :code:`dict[str, Any]` ↦ {``idx`` ↠ {``column`` ↠ ``value``}}
                - :obj:`columns` - :code:`dict[str, Any]` ↦ {``column`` ↠ {``index`` ↠ ``value``}}
                - :obj:`values` - :code:`dict[str, Any]` ↦ Values arrays
        columns: List of columns name that users wish to update.
        apply_column_names: Whether to update incoming DataFrame columns. If :code:`apply_column_names=True`,
                            then ``columns`` must be specified.
        dtype: Data type users wish to convert their inputs/outputs to. If it is a boolean,
               then pandas will infer dtypes. Else if it is a dictionary of column to
               ``dtype``, then applies those to incoming dataframes. If ``False``, then don't
               infer dtypes at all (only applies to the data). This is not applicable for :code:`orient='table'`.
        enforce_dtype: Whether to enforce a certain data type. if :code:`enforce_dtype=True` then :code:`dtype` must be specified.
        shape: Optional shape check that users can specify for their incoming HTTP
               requests. We will only check the number of columns you specified for your
               given shape:

               .. code-block:: python
                  :caption: `service.py`


                  import pandas as pd
                  from bentoml.io import PandasDataFrame

                  df = pd.DataFrame([[1, 2, 3]])  # shape (1,3)
                  inp = PandasDataFrame.from_sample(df)

                  @svc.api(
                      input=PandasDataFrame(shape=(51, 10),
                            enforce_shape=True),
                      output=PandasDataFrame()
                  )
                  def predict(input_df: pd.DataFrame) -> pd.DataFrame:
                      # if input_df have shape (40,9),
                      # it will throw out errors
                      ...
        enforce_shape: Whether to enforce a certain shape. If ``enforce_shape=True`` then ``shape`` must be specified.
        default_format: The default serialization format to use if the request does not specify a ``Content-Type`` Headers.
                        It is also the serialization format used for the response. Possible values are:

                        - :obj:`json` - JSON text format (inferred from content-type ``"application/json"``)
                        - :obj:`parquet` - Parquet binary format (inferred from content-type ``"application/octet-stream"``)
                        - :obj:`csv` - CSV text format (inferred from content-type ``"text/csv"``)

    Returns:
        :obj:`PandasDataFrame`: IO Descriptor that represents a :code:`pd.DataFrame`.
    """

    _proto_fields = ("dataframe",)

    def __init__(
        self,
        orient: ext.DataFrameOrient = "records",
        columns: list[str] | None = None,
        apply_column_names: bool = False,
        dtype: bool | ext.PdDTypeArg | None = None,
        enforce_dtype: bool = False,
        shape: tuple[int, ...] | None = None,
        enforce_shape: bool = False,
        default_format: t.Literal["json", "parquet", "csv"] = "json",
    ):
        self._orient = orient
        self._columns = columns
        self._apply_column_names = apply_column_names
        self._dtype = dtype
        self._enforce_dtype = enforce_dtype
        self._shape = shape
        self._enforce_shape = enforce_shape
        self._default_format = SerializationFormat[default_format.upper()]

        _validate_serialization_format(self._default_format)
        self._mime_type = self._default_format.mime_type

        self._sample_input = None

    @property
    def sample_input(self) -> ext.PdDataFrame | None:
        return self._sample_input

    @sample_input.setter
    def sample_input(self, value: ext.PdDataFrame) -> None:
        self._sample_input = value

    def input_type(self) -> LazyType[ext.PdDataFrame]:
        return LazyType("pandas", "DataFrame")

    def openapi_schema(self) -> Schema:
        return _dataframe_openapi_schema(self._dtype, self._orient)

    def openapi_components(self) -> dict[str, t.Any] | None:
        pass

    def openapi_request_body(self) -> RequestBody:
        return RequestBody(
            content={self._mime_type: MediaType(schema=self.openapi_schema())},
            required=True,
        )

    def openapi_responses(self) -> OpenAPIResponse:
        return OpenAPIResponse(
            description=SUCCESS_DESCRIPTION,
            content={self._mime_type: MediaType(schema=self.openapi_schema())},
        )

    async def from_http_request(self, request: Request) -> ext.PdDataFrame:
        """
        Process incoming requests and convert incoming
        objects to `pd.DataFrame`

        Args:
            request (`starlette.requests.Requests`):
                Incoming Requests
        Returns:
            a `pd.DataFrame` object. This can then be used
             inside users defined logics.
        Raises:
            BadInput:
                Raised when the incoming requests are bad formatted.
        """

        serialization_format = _infer_serialization_format_from_request(
            request, self._default_format
        )
        _validate_serialization_format(serialization_format)

        obj = await request.body()
        if serialization_format is SerializationFormat.JSON:
            assert not isinstance(self._dtype, bool)
            res = pd.read_json(io.BytesIO(obj), dtype=self._dtype, orient=self._orient)
        elif serialization_format is SerializationFormat.PARQUET:
            res = pd.read_parquet(io.BytesIO(obj), engine=get_parquet_engine())
        elif serialization_format is SerializationFormat.CSV:
            assert not isinstance(self._dtype, bool)
            res: ext.PdDataFrame = pd.read_csv(io.BytesIO(obj), dtype=self._dtype)
        else:
            raise InvalidArgument(
                f"Unknown serialization format ({serialization_format})."
            ) from None

        assert isinstance(res, pd.DataFrame)
        return self.validate_dataframe(res)

    async def to_http_response(
        self, obj: ext.PdDataFrame, ctx: Context | None = None
    ) -> Response:
        """
        Process given objects and convert it to HTTP response.

        Args:
            obj (`pd.DataFrame`):
                `pd.DataFrame` that will be serialized to JSON or parquet
        Returns:
            HTTP Response of type `starlette.responses.Response`. This can
             be accessed via cURL or any external web traffic.
        """
        obj = self.validate_dataframe(obj)

        # For the response it doesn't make sense to enforce the same serialization format as specified
        # by the request's headers['content-type']. Instead we simply use the _default_format.
        serialization_format = self._default_format

        if not LazyType["ext.PdDataFrame"](pd.DataFrame).isinstance(obj):
            raise InvalidArgument(
                f"return object is not of type `pd.DataFrame`, got type {type(obj)} instead"
            )
        if serialization_format is SerializationFormat.JSON:
            resp = obj.to_json(orient=self._orient)
        elif serialization_format is SerializationFormat.PARQUET:
            resp = obj.to_parquet(engine=get_parquet_engine())
        elif serialization_format is SerializationFormat.CSV:
            resp = obj.to_csv()
        else:
            raise InvalidArgument(
                f"Unknown serialization format ({serialization_format})."
            ) from None

        if ctx is not None:
            res = Response(
                resp,
                media_type=serialization_format.mime_type,
                headers=ctx.response.headers,  # type: ignore (bad starlette types)
                status_code=ctx.response.status_code,
            )
            set_cookies(res, ctx.response.cookies)
            return res
        else:
            return Response(resp, media_type=serialization_format.mime_type)

    @classmethod
    def from_sample(
        cls,
        sample_input: ext.PdDataFrame,
        orient: ext.DataFrameOrient = "records",
        apply_column_names: bool = True,
        enforce_shape: bool = True,
        enforce_dtype: bool = True,
        default_format: t.Literal["json", "parquet", "csv"] = "json",
    ) -> PandasDataFrame:
        """
        Create a :obj:`PandasDataFrame` IO Descriptor from given inputs.

        Args:
            sample_input: Given sample ``pd.DataFrame`` data
            orient: Indication of expected JSON string format. Compatible JSON strings can be
                    produced by :func:`pandas.io.json.to_json()` with a corresponding orient value.
                    Possible orients are:

                    - :obj:`split` - :code:`dict[str, Any]` ↦ {``idx`` ↠ ``[idx]``, ``columns`` ↠ ``[columns]``, ``data`` ↠ ``[values]``}
                    - :obj:`records` - :code:`list[Any]` ↦ [{``column`` ↠ ``value``}, ..., {``column`` ↠ ``value``}]
                    - :obj:`index` - :code:`dict[str, Any]` ↦ {``idx`` ↠ {``column`` ↠ ``value``}}
                    - :obj:`columns` - :code:`dict[str, Any]` ↦ {``column`` ↠ {``index`` ↠ ``value``}}
                    - :obj:`values` - :code:`dict[str, Any]` ↦ Values arrays
            apply_column_names: Update incoming DataFrame columns. ``columns`` must be specified at
                                function signature. If you don't want to enforce a specific columns
                                name then change ``apply_column_names=False``.
            enforce_dtype: Enforce a certain data type. `dtype` must be specified at function
                           signature. If you don't want to enforce a specific dtype then change
                           ``enforce_dtype=False``.
            enforce_shape: Enforce a certain shape. ``shape`` must be specified at function
                           signature. If you don't want to enforce a specific shape then change
                           ``enforce_shape=False``.
            default_format: The default serialization format to use if the request does not specify a ``Content-Type`` Headers.
                            It is also the serialization format used for the response. Possible values are:

                            - :obj:`json` - JSON text format (inferred from content-type ``"application/json"``)
                            - :obj:`parquet` - Parquet binary format (inferred from content-type ``"application/octet-stream"``)
                            - :obj:`csv` - CSV text format (inferred from content-type ``"text/csv"``)

        Returns:
            :obj:`PandasDataFrame`: :code:`PandasDataFrame` IODescriptor from given users inputs.

        Example:

        .. code-block:: python
           :caption: `service.py`

           import pandas as pd
           from bentoml.io import PandasDataFrame
           arr = [[1,2,3]]
           input_spec = PandasDataFrame.from_sample(pd.DataFrame(arr))

           @svc.api(input=input_spec, output=PandasDataFrame())
           def predict(inputs: pd.DataFrame) -> pd.DataFrame: ...
        """
        inst = cls(
            orient=orient,
            enforce_shape=enforce_shape,
            shape=sample_input.shape,
            apply_column_names=apply_column_names,
            columns=[str(x) for x in list(sample_input.columns)],
            enforce_dtype=enforce_dtype,
            dtype=True,  # set to True to infer from given input
            default_format=default_format,
        )
        inst.sample_input = sample_input

        return inst

    def validate_dataframe(
        self, dataframe: ext.PdDataFrame, exception_cls: t.Type[Exception] = BadInput
    ) -> ext.PdDataFrame:

        if not LazyType["ext.PdDataFrame"]("pd.DataFrame").isinstance(dataframe):
            raise InvalidArgument(
                f"return object is not of type 'pd.DataFrame', got type '{type(dataframe)}' instead"
            ) from None

        # TODO: dtype check
        # if self._dtype is not None and self._dtype != dataframe.dtypes:
        #     msg = f'{self.__class__.__name__}: Expecting DataFrame of dtype "{self._dtype}", but "{dataframe.dtypes}" was received.'
        #     if self._enforce_dtype:
        #         raise exception_cls(msg) from None

        if self._columns is not None and len(self._columns) != dataframe.shape[1]:
            msg = f"length of 'columns' ({len(self._columns)}) does not match the # of columns of incoming data."
            if self._apply_column_names:
                raise BadInput(msg) from None
            else:
                logger.debug(msg)
                dataframe.columns = pd.Index(self._columns)

        # TODO: convert from wide to long format (melt())
        if self._shape is not None and self._shape != dataframe.shape:
            msg = f'{self.__class__.__name__}: Expecting DataFrame of shape "{self._shape}", but "{dataframe.shape}" was received.'
            if self._enforce_shape and not all(
                left == right
                for left, right in zip(self._shape, dataframe.shape)
                if left != -1 and right != -1
            ):
                raise exception_cls(msg) from None

        return dataframe

    async def from_proto(self, field: pb.DataFrame | bytes) -> ext.PdDataFrame:
        """
        Process incoming protobuf request and convert it to ``pandas.DataFrame``

        Args:
            request: Incoming RPC request message.
            context: grpc.ServicerContext

        Returns:
            a ``pandas.DataFrame`` object. This can then be used
             inside users defined logics.
        """
        # TODO: support different serialization format
        if isinstance(field, bytes):
            # TODO: handle serialized_bytes for dataframe
            raise NotImplementedError(
                'Currently not yet implemented. Use "dataframe" instead.'
            )
        else:
            # note that there is a current bug where we don't check for
            # dtype of given fields per Series to match with types of a given
            # columns, hence, this would result in a wrong DataFrame that is not
            # expected by our users.
            assert isinstance(field, pb.DataFrame)
            # columns orient: { column_name : {index : columns.series._value}}
            if self._orient != "columns":
                raise BadInput(
                    f"'dataframe' field currently only supports 'columns' orient. Make sure to set 'orient=columns' in {self.__class__.__name__}."
                ) from None
            data: list[t.Any] = []

            def process_columns_contents(content: pb.Series) -> dict[str, t.Any]:
                # To be use inside a ThreadPoolExecutor to handle
                # large tabular data
                if len(content.ListFields()) != 1:
                    raise BadInput(
                        f"Array contents can only be one of given values key. Use one of '{list(map(lambda f: f[0].name,content.ListFields()))}' instead."
                    ) from None
                return {str(i): c for i, c in enumerate(content.ListFields()[0][1])}

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = executor.map(process_columns_contents, field.columns)
                data.extend([i for i in list(futures)])
            dataframe = pd.DataFrame(
                dict(zip(field.column_names, data)),
                columns=t.cast(t.List[str], field.column_names),
            )
        return self.validate_dataframe(dataframe)

    async def to_proto(self, obj: ext.PdDataFrame) -> pb.DataFrame:
        """
        Process given objects and convert it to grpc protobuf response.

        Args:
            obj: ``pandas.DataFrame`` that will be serialized to protobuf
            context: grpc.aio.ServicerContext from grpc.aio.Server
        Returns:
            ``service_pb2.Response``:
                Protobuf representation of given ``pandas.DataFrame``
        """
        from bentoml._internal.io_descriptors.numpy import npdtype_to_fieldpb_map

        # TODO: support different serialization format
        obj = self.validate_dataframe(obj)
        mapping = npdtype_to_fieldpb_map()
        # note that this is not safe, since we are not checking the dtype of the series
        # FIXME(aarnphm): validate and handle mix columns dtype
        # currently we don't support ExtensionDtype
        columns_name: list[str] = list(map(str, obj.columns))
        not_supported: list[ext.PdDType] = list(
            filter(
                lambda x: x not in mapping,
                map(lambda x: t.cast("ext.PdSeries", obj[x]).dtype, columns_name),
            )
        )
        if len(not_supported) > 0:
            raise UnprocessableEntity(
                f'dtype in column "{obj.columns}" is not currently supported.'
            ) from None
        return pb.DataFrame(
            column_names=columns_name,
            columns=[
                pb.Series(
                    **{mapping[t.cast("ext.NpDTypeLike", obj[col].dtype)]: obj[col]}
                )
                for col in columns_name
            ],
        )


class PandasSeries(IODescriptor["ext.PdSeries"]):
    """
    :code:`PandasSeries` defines API specification for the inputs/outputs of a Service, where
    either inputs will be converted to or outputs will be converted from type
    :code:`pd.Series` as specified in your API function signature.

    A sample service implementation:

    .. code-block:: python
       :caption: `service.py`

        import bentoml
        import pandas as pd
        import numpy as np
        from bentoml.io import PandasSeries

        runner = bentoml.sklearn.get("sklearn_model_clf").to_runner()

        svc = bentoml.Service("iris-classifier", runners=[runner])

        @svc.api(input=PandasSeries(), output=PandasSeries())
        def predict(input_arr):
            res = runner.run(input_arr)  # type: np.ndarray
            return pd.Series(res)

    Users then can then serve this service with :code:`bentoml serve`:

    .. code-block:: bash

        % bentoml serve ./service.py:svc --reload

    Users can then send requests to the newly started services with any client:

    .. tab-set::

        .. tab-item:: Bash

            .. code-block:: bash

                % curl -X POST -H "Content-Type: application/json" \\
                        --data '[{"0":5,"1":4,"2":3,"3":2}]' http://0.0.0.0:3000/predict

                # [{"0": 1}]%

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
        orient: Indication of expected JSON string format. Compatible JSON strings can be
                produced by :func:`pandas.io.json.to_json()` with a corresponding orient value.
                Possible orients are:

                - :obj:`split` - :code:`dict[str, Any]` ↦ {``idx`` ↠ ``[idx]``, ``columns`` ↠ ``[columns]``, ``data`` ↠ ``[values]``}
                - :obj:`records` - :code:`list[Any]` ↦ [{``column`` ↠ ``value``}, ..., {``column`` ↠ ``value``}]
                - :obj:`index` - :code:`dict[str, Any]` ↦ {``idx`` ↠ {``column`` ↠ ``value``}}
                - :obj:`columns` - :code:`dict[str, Any]` ↦ {``column`` ↠ {``index`` ↠ ``value``}}
                - :obj:`values` - :code:`dict[str, Any]` ↦ Values arrays
        columns: List of columns name that users wish to update.
        apply_column_names (`bool`, `optional`, default to :code:`False`):
        apply_column_names: Whether to update incoming DataFrame columns. If :code:`apply_column_names=True`,
                            then ``columns`` must be specified.
        dtype: Data type users wish to convert their inputs/outputs to. If it is a boolean,
               then pandas will infer dtypes. Else if it is a dictionary of column to
               ``dtype``, then applies those to incoming dataframes. If ``False``, then don't
               infer dtypes at all (only applies to the data). This is not applicable for :code:`orient='table'`.
        enforce_dtype: Whether to enforce a certain data type. if :code:`enforce_dtype=True` then :code:`dtype` must be specified.
        shape: Optional shape check that users can specify for their incoming HTTP
               requests. We will only check the number of columns you specified for your
               given shape:

               .. code-block:: python
                  :caption: `service.py`

                  import pandas as pd
                  from bentoml.io import PandasSeries

                  @svc.api(input=PandasSeries(shape=(51,10), enforce_shape=True), output=PandasSeries())
                  def infer(input_series: pd.Series) -> pd.Series:
                  # if input_series have shape (40,9), it will throw out errors
                        ...
        enforce_shape: Whether to enforce a certain shape. If ``enforce_shape=True`` then ``shape`` must be specified.

    Returns:
        :obj:`PandasSeries`: IO Descriptor that represents a :code:`pd.Series`.
    """

    _proto_fields = ("series",)
    _mime_type = "application/json"

    def __init__(
        self,
        orient: ext.SeriesOrient = "records",
        dtype: ext.PdDTypeArg | None = None,
        enforce_dtype: bool = False,
        shape: tuple[int, ...] | None = None,
        enforce_shape: bool = False,
    ):
        self._orient = orient
        self._dtype = dtype
        self._enforce_dtype = enforce_dtype
        self._shape = shape
        self._enforce_shape = enforce_shape
        # TODO: support parquet for serde pd.Series

    def input_type(self) -> LazyType[ext.PdSeries]:
        return LazyType("pandas", "Series")

    def openapi_schema(self) -> Schema:
        return _series_openapi_schema(self._dtype, self._orient)

    def openapi_components(self) -> dict[str, t.Any] | None:
        pass

    def openapi_request_body(self) -> RequestBody:
        return RequestBody(
            content={self._mime_type: MediaType(schema=self.openapi_schema())},
            required=True,
        )

    def openapi_responses(self) -> OpenAPIResponse:
        return OpenAPIResponse(
            description=SUCCESS_DESCRIPTION,
            content={self._mime_type: MediaType(schema=self.openapi_schema())},
        )

    async def from_http_request(self, request: Request) -> ext.PdSeries:
        """
        Process incoming requests and convert incoming objects to ``pd.Series``.

        Args:
            request: Incoming Requests
        Returns:
            a ``pd.Series`` object. This can then be used inside users defined logics.
        """
        obj = await request.body()
        res: ext.PdSeries = pd.read_json(
            io.BytesIO(obj),
            typ="series",
            orient=self._orient,
            dtype=self._dtype,
        )
        return self.validate_series(res)

    async def to_http_response(
        self, obj: t.Any, ctx: Context | None = None
    ) -> Response:
        """
        Process given objects and convert it to HTTP response.

        Args:
            obj: `pd.Series` that will be serialized to JSON
        Returns:
            HTTP Response of type ``starlette.responses.Response``. This can be accessed via cURL or any external web traffic.
        """
        obj = self.validate_series(obj)
        if ctx is not None:
            res = Response(
                obj.to_json(orient=self._orient),
                media_type=self._mime_type,
                headers=ctx.response.headers,  # type: ignore (bad starlette types)
                status_code=ctx.response.status_code,
            )
            set_cookies(res, ctx.response.cookies)
            return res
        else:
            return Response(
                obj.to_json(orient=self._orient), media_type=self._mime_type
            )

    def validate_series(
        self, series: ext.PdSeries, exception_cls: t.Type[Exception] = BadInput
    ) -> ext.PdSeries:
        # TODO: dtype check
        if not LazyType["ext.PdSeries"]("pd.Series").isinstance(series):
            raise InvalidArgument(
                f"return object is not of type 'pd.Series', got type '{type(series)}' instead"
            ) from None
        # TODO: convert from wide to long format (melt())
        if self._shape is not None and self._shape != series.shape:
            msg = f"{self.__class__.__name__}: Expecting Series of shape '{self._shape}', but '{series.shape}' was received."
            if self._enforce_shape and not all(
                left == right
                for left, right in zip(self._shape, series.shape)
                if left != -1 and right != -1
            ):
                raise exception_cls(msg) from None

        return series

    async def from_proto(self, field: pb.Series | bytes) -> ext.PdSeries:
        raise NotImplementedError("Currently not yet implemented.")

    async def to_proto(self, obj: ext.PdSeries) -> pb.Series:
        raise NotImplementedError("Currently not yet implemented.")
