import io
import typing as t
import logging
import functools
import importlib.util
from enum import Enum
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

from .base import IODescriptor
from .json import MIME_TYPE_JSON
from ..types import LazyType
from ...exceptions import BadInput
from ...exceptions import InvalidArgument
from ...exceptions import MissingDependencyException
from ..utils.lazy_loader import LazyLoader

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import]

    from .. import external_typing as ext

else:
    pd = LazyLoader(
        "pd",
        globals(),
        "pandas",
        exc_msg="`pandas` is required to use PandasDataFrame or PandasSeries. Install with `pip install -U pandas`",
    )

logger = logging.getLogger(__name__)


# Check for parquet support
@functools.lru_cache(maxsize=1)
def get_parquet_engine():
    if importlib.util.find_spec("pyarrow") is not None:
        return "pyarrow"
    elif importlib.util.find_spec("fastparquet") is not None:
        return "fastparquet"
    else:
        logger.warning(
            "Neither pyarrow nor fastparquet packages found. Parquet de/serialization will not be available."
        )
        return None


def _infer_type(item: str) -> str:  # pragma: no cover
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


def _schema_type(
    dtype: t.Optional[t.Union[bool, t.Dict[str, t.Any]]]
) -> t.Dict[str, t.Any]:  # pragma: no cover
    if isinstance(dtype, dict):
        return {
            "type": "object",
            "properties": {
                k: {"type": "array", "items": {"type": _infer_type(v)}}
                for k, v in dtype.items()
            },
        }
    else:
        return {"type": "object"}


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
            "Unknown content-type (%s), falling back to %s serialization format.",
            content_type,
            default_format,
        )
        return default_format
    else:
        logger.debug(
            "Content-type not specified, falling back to %s serialization format.",
            default_format,
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
    :code:`PandasDataFrame` defines API specification for the inputs/outputs of a Service,
    where either inputs will be converted to or outputs will be converted from type
    :code:`numpy.ndarray` as specified in your API function signature.

    Sample implementation of a sklearn service:

    .. code-block:: python

        # sklearn_svc.py
        import bentoml
        import pandas as pd
        import numpy as np
        from bentoml.io import PandasDataFrame
        import bentoml.sklearn

        input_spec = PandasDataFrame.from_sample(pd.DataFrame(np.array([[5,4,3,2]])))

        runner = bentoml.sklearn.load_runner("sklearn_model_clf")

        svc = bentoml.Service("iris-classifier", runners=[runner])

        @svc.api(input=input_spec, output=PandasDataFrame())
        def predict(input_arr):
            res = runner.run_batch(input_arr)  # type: np.ndarray
            return pd.DataFrame(res)

    Users then can then serve this service with :code:`bentoml serve`:

    .. code-block:: bash

        % bentoml serve ./sklearn_svc.py:svc --auto-reload

        (Press CTRL+C to quit)
        [INFO] Starting BentoML API server in development mode with auto-reload enabled
        [INFO] Serving BentoML Service "iris-classifier" defined in "sklearn_svc.py"
        [INFO] API Server running on http://0.0.0.0:3000

    Users can then send requests to the newly started services with any client:

    .. tabs::

        .. code-tab:: python

            import requests
            requests.post(
                "http://0.0.0.0:3000/predict",
                headers={"content-type": "application/json"},
                data='[{"0":5,"1":4,"2":3,"3":2}]'
            ).text

        .. code-tab:: bash

            % curl -X POST -H "Content-Type: application/json" --data '[{"0":5,"1":4,"2":3,"3":2}]' http://0.0.0.0:3000/predict

            [{"0": 1}]%

    Args:
        orient (:code:`str`, `optional`, default to :code:`records`): Indication of expected JSON string format. Compatible JSON strings can be
            produced by :func:`pandas.io.json.to_json()` with a corresponding orient value.
            Possible orients are:
                - :obj:`split` - :code:`Dict[str, Any]`: {idx -> [idx], columns -> [columns], data -> [values]}
                - :obj:`records` - :code:`List[Any]`: [{column -> value}, ..., {column -> value}]
                - :obj:`index` - :code:`Dict[str, Any]`: {idx -> {column -> value}}
                - :obj:`columns` - :code:`Dict[str, Any]`: {column -> {index -> value}}
                - :obj:`values` - :code:`Dict[str, Any]`: Values arrays
        columns (:code:`List[str]`, `optional`, default to :code:`None`):
            List of columns name that users wish to update.
        apply_column_names (:code:`bool`, `optional`, default to :code:`False`):
            Whether to update incoming DataFrame columns. If :code:`apply_column_names`=True,
            then `columns` must be specified.
        dtype (:code:`Union[bool, Dict[str, Any]]`, `optional`, default to :code:`None`):
            Data Type users wish to convert their inputs/outputs to. If it is a boolean,
            then pandas will infer dtypes. Else if it is a dictionary of column to
            dtype, then applies those to incoming dataframes. If False, then don't
            infer dtypes at all (only applies to the data). This is not applicable when
            :code:`orient='table'`.
        enforce_dtype (:code:`bool`, `optional`, default to :code:`False`):
            Whether to enforce a certain data type. if :code:`enforce_dtype=True` then :code:`dtype`
            must be specified.
        shape (:code:`Tuple[int, ...]`, `optional`, default to :code:`None`): Optional shape check that users can specify for their incoming HTTP
            requests. We will only check the number of columns you specified for your
            given shape:

            .. code-block:: python

                import pandas as pd
                from bentoml.io import PandasDataFrame

                df = pd.DataFrame([[1,2,3]])  # shape (1,3)
                inp = PandasDataFrame.from_sample(arr)

                @svc.api(input=PandasDataFrame(
                  shape=(51,10), enforce_shape=True
                ), output=PandasDataFrame())
                def infer(input_df: pd.DataFrame) -> pd.DataFrame:...
                # if input_df have shape (40,9), it will throw out errors

        enforce_shape (`bool`, `optional`, default to :code:`False`):
            Whether to enforce a certain shape. If `enforce_shape=True` then `shape`
            must be specified

        default_format (:code:`str`, `optional`, default to :obj:`json`):
            The default serialization format to use if the request does not specify a :code:`headers['content-type']`.
            It is also the serialization format used for the response. Possible values are:
                - :obj:`json` - JSON text format (inferred from content-type "application/json")
                - :obj:`parquet` - Parquet binary format (inferred from content-type "application/octet-stream")
                - :obj:`csv` - CSV text format (inferred from content-type "text/csv")

    Returns:
        :obj:`~bentoml._internal.io_descriptors.IODescriptor`: IO Descriptor that `pd.DataFrame`.
    """

    def __init__(
        self,
        orient: "ext.DataFrameOrient" = "records",
        apply_column_names: bool = False,
        columns: t.Optional[t.List[str]] = None,
        dtype: t.Optional[t.Union[bool, t.Dict[str, t.Any]]] = None,
        enforce_dtype: bool = False,
        shape: t.Optional[t.Tuple[int, ...]] = None,
        enforce_shape: bool = False,
        default_format: "t.Literal['json', 'parquet', 'csv']" = "json",
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

    def openapi_schema_type(self) -> t.Dict[str, t.Any]:
        return _schema_type(self._dtype)

    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""
        return {self._default_format.mime_type: {"schema": self.openapi_schema_type()}}

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""
        return {self._default_format.mime_type: {"schema": self.openapi_schema_type()}}

    async def from_http_request(self, request: Request) -> "ext.PdDataFrame":
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
        if self._enforce_dtype:
            if self._dtype is None:
                logger.warning(
                    "`dtype` is None or undefined, while `enforce_dtype`=True"
                )
            # TODO(jiang): check dtype

        if serialization_format is SerializationFormat.JSON:
            res = pd.read_json(  # type: ignore[arg-type]
                io.BytesIO(obj),
                dtype=self._dtype,  # type: ignore[arg-type]
                orient=self._orient,
            )
        elif serialization_format is SerializationFormat.PARQUET:
            res = pd.read_parquet(  # type: ignore[arg-type]
                io.BytesIO(obj),
                engine=get_parquet_engine(),
            )
        elif serialization_format is SerializationFormat.CSV:
            res = pd.read_csv(  # type: ignore[arg-type]
                io.BytesIO(obj),
                dtype=self._dtype,  # type: ignore[arg-type]
            )
        else:
            raise InvalidArgument(
                f"Unknown serialization format ({serialization_format})."
            )

        assert isinstance(res, pd.DataFrame)

        if self._apply_column_names:
            if self._columns is None:
                logger.warning(
                    "`columns` is None or undefined, while `apply_column_names`=True"
                )
            elif len(self._columns) == res.shape[1]:
                raise BadInput(
                    "length of `columns` does not match the columns of incoming data"
                )
            else:
                res.columns = pd.Index(self._columns)
        if self._enforce_shape:
            if self._shape is None:
                logger.warning(
                    "`shape` is None or undefined, while `enforce_shape`=True"
                )
            else:
                assert all(
                    left == right
                    for left, right in zip(self._shape, res.shape)
                    if left != -1 and right != -1
                ), f"incoming has shape {res.shape} where enforced shape to be {self._shape}"
        return res

    async def to_http_response(self, obj: "pd.DataFrame") -> Response:
        """
        Process given objects and convert it to HTTP response.

        Args:
            obj (`pd.DataFrame`):
                `pd.DataFrame` that will be serialized to JSON or parquet
        Returns:
            HTTP Response of type `starlette.responses.Response`. This can
             be accessed via cURL or any external web traffic.
        """

        # For the response it doesn't make sense to enforce the same serialization format as specified
        # by the request's headers['content-type']. Instead we simply use the _default_format.
        serialization_format = self._default_format
        _validate_serialization_format(serialization_format)

        if not LazyType["ext.PdDataFrame"](pd.DataFrame).isinstance(obj):
            raise InvalidArgument(
                f"return object is not of type `pd.DataFrame`, got type {type(obj)} instead"
            )
        if serialization_format is SerializationFormat.JSON:
            resp = obj.to_json(orient=self._orient)  # type: ignore[arg-type]
        elif serialization_format is SerializationFormat.PARQUET:
            resp = obj.to_parquet(engine=get_parquet_engine())
        elif serialization_format is SerializationFormat.CSV:
            resp = obj.to_csv()
        else:
            raise InvalidArgument(
                f"Unknown serialization format ({serialization_format})."
            )

        return Response(resp, media_type=serialization_format.mime_type)

    @classmethod
    def from_sample(
        cls,
        sample_input: "pd.DataFrame",
        orient: "ext.DataFrameOrient" = "records",
        apply_column_names: bool = True,
        enforce_shape: bool = True,
        enforce_dtype: bool = False,
        default_format: "t.Literal['json', 'parquet', 'csv']" = "json",
    ) -> "PandasDataFrame":
        """
        Create a PandasDataFrame IO Descriptor from given inputs.

        Args:
            sample_input (`pd.DataFrame`): Given sample pd.DataFrame data
            orient (:code:`str`, `optional`, default to :code:`records`): Indication of expected JSON string format. Compatible JSON strings can be
                produced by :func:`pandas.io.json.to_json()` with a corresponding orient value.
                Possible orients are:
                    - :obj:`split` - :code:`Dict[str, Any]`: {idx -> [idx], columns -> [columns], data -> [values]}
                    - :obj:`records` - :code:`List[Any]`: [{column -> value}, ..., {column -> value}]
                    - :obj:`index` - :code:`Dict[str, Any]`: {idx -> {column -> value}}
                    - :obj:`columns` - :code:`Dict[str, Any]`: {column -> {index -> value}}
                    - :obj:`values` - :code:`Dict[str, Any]`: Values arrays
            apply_column_names (`bool`, `optional`, default to :code:`True`):
                Update incoming DataFrame columns. `columns` must be specified at
                function signature. If you don't want to enforce a specific columns
                name then change `apply_column_names=False`.
            enforce_dtype (`bool`, `optional`, default to :code:`True`):
                Enforce a certain data type. `dtype` must be specified at function
                signature. If you don't want to enforce a specific dtype then change
                `enforce_dtype=False`.
            enforce_shape (`bool`, `optional`, default to :code:`False`):
                Enforce a certain shape. `shape` must be specified at function
                signature. If you don't want to enforce a specific shape then change
                `enforce_shape=False`.
            default_format (:code:`str`, `optional`, default to :code:`json`):
                The default serialization format to use if the request does not specify a :code:`headers['content-type']`.
                It is also the serialization format used for the response. Possible values are:
                    - :obj:`json` - JSON text format (inferred from content-type "application/json")
                    - :obj:`parquet` - Parquet binary format (inferred from content-type "application/octet-stream")
                    - :obj:`csv` - CSV text format (inferred from content-type "text/csv")

        Returns:
            :obj:`bentoml._internal.io_descriptors.PandasDataFrame`: :code:`PandasDataFrame` IODescriptor from given users inputs.

        Example:

        .. code-block:: python

            import pandas as pd
            from bentoml.io import PandasDataFrame
            arr = [[1,2,3]]
            inp = PandasDataFrame.from_sample(pd.DataFrame(arr))

            ...
            @svc.api(input=inp, output=PandasDataFrame())
            def predict(inputs: pd.DataFrame) -> pd.DataFrame:...
        """
        columns = [str(x) for x in list(sample_input.columns)]  # type: ignore[reportUnknownVariableType]
        return cls(
            orient=orient,
            enforce_shape=enforce_shape,
            shape=sample_input.shape,
            apply_column_names=apply_column_names,
            columns=columns,
            enforce_dtype=enforce_dtype,
            dtype=None,  # TODO: not breaking atm
            default_format=default_format,
        )


class PandasSeries(IODescriptor["ext.PdSeries"]):
    """
    :code:`PandasSeries` defines API specification for the inputs/outputs of a Service, where
    either inputs will be converted to or outputs will be converted from type
    :code:`numpy.ndarray` as specified in your API function signature.

    Sample implementation of a sklearn service:

    .. code-block:: python

        # sklearn_svc.py
        import bentoml
        import pandas as pd
        import numpy as np
        from bentoml.io import PandasSeries
        import bentoml.sklearn

        input_spec = PandasSeries.from_sample(pd.Series(np.array([[5,4,3,2]])))

        runner = bentoml.sklearn.load_runner("sklearn_model_clf")

        svc = bentoml.Service("iris-classifier", runners=[runner])

        @svc.api(input=input_spec, output=PandasSeries())
        def predict(input_arr):
            res = runner.run_batch(input_arr)  # type: np.ndarray
            return pd.Series(res)

    Users then can then serve this service with :code:`bentoml serve`:

    .. code-block:: bash

        % bentoml serve ./sklearn_svc.py:svc --auto-reload

        (Press CTRL+C to quit)
        [INFO] Starting BentoML API server in development mode with auto-reload enabled
        [INFO] Serving BentoML Service "iris-classifier" defined in "sklearn_svc.py"
        [INFO] API Server running on http://0.0.0.0:3000

    Users can then send requests to the newly started services with any client:

    .. tabs::

        .. code-tab:: python

            import requests
            requests.post(
                "http://0.0.0.0:3000/predict",
                headers={"content-type": "application/json"},
                data='[{"0":5,"1":4,"2":3,"3":2}]'
            ).text

        .. code-tab:: bash

            % curl -X POST -H "Content-Type: application/json" --data '[{"0":5,"1":4,"2":3,"3":2}]' http://0.0.0.0:3000/predict

            [{"0": 1}]%

    Args:
        orient (:code:`str`, `optional`, default to :code:`records`): Indication of expected JSON string format. Compatible JSON strings can be
            produced by :func:`pandas.io.json.to_json()` with a corresponding orient value.
            Possible orients are:
                - :obj:`split` - :code:`Dict[str, Any]`: {idx -> [idx], columns -> [columns], data -> [values]}
                - :obj:`records` - :code:`List[Any]`: [{column -> value}, ..., {column -> value}]
                - :obj:`index` - :code:`Dict[str, Any]`: {idx -> {column -> value}}
                - :obj:`columns` - :code:`Dict[str, Any]`: {column -> {index -> value}}
                - :obj:`values` - :code:`Dict[str, Any]`: Values arrays
        columns (`List[str]`, `optional`, default to :code:`None`):
            List of columns name that users wish to update
        apply_column_names (`bool`, `optional`, default to :code:`False`):
            Whether to update incoming DataFrame columns. If
            :code:`apply_column_names=True`, then `columns` must be specified.
        dtype (:code:`Union[bool, Dict[str, Any]]`, `optional`, default to :code:`None`):
            Data Type users wish to convert their inputs/outputs to. If it is a boolean,
            then pandas will infer dtypes. Else if it is a dictionary of column to
            dtype, then applies those to incoming dataframes. If False, then don't
            infer dtypes at all (only applies to the data). This is not applicable when
            :code:`orient='table'`.
        enforce_dtype (`bool`, `optional`, default to :code:`False`):
            Whether to enforce a certain data type. if :code:`enforce_dtype=True` then
            :code:`dtype` must be specified.
        shape (`Tuple[int, ...]`, `optional`, default to :code:`None`):
            Optional shape check that users can specify for their incoming HTTP
            requests. We will only check the number of columns you specified for your
            given shape. Examples usage:

            .. code-block:: python

                import pandas as pd
                from bentoml.io import PandasSeries

                df = pd.DataFrame([[1,2,3]])  # shape (1,3)
                inp = PandasSeries.from_sample(arr)

                ...
                @svc.api(input=inp, output=PandasSeries())
                # the given shape above is valid
                def predict(input_df: pd.DataFrame) -> pd.DataFrame:
                    result = await runner.run(input_df)
                    return result

                @svc.api(input=PandasSeries(shape=(51,10), enforce_shape=True), output=PandasSeries())
                def infer(input_df: pd.DataFrame) -> pd.DataFrame: ...
                # if input_df have shape (40,9), it will throw out errors
        enforce_shape (`bool`, `optional`, default to :code:`False`):
            Whether to enforce a certain shape. If `enforce_shape=True` then `shape`
            must be specified

    Returns:
        :obj:`~bentoml._internal.io_descriptors.IODescriptor`: IO Descriptor that `pd.DataFrame`.
    """

    def __init__(
        self,
        orient: "ext.SeriesOrient" = "records",
        dtype: t.Optional[t.Union[bool, t.Dict[str, t.Any]]] = None,
        enforce_dtype: bool = False,
        shape: t.Optional[t.Tuple[int, ...]] = None,
        enforce_shape: bool = False,
    ):
        self._orient = orient
        self._dtype = dtype
        self._enforce_dtype = enforce_dtype
        self._shape = shape
        self._enforce_shape = enforce_shape
        self._mime_type = "application/json"

    def openapi_schema_type(self) -> t.Dict[str, t.Any]:
        return _schema_type(self._dtype)

    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""
        return {self._mime_type: {"schema": self.openapi_schema_type()}}

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for outgoing responses"""
        return {self._mime_type: {"schema": self.openapi_schema_type()}}

    async def from_http_request(self, request: Request) -> "ext.PdSeries":
        """
        Process incoming requests and convert incoming
         objects to `pd.Series`

        Args:
            request (`starlette.requests.Requests`):
                Incoming Requests
        Returns:
            a `pd.Series` object. This can then be used
             inside users defined logics.
        """
        obj = await request.body()
        if self._enforce_dtype:
            if self._dtype is None:
                logger.warning(
                    "`dtype` is None or undefined, while `enforce_dtype`=True"
                )

        # TODO(jiang): check dtypes when enforce_dtype is set
        res = pd.read_json(  # type: ignore[arg-type]
            obj,
            typ="series",
            orient=self._orient,
            dtype=self._dtype,  # type: ignore[arg-type]
        )

        assert isinstance(res, pd.Series)

        if self._enforce_shape:
            if self._shape is None:
                logger.warning(
                    "`shape` is None or undefined, while `enforce_shape`=True"
                )
            else:
                assert all(
                    left == right
                    for left, right in zip(self._shape, res.shape)
                    if left != -1 and right != -1
                ), f"incoming has shape {res.shape} where enforced shape to be {self._shape}"
        return res

    async def to_http_response(self, obj: t.Union[t.Any, "ext.PdSeries"]) -> Response:
        """
        Process given objects and convert it to HTTP response.

        Args:
            obj (`pd.Series`):
                `pd.Series` that will be serialized to JSON
        Returns:
            HTTP Response of type `starlette.responses.Response`. This can
             be accessed via cURL or any external web traffic.
        """
        if not LazyType["ext.PdSeries"](pd.Series).isinstance(obj):
            raise InvalidArgument(
                f"return object is not of type `pd.Series`, got type {type(obj)} instead"
            )
        resp = obj.to_json(orient=self._orient)  # type: ignore[arg-type]
        return Response(resp, media_type=MIME_TYPE_JSON)
