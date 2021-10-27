import logging
import typing as t

from starlette.requests import Request
from starlette.responses import Response

from ..utils.dataframe import from_json_or_csv
from ..utils.lazy_loader import LazyLoader
from .base import IODescriptor
from .json import MIME_TYPE_JSON

if t.TYPE_CHECKING:
    import pandas as pd
else:
    pd = LazyLoader("pd", globals(), "pandas")

logger = logging.getLogger(__name__)


class PandasDataFrame(IODescriptor):
    """
    `PandasDataFrame` defines API specification for the inputs/outputs of a Service, where either inputs will be
    converted to or outputs will be converted from type `numpy.ndarray` as specified in your API function signature.

    .. Toy implementation of a sklearn service::
        # sklearn_svc.py
        import pandas as pd
        import bentoml.sklearn

        from bentoml import Service
        from bentoml.io import PandasDataFrame

        my_runner = bentoml.sklearn.load_runner("my_sklearn_model:latest")

        svc = Service("IrisClassifierService", runner=[my_runner])

        # Create API function with pre- and post- processing logic
        @svc.api(input=PandasDataFrame(), output=PandasDataFrame())
        def predict(input_df: pd.DataFrame) -> pd.DataFrame:
            result = await runner.run(input_array)
            # Define post-processing logic
            return pd.DataFrame(result)

    Users then can then serve this service with `bentoml serve`::
        % bentoml serve ./sklearn_svc.py:svc --auto-reload

        (Press CTRL+C to quit)
        [INFO] Starting BentoML API server in development mode with auto-reload enabled
        [INFO] Serving BentoML Service "IrisClassifierService" defined in "sklearn_svc.py"
        [INFO] API Server running on http://0.0.0.0:5000

    Users can then send a cURL requests like shown in different terminal session::
        % curl -X POST -H "application/json" --data "[{'0':5,'1':4,'2':3,'3':2}]" http://0.0.0.0:5000/predict

        {res: [{"0":1}]}%

    Args:
        orient (`str`, `optional`, default to `records`):
            Indication of expected JSON string format. Compatible JSON strings can be produced
             by `pandas.io.json.to_json()` with a corresponding orient value. Possible orients are:
                - `split` - `Dict[str, Any]`: {idx -> [idx], columns -> [columns], data -> [values]}
                - `records` - `List[Any]`: [{column -> value}, ..., {column -> value}]
                - `index` - `Dict[str, Any]`: {idx -> {column -> value}}
                - `columns` - `Dict[str, Any]`: {column -> {index -> value}}
                - `values` - `Dict[str, Any]`: Values arrays
        columns (`List[str]`, `optional`, default to `None`):
            List of columns name that users wish to update
        apply_column_names (`bool`, `optional`, default to `False`):
            Whether to update incoming DataFrame columns. If `apply_column_names`=True, then
             `columns` must be specified.
        dtype (`Union[bool, Dict[str, Any]]`, `optional`, default to `None`):
            Data Type users wish to convert their inputs/outputs to. If it is a boolean,
             then pandas will infer dtypes. Else if it is a dictionary of column to dtype, then
             applies those to incoming dataframes. If False, then don't infer dtypes at all (only
             applies to the data). This is not applicable when `orient='table'`.
        enforce_dtype (`bool`, `optional`, default to `False`):
            Whether to enforce a certain data type. if `enforce_dtype=True` then `dtype` must
             be specified.
        shape (`Tuple[int, ...]`, `optional`, default to `None`):
            Optional shape check that users can specify for their incoming HTTP requests. We will
             only check the number of columns you specified for your given shape.
            Examples::
                import pandas as pd
                from bentoml.io import PandasDataFrame

                df = pd.DataFrame([[1,2,3]])  # shape (1,3)
                inp = PandasDataFrame.from_sample(arr)

                ...
                @svc.api(input=inp(shape=(51,3), enforce_shape=True), output=PandasDataFrame())
                # the given shape above is valid
                def predict(input_array: np.ndarray) -> np.ndarray:
                    result = await runner.run(input_array)
                    return result

                @svc.api(input=inp(shape=(51,10), enforce_shape=True), output=PandasDataFrame())
                # the given shape above will throw errors
                def infer(input_array: np.ndarray) -> np.ndarray:...
        enforce_shape (`bool`, `optional`, default to `False`):
            Whether to enforce a certain shape. If `enforce_shape=True` then `shape` must
             be specified
    Returns:
        IO Descriptor that represents `pd.DataFrame`.
    """  # noqa

    def __init__(
        self,
        orient: str = "records",
        columns: t.Optional[t.List[str]] = None,
        apply_column_names: bool = False,
        dtype: t.Optional[t.Union[bool, t.Dict[str, t.Any]]] = None,
        enforce_dtype: bool = False,
        shape: t.Optional[t.Tuple[int, ...]] = None,
        enforce_shape: bool = False,
    ):
        self._orient = orient
        self._columns = columns
        self._apply_column_names = apply_column_names
        self._dtype = dtype
        self._enforce_dtype = enforce_dtype
        self._shape = shape
        self._enforce_shape = enforce_shape

    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""

    async def from_http_request(self, request: Request) -> "pd.DataFrame":
        """
        Process incoming requests and convert incoming
         objects to `pd.DataFrame`

        Args:
            request (`starlette.requests.Requests`):
                Incoming Requests
        Returns:
            a `pd.DataFrame` object. This can then be used
             inside users defined logics.
        """
        obj = await request.json()
        if self._enforce_dtype:
            if self._dtype is None:
                logger.warning(
                    "`dtype` is None or undefined, while `enforce_dtype`=True"
                )
        # TODO: check when inputs type is CSV or AWS Lambda to formats
        res, _ = from_json_or_csv(
            obj,
            formats="json",
            orient=self._orient,
            columns=self._columns,
            dtype=self._dtype,
        )  # type: pd.DataFrame, t.Tuple[int, ...]
        if self._apply_column_names:
            if self._columns is None:
                logger.warning(
                    "`columns` is None or undefined, while `apply_column_names`=True"
                )
            else:
                res.column = self._columns
        if self._enforce_shape:
            if self._shape is None:
                logger.warning(
                    "`shape` is None or undefined, while `enforce_shape`=True"
                )
            else:
                assert (
                    self._shape[1] == res.shape[1]
                ), f"incoming has shape {res.shape} where enforced shape to be {self._shape}"  # noqa
        return res

    async def to_http_response(self, obj: "pd.DataFrame") -> Response:
        """
        Process given objects and convert it to HTTP response.

        Args:
            obj (`pd.DataFrame`):
                `pd.DataFrame` that will be serialized to JSON
        Returns:
            HTTP Response of type `starlette.responses.Response`. This can
             be accessed via cURL or any external web traffic.
        """
        resp = obj.to_json(orient=self._orient)
        return Response(resp, media_type=MIME_TYPE_JSON)

    @classmethod
    def from_sample(
        cls,
        sample_input: "pd.DataFrame",
        apply_column_names: bool = True,
        enforce_dtype: bool = True,
        enforce_shape: bool = True,
    ) -> "PandasDataFrame":
        """Create an IO Descriptor from given inputs."""
