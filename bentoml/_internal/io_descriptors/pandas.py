import logging
import typing as t

from starlette.requests import Request
from starlette.responses import Response

from ..utils.lazy_loader import LazyLoader
from .base import IODescriptor
from .json import MIME_TYPE_JSON

if t.TYPE_CHECKING:
    import pandas as pd
else:
    pd = LazyLoader("pd", globals(), "pandas")

logger = logging.getLogger(__name__)


class PandasDataFrame(IODescriptor):
    def __init__(
        self,
        orient: str = "records",
        columns: t.Optional[t.List[str]] = None,
        apply_column_names: bool = False,
        dtype: t.Optional[t.Dict[str, str]] = None,
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
        pass

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        pass

    async def from_http_request(self, request: Request) -> "pd.DataFrame":
        obj = await request.json()
        if self._enforce_dtype:
            if self._dtype is None:
                logger.warning(
                    "`dtype` is None or undefined, while `enforce_dtype`=True"
                )
        res = pd.read_json(
            obj, orient=self._orient, dtype=self._dtype
        )  # type: pd.DataFrame
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
        pass
