from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..types import HTTPRequest, HTTPResponse
from ..utils.lazy_loader import LazyLoader
from .base import IODescriptor

if TYPE_CHECKING:
    import pandas as pd
else:
    pd = LazyLoader("pd", globals(), "pandas")


class PandasDataFrame(IODescriptor):
    def __init__(
        self,
        orient: str = "records",
        columns: Optional[List[str]] = None,
        apply_column_names: bool = False,
        dtype: Optional[Dict[str, str]] = None,
        enforce_dtype: bool = False,
        shape: Optional[Tuple] = None,
        enforce_shape: bool = False,
    ):
        self._orient = orient
        self._columns = columns
        self._apply_column_names = apply_column_names
        self._dtype = dtype
        self._enforce_dtype = enforce_dtype
        self._shape = shape
        self._enforce_shape = enforce_shape

    def openapi_request_schema(self):
        pass

    def openapi_responses_schema(self):
        pass

    def from_http_request(self, request: HTTPRequest) -> "pd.DataFrame":
        pass

    def to_http_response(self, obj: "pd.DataFrame") -> HTTPResponse:
        pass

    @classmethod
    def from_sample(
        cls,
        sample_input: "pd.DataFrame",
        apply_column_names=True,
        enforce_dtype=True,
        enforce_shape=True,
    ):
        pass
