import typing as t

from ..types import HTTPRequest, HTTPResponse
from ..utils.lazy_loader import LazyLoader
from .base import IODescriptor

if t.TYPE_CHECKING:
    import pandas as pd
else:
    pd = LazyLoader("pd", globals(), "pandas")


class PandasDataFrame(IODescriptor):
    def openapi_schema(self):
        pass

    def from_http_request(self, request: HTTPRequest) -> pd.DataFrame:
        pass

    def to_http_response(self, obj: pd.DataFrame) -> HTTPResponse:
        pass
