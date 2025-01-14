"""Type definitions for BentoML external dependencies."""

import typing as t
from typing import TYPE_CHECKING
from typing import Literal

# Import starlette types
from .starlette import ASGIApp  # type: ignore
from .starlette import ASGIMessage  # type: ignore
from .starlette import AsgiMiddleware  # type: ignore
from .starlette import ASGIReceive  # type: ignore
from .starlette import ASGIScope  # type: ignore
from .starlette import ASGISend  # type: ignore

# Type aliases
F = t.Callable[..., t.Any]
WSGIApp = t.Callable[[F, t.Mapping[str, t.Any]], t.Iterable[bytes]]

if TYPE_CHECKING:
    from catboost.core import Pool as CbPool  # type: ignore
    from numpy import generic as NpGeneric
    from numpy.typing import DTypeLike as NpDTypeLike
    from numpy.typing import NDArray as _NDArray
    from pandas import DataFrame as PdDataFrame
    from pandas import Series as _PdSeries
    from pandas._typing import Dtype as PdDType
    from pandas._typing import DtypeArg as PdDTypeArg
    from PIL.Image import Image as PILImage
    from xgboost.core import DMatrix  # type: ignore

    # Type aliases that require type checking
    PdSeries = _PdSeries[t.Any]
    NpNDArray = _NDArray[t.Any]
    DataFrameOrient = Literal["split", "records", "index", "columns", "values", "table"]
    SeriesOrient = Literal["split", "records", "index", "table"]

__all__ = [
    "PdSeries",
    "PdDataFrame",
    "PdDType",
    "PdDTypeArg",
    "PILImage",
    "DataFrameOrient",
    "SeriesOrient",
    # xgboost
    "DMatrix",
    "CbPool",
    # numpy
    "NpNDArray",
    "NpGeneric",
    "NpDTypeLike",
    # starlette
    "AsgiMiddleware",
    "ASGIApp",
    "ASGIScope",
    "ASGISend",
    "ASGIReceive",
    "ASGIMessage",
    # misc
    "WSGIApp",
]
