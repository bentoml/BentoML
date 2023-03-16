import typing as t
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

    F = t.Callable[..., t.Any]

    from pandas import Series as _PdSeries
    from pandas import DataFrame as PdDataFrame
    from pandas._typing import Dtype as PdDType
    from pandas._typing import DtypeArg as PdDTypeArg
    from pyarrow.plasma import ObjectID
    from pyarrow.plasma import PlasmaClient

    PdSeries = _PdSeries[t.Any]
    DataFrameOrient = Literal["split", "records", "index", "columns", "values", "table"]
    SeriesOrient = Literal["split", "records", "index", "table"]

    # numpy is always required by bentoml
    from numpy import generic as NpGeneric
    from numpy.typing import NDArray as _NDArray
    from numpy.typing import DTypeLike as NpDTypeLike

    NpNDArray = _NDArray[t.Any]

    from xgboost import DMatrix
    from catboost import Pool as CbPool

    from .starlette import ASGIApp
    from .starlette import ASGISend
    from .starlette import ASGIScope
    from .starlette import ASGIMessage
    from .starlette import ASGIReceive
    from .starlette import AsgiMiddleware

    WSGIApp = t.Callable[[F, t.Mapping[str, t.Any]], t.Iterable[bytes]]

    __all__ = [
        "PdSeries",
        "PdDataFrame",
        "PdDType",
        "PdDTypeArg",
        "DataFrameOrient",
        "SeriesOrient",
        "ObjectID",
        "PlasmaClient",
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
