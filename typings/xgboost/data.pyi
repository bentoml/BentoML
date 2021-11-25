import ctypes
from typing import Any, Callable, List, Optional

from .core import DataIter, DMatrix, _ProxyDMatrix

"""Data dispatching for DMatrix."""
c_bst_ulong = ctypes.c_uint64
CAT_T = ...
_pandas_dtype_mapper = ...
_dt_type_mapper = ...
_dt_type_mapper2 = ...

def dispatch_data_backend(
    data,
    missing,
    threads,
    feature_names: Optional[List[str]],
    feature_types: Optional[List[str]],
    enable_categorical: bool = ...,
):  # -> tuple[c_void_p, List[str] | None, List[str] | None] | Tuple[c_void_p, Any, Any]:
    """Dispatch data for DMatrix."""
    ...

def dispatch_meta_backend(matrix: DMatrix, data, name: str, dtype: str = ...):  # -> None:
    """Dispatch for meta info."""
    ...

class SingleBatchInternalIter(DataIter):
    """An iterator for single batch data to help creating device DMatrix.
    Transforming input directly to histogram with normal single batch data API
    can not access weight for sketching.  So this iterator acts as a staging
    area for meta info.

    """

    def __init__(self, **kwargs: Any) -> None: ...
    def next(self, input_data: Callable) -> int: ...
    def reset(self) -> None: ...

def dispatch_proxy_set_data(
    proxy: _ProxyDMatrix, data: Any, cat_codes: Optional[list], allow_host: bool
) -> None:
    """Dispatch for DeviceQuantileDMatrix."""
    ...
