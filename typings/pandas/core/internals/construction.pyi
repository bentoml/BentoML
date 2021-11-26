from typing import TYPE_CHECKING, Sequence
import numpy as np
from numpy.ma.mrecords import MaskedRecords
from pandas._typing import ArrayLike, DtypeObj, Manager
from pandas.core.indexes.api import Index

if TYPE_CHECKING: ...

def arrays_to_mgr(
    arrays,
    arr_names: Index,
    index,
    columns,
    *,
    dtype: DtypeObj | None = ...,
    verify_integrity: bool = ...,
    typ: str | None = ...,
    consolidate: bool = ...
) -> Manager: ...
def rec_array_to_mgr(
    data: MaskedRecords | np.recarray | np.ndarray,
    index,
    columns,
    dtype: DtypeObj | None,
    copy: bool,
    typ: str,
): ...
def fill_masked_arrays(data: MaskedRecords, arr_columns: Index) -> list[np.ndarray]: ...
def mgr_to_mgr(mgr, typ: str, copy: bool = ...): ...
def ndarray_to_mgr(
    values, index, columns, dtype: DtypeObj | None, copy: bool, typ: str
) -> Manager: ...
def dict_to_mgr(
    data: dict,
    index,
    columns,
    *,
    dtype: DtypeObj | None = ...,
    typ: str = ...,
    copy: bool = ...
) -> Manager: ...
def nested_data_to_arrays(
    data: Sequence, columns: Index | None, index: Index | None, dtype: DtypeObj | None
) -> tuple[list[ArrayLike], Index, Index]: ...
def treat_as_nested(data) -> bool: ...
def reorder_arrays(
    arrays: list[ArrayLike], arr_columns: Index, columns: Index | None
) -> tuple[list[ArrayLike], Index]: ...
def dataclasses_to_dicts(data): ...
def to_arrays(
    data, columns: Index | None, dtype: DtypeObj | None = ...
) -> tuple[list[ArrayLike], Index]: ...
