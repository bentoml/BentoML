from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.ma.mrecords import MaskedRecords
from pandas._typing import ArrayLike, DtypeObj, Manager
from pandas.core.indexes.api import Index

"""
Functions for preparing various inputs passed to the DataFrame or Series
constructors before passing them to a BlockManager.
"""
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
) -> Manager:
    """
    Segregate Series based on type and coerce into matrices.

    Needs to handle a lot of exceptional cases.
    """
    ...

def rec_array_to_mgr(
    data: MaskedRecords | np.recarray | np.ndarray,
    index,
    columns,
    dtype: DtypeObj | None,
    copy: bool,
    typ: str,
):  # -> Manager:
    """
    Extract from a masked rec array and create the manager.
    """
    ...

def fill_masked_arrays(data: MaskedRecords, arr_columns: Index) -> list[np.ndarray]:
    """
    Convert numpy MaskedRecords to ensure mask is softened.
    """
    ...

def mgr_to_mgr(mgr, typ: str, copy: bool = ...):  # -> Manager:
    """
    Convert to specific type of Manager. Does not copy if the type is already
    correct. Does not guarantee a copy otherwise. `copy` keyword only controls
    whether conversion from Block->ArrayManager copies the 1D arrays.
    """
    ...

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
) -> Manager:
    """
    Segregate Series based on type and coerce into matrices.
    Needs to handle a lot of exceptional cases.

    Used in DataFrame.__init__
    """
    ...

def nested_data_to_arrays(
    data: Sequence, columns: Index | None, index: Index | None, dtype: DtypeObj | None
) -> tuple[list[ArrayLike], Index, Index]:
    """
    Convert a single sequence of arrays to multiple arrays.
    """
    ...

def treat_as_nested(data) -> bool:
    """
    Check if we should use nested_data_to_arrays.
    """
    ...

def reorder_arrays(
    arrays: list[ArrayLike], arr_columns: Index, columns: Index | None
) -> tuple[list[ArrayLike], Index]: ...
def dataclasses_to_dicts(data):  # -> list[dict[str, Any]]:
    """
    Converts a list of dataclass instances to a list of dictionaries.

    Parameters
    ----------
    data : List[Type[dataclass]]

    Returns
    --------
    list_dict : List[dict]

    Examples
    --------
    >>> @dataclass
    >>> class Point:
    ...     x: int
    ...     y: int

    >>> dataclasses_to_dicts([Point(1,2), Point(2,3)])
    [{"x":1,"y":2},{"x":2,"y":3}]

    """
    ...

def to_arrays(
    data, columns: Index | None, dtype: DtypeObj | None = ...
) -> tuple[list[ArrayLike], Index]:
    """
    Return list of arrays, columns.
    """
    ...
