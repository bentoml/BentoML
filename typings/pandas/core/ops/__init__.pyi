from __future__ import annotations

import operator
import warnings
from typing import TYPE_CHECKING

import numpy as np
from pandas import DataFrame, Series
from pandas._libs.ops_dispatch import maybe_dispatch_ufunc_to_dunder_op
from pandas._typing import Level
from pandas.core import algorithms, roperator
from pandas.core.dtypes.common import is_array_like, is_list_like
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.missing import isna
from pandas.core.ops.array_ops import (
    arithmetic_op,
    comp_method_OBJECT_ARRAY,
    comparison_op,
    get_array_op,
    logical_op,
    maybe_prepare_scalar_for_op,
)
from pandas.core.ops.common import get_op_result_name, unpack_zerodim_and_defer
from pandas.core.ops.docstrings import (
    _flex_comp_doc_FRAME,
    _op_descriptions,
    make_flex_doc,
)
from pandas.core.ops.invalid import invalid_comparison
from pandas.core.ops.mask_ops import kleene_and, kleene_or, kleene_xor
from pandas.core.ops.methods import add_flex_arithmetic_methods
from pandas.core.roperator import (
    radd,
    rand_,
    rdiv,
    rdivmod,
    rfloordiv,
    rmod,
    rmul,
    ror_,
    rpow,
    rsub,
    rtruediv,
    rxor,
)
from pandas.util._decorators import Appender

"""
Arithmetic operations for PandasObjects

This is not a public API.
"""
if TYPE_CHECKING: ...
ARITHMETIC_BINOPS: set[str] = ...
COMPARISON_BINOPS: set[str] = ...

def fill_binop(left, right, fill_value):  # -> tuple[Unknown, Unknown]:
    """
    If a non-None fill_value is given, replace null entries in left and right
    with this value, but only in positions where _one_ of left/right is null,
    not both.

    Parameters
    ----------
    left : array-like
    right : array-like
    fill_value : object

    Returns
    -------
    left : array-like
    right : array-like

    Notes
    -----
    Makes copies if fill_value is not None and NAs are present.
    """
    ...

def align_method_SERIES(
    left: Series, right, align_asobject: bool = ...
):  # -> tuple[Series, DataFrame | Series | Unknown]:
    """align lhs and rhs Series"""
    ...

def flex_method_SERIES(op): ...
def align_method_FRAME(
    left, right, axis, flex: bool | None = ..., level: Level = ...
):  # -> tuple[Unknown, Unknown | ABCDataFrame | Series | ndarray]:
    """
    Convert rhs to meet lhs dims if input is list, tuple or np.ndarray.

    Parameters
    ----------
    left : DataFrame
    right : Any
    axis : int, str, or None
    flex : bool or None, default False
        Whether this is a flex op, in which case we reindex.
        None indicates not to check for alignment.
    level : int or level name, default None

    Returns
    -------
    left : DataFrame
    right : Any
    """
    ...

def should_reindex_frame_op(
    left: DataFrame, right, op, axis, default_axis, fill_value, level
) -> bool:
    """
    Check if this is an operation between DataFrames that will need to reindex.
    """
    ...

def frame_arith_method_with_reindex(left: DataFrame, right: DataFrame, op) -> DataFrame:
    """
    For DataFrame-with-DataFrame operations that require reindexing,
    operate only on shared columns, then reindex.

    Parameters
    ----------
    left : DataFrame
    right : DataFrame
    op : binary operator

    Returns
    -------
    DataFrame
    """
    ...

def flex_arith_method_FRAME(op): ...
def flex_comp_method_FRAME(op): ...
