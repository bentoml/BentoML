from typing import Any, Callable, Literal, overload

import numpy as np

_BinOp = Callable[[Any, Any], Any]
_BoolOp = Callable[[Any, Any], bool]

def scalar_compare(values: np.ndarray, val: object, op: _BoolOp) -> np.ndarray: ...
def vec_compare(left: np.ndarray, right: np.ndarray, op: _BoolOp) -> np.ndarray: ...
def scalar_binop(values: np.ndarray, val: object, op: _BinOp) -> np.ndarray: ...
def vec_binop(left: np.ndarray, right: np.ndarray, op: _BinOp) -> np.ndarray: ...
@overload
def maybe_convert_bool(
    arr: np.ndarray,
    true_values=...,
    false_values=...,
    convert_to_masked_nullable: Literal[False] = ...,
) -> tuple[np.ndarray, None]: ...
@overload
def maybe_convert_bool(
    arr: np.ndarray,
    true_values=...,
    false_values=...,
    *,
    convert_to_masked_nullable: Literal[True]
) -> tuple[np.ndarray, np.ndarray]: ...
