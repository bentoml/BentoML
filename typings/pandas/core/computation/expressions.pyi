from pandas._typing import FuncType
from pandas.core.computation.check import NUMEXPR_INSTALLED

"""
Expressions
-----------

Offer fast expression evaluation through numexpr

"""
if NUMEXPR_INSTALLED: ...
_TEST_MODE: bool | None = ...
_TEST_RESULT: list[bool] = ...
USE_NUMEXPR = ...
_evaluate: FuncType | None = ...
_where: FuncType | None = ...
_ALLOWED_DTYPES = ...
_MIN_ELEMENTS = ...

def set_use_numexpr(v=...): ...
def set_numexpr_threads(n=...): ...

_op_str_mapping = ...
_BOOL_OP_UNSUPPORTED = ...

def evaluate(op, a, b, use_numexpr: bool = ...):
    """
    Evaluate and return the expression of the op on a and b.

    Parameters
    ----------
    op : the actual operand
    a : left operand
    b : right operand
    use_numexpr : bool, default True
        Whether to try to use numexpr.
    """
    ...

def where(cond, a, b, use_numexpr=...):  # -> Any:
    """
    Evaluate the where condition cond on a and b.

    Parameters
    ----------
    cond : np.ndarray[bool]
    a : return if cond is True
    b : return if cond is False
    use_numexpr : bool, default True
        Whether to try to use numexpr.
    """
    ...

def set_test_mode(v: bool = ...) -> None:
    """
    Keeps track of whether numexpr was used.

    Stores an additional ``True`` for every successful use of evaluate with
    numexpr since the last ``get_test_result``.
    """
    ...

def get_test_result() -> list[bool]:
    """
    Get test result and reset test_results.
    """
    ...
