from typing import Any

from pandas.core.computation import expr, ops
from pandas.core.computation import scope as _scope
from pandas.core.computation.expr import BaseExprVisitor
from pandas.core.indexes.base import Index

""" manage PyTables query interface via Expressions """

class PyTablesScope(_scope.Scope):
    __slots__ = ...
    queryables: dict[str, Any]
    def __init__(
        self,
        level: int,
        global_dict=...,
        local_dict=...,
        queryables: dict[str, Any] | None = ...,
    ) -> None: ...

class Term(ops.Term):
    env: PyTablesScope
    def __new__(cls, name, env, side=..., encoding=...): ...
    def __init__(self, name, env: PyTablesScope, side=..., encoding=...) -> None: ...
    @property
    def value(self): ...

class Constant(Term):
    def __init__(self, value, env: PyTablesScope, side=..., encoding=...) -> None: ...

class BinOp(ops.BinOp):
    _max_selectors = ...
    op: str
    queryables: dict[str, Any]
    condition: str | None
    def __init__(
        self, op: str, lhs, rhs, queryables: dict[str, Any], encoding
    ) -> None: ...
    def prune(self, klass): ...
    def conform(self, rhs):  # -> ndarray | list[Unknown]:
        """inplace conform rhs"""
        ...
    @property
    def is_valid(self) -> bool:
        """return True if this is a valid field"""
        ...
    @property
    def is_in_table(self) -> bool:
        """
        return True if this is a valid column name for generation (e.g. an
        actual column in the table)
        """
        ...
    @property
    def kind(self):  # -> Any | None:
        """the kind of my field"""
        ...
    @property
    def meta(self):  # -> Any | None:
        """the meta of my field"""
        ...
    @property
    def metadata(self):  # -> Any | None:
        """the metadata of my field"""
        ...
    def generate(self, v) -> str:
        """create and return the op string for this TermValue"""
        ...
    def convert_value(self, v) -> TermValue:
        """
        convert the expression that is in the term to something that is
        accepted by pytables
        """
        ...
    def convert_values(self): ...

class FilterBinOp(BinOp):
    filter: tuple[Any, Any, Index] | None = ...
    def __repr__(self) -> str: ...
    def invert(self):  # -> Self@FilterBinOp:
        """invert the filter"""
        ...
    def format(self):  # -> list[tuple[Any, Any, Index] | None]:
        """return the actual filter format"""
        ...
    def evaluate(self): ...
    def generate_filter_op(self, invert: bool = ...): ...

class JointFilterBinOp(FilterBinOp):
    def format(self): ...
    def evaluate(self): ...

class ConditionBinOp(BinOp):
    def __repr__(self) -> str: ...
    def invert(self):
        """invert the condition"""
        ...
    def format(self):  # -> str | None:
        """return the actual ne format"""
        ...
    def evaluate(self): ...

class JointConditionBinOp(ConditionBinOp):
    def evaluate(self): ...

class UnaryOp(ops.UnaryOp):
    def prune(self, klass): ...

class PyTablesExprVisitor(BaseExprVisitor):
    const_type = Constant
    term_type = Term
    def __init__(self, env, engine, parser, **kwargs) -> None: ...
    def visit_UnaryOp(self, node, **kwargs): ...
    def visit_Index(self, node, **kwargs): ...
    def visit_Assign(self, node, **kwargs): ...
    def visit_Subscript(self, node, **kwargs): ...
    def visit_Attribute(self, node, **kwargs): ...
    def translate_In(self, op): ...

class PyTablesExpr(expr.Expr):
    """
    Hold a pytables-like expression, comprised of possibly multiple 'terms'.

    Parameters
    ----------
    where : string term expression, PyTablesExpr, or list-like of PyTablesExprs
    queryables : a "kinds" map (dict of column name -> kind), or None if column
        is non-indexable
    encoding : an encoding that will encode the query terms

    Returns
    -------
    a PyTablesExpr object

    Examples
    --------
    'index>=date'
    "columns=['A', 'D']"
    'columns=A'
    'columns==A'
    "~(columns=['A','B'])"
    'index>df.index[3] & string="bar"'
    '(index>df.index[3] & index<=df.index[6]) | string="bar"'
    "ts>=Timestamp('2012-02-01')"
    "major_axis>=20130101"
    """

    _visitor: PyTablesExprVisitor | None
    env: PyTablesScope
    expr: str
    def __init__(
        self,
        where,
        queryables: dict[str, Any] | None = ...,
        encoding=...,
        scope_level: int = ...,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def evaluate(self):  # -> tuple[Any, Any]:
        """create and return the numexpr condition and filter"""
        ...

class TermValue:
    """hold a term value the we use to construct a condition/filter"""

    def __init__(self, value, converted, kind: str) -> None: ...
    def tostring(self, encoding) -> str:
        """quote the string if not encoded else encode and return"""
        ...

def maybe_expression(s) -> bool:
    """loose checking if s is a pytables-acceptable expression"""
    ...
