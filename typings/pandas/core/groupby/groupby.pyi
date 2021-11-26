from contextlib import contextmanager
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Callable,
    Hashable,
    Iterator,
    List,
    Literal,
    Mapping,
    Sequence,
    Union,
)
import numpy as np
from pandas._libs import lib
from pandas._typing import FrameOrSeries, FrameOrSeriesUnion, IndexLabel, T, final
from pandas.core.base import PandasObject, SelectionMixin
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby import ops
from pandas.core.series import Series
from pandas.util._decorators import Appender, Substitution, doc

if TYPE_CHECKING: ...
_common_see_also = ...
_apply_docs = ...
_groupby_agg_method_template = ...
_pipe_template = ...
_transform_template = ...
_agg_template = ...

@final
class GroupByPlot(PandasObject):
    def __init__(self, groupby: GroupBy) -> None: ...
    def __call__(self, *args, **kwargs): ...
    def __getattr__(self, name: str): ...

@contextmanager
def group_selection_context(groupby: GroupBy) -> Iterator[GroupBy]: ...

_KeysArgType = Union[
    Hashable,
    List[Hashable],
    Callable[[Hashable], Hashable],
    List[Callable[[Hashable], Hashable]],
    Mapping[Hashable, Hashable],
]

class BaseGroupBy(PandasObject, SelectionMixin[FrameOrSeries]):
    _group_selection: IndexLabel | None = ...
    _apply_allowlist: frozenset[str] = ...
    _hidden_attrs = ...
    axis: int
    grouper: ops.BaseGrouper
    group_keys: bool
    @final
    def __len__(self) -> int: ...
    @final
    def __repr__(self) -> str: ...
    @final
    @property
    def groups(self) -> dict[Hashable, np.ndarray]: ...
    @final
    @property
    def ngroups(self) -> int: ...
    @final
    @property
    def indices(self): ...
    @Substitution(
        klass="GroupBy",
        examples=dedent(
            """        >>> df = pd.DataFrame({'A': 'a b a b'.split(), 'B': [1, 2, 3, 4]})
        >>> df
           A  B
        0  a  1
        1  b  2
        2  a  3
        3  b  4
        To get the difference between each groups maximum and minimum value in one
        , you can do
        >>> df.groupby('A').pipe(lambda x: x.max() - x.min())
           B
        A
        a  2
        b  2"""
        ),
    )
    @Appender(_pipe_template)
    def pipe(
        self, func: Callable[..., T] | tuple[Callable[..., T], str], *args, **kwargs
    ) -> T: ...
    plot = ...
    @final
    def get_group(self, name, obj=...) -> FrameOrSeriesUnion: ...
    @final
    def __iter__(self) -> Iterator[tuple[Hashable, FrameOrSeries]]: ...

OutputFrameOrSeries = ...

class GroupBy(BaseGroupBy[FrameOrSeries]):
    grouper: ops.BaseGrouper
    as_index: bool
    @final
    def __init__(
        self,
        obj: FrameOrSeries,
        keys: _KeysArgType | None = ...,
        axis: int = ...,
        level: IndexLabel | None = ...,
        grouper: ops.BaseGrouper | None = ...,
        exclusions: frozenset[Hashable] | None = ...,
        selection: IndexLabel | None = ...,
        as_index: bool = ...,
        sort: bool = ...,
        group_keys: bool = ...,
        squeeze: bool = ...,
        observed: bool = ...,
        mutated: bool = ...,
        dropna: bool = ...,
    ) -> None: ...
    def __getattr__(self, attr: str): ...
    @Appender(
        _apply_docs["template"].format(
            input="dataframe", examples=_apply_docs["dataframe_examples"]
        )
    )
    def apply(self, func, *args, **kwargs): ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def any(self, skipna: bool = ...): ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def all(self, skipna: bool = ...): ...
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def count(self): ...
    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def mean(self, numeric_only: bool | lib.NoDefault = ...): ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def median(self, numeric_only: bool | lib.NoDefault = ...): ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def std(self, ddof: int = ...): ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def var(self, ddof: int = ...): ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def sem(self, ddof: int = ...): ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def size(self) -> FrameOrSeriesUnion: ...
    @final
    @doc(_groupby_agg_method_template, fname="sum", no=True, mc=0)
    def sum(self, numeric_only: bool | lib.NoDefault = ..., min_count: int = ...): ...
    @final
    @doc(_groupby_agg_method_template, fname="prod", no=True, mc=0)
    def prod(self, numeric_only: bool | lib.NoDefault = ..., min_count: int = ...): ...
    @final
    @doc(_groupby_agg_method_template, fname="min", no=False, mc=-1)
    def min(self, numeric_only: bool = ..., min_count: int = ...): ...
    @final
    @doc(_groupby_agg_method_template, fname="max", no=False, mc=-1)
    def max(self, numeric_only: bool = ..., min_count: int = ...): ...
    @final
    @doc(_groupby_agg_method_template, fname="first", no=False, mc=-1)
    def first(self, numeric_only: bool = ..., min_count: int = ...): ...
    @final
    @doc(_groupby_agg_method_template, fname="last", no=False, mc=-1)
    def last(self, numeric_only: bool = ..., min_count: int = ...): ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def ohlc(self) -> DataFrame: ...
    @final
    @doc(DataFrame.describe)
    def describe(self, **kwargs): ...
    @final
    def resample(self, rule, *args, **kwargs): ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def rolling(self, *args, **kwargs): ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def expanding(self, *args, **kwargs): ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def ewm(self, *args, **kwargs): ...
    @final
    @Substitution(name="groupby")
    def pad(self, limit=...): ...
    ffill = ...
    @final
    @Substitution(name="groupby")
    def backfill(self, limit=...): ...
    bfill = ...
    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def nth(
        self, n: int | list[int], dropna: Literal["any", "all", None] = ...
    ) -> DataFrame: ...
    @final
    def quantile(self, q=..., interpolation: str = ...): ...
    @final
    @Substitution(name="groupby")
    def ngroup(self, ascending: bool = ...): ...
    @final
    @Substitution(name="groupby")
    def cumcount(self, ascending: bool = ...): ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def rank(
        self,
        method: str = ...,
        ascending: bool = ...,
        na_option: str = ...,
        pct: bool = ...,
        axis: int = ...,
    ): ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def cumprod(self, axis=..., *args, **kwargs): ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def cumsum(self, axis=..., *args, **kwargs): ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def cummin(self, axis=..., **kwargs): ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def cummax(self, axis=..., **kwargs): ...
    @final
    @Substitution(name="groupby")
    def shift(self, periods=..., freq=..., axis=..., fill_value=...): ...
    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def pct_change(
        self, periods=..., fill_method=..., limit=..., freq=..., axis=...
    ): ...
    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def head(self, n=...): ...
    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def tail(self, n=...): ...
    @final
    def sample(
        self,
        n: int | None = ...,
        frac: float | None = ...,
        replace: bool = ...,
        weights: Sequence | Series | None = ...,
        random_state=...,
    ): ...

@doc(GroupBy)
def get_groupby(
    obj: NDFrame,
    by: _KeysArgType | None = ...,
    axis: int = ...,
    level=...,
    grouper: ops.BaseGrouper | None = ...,
    exclusions=...,
    selection=...,
    as_index: bool = ...,
    sort: bool = ...,
    group_keys: bool = ...,
    squeeze: bool = ...,
    observed: bool = ...,
    mutated: bool = ...,
    dropna: bool = ...,
) -> GroupBy: ...
