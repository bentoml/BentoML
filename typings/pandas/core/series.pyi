from textwrap import dedent
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Iterable,
    Literal,
    Sequence,
    overload,
)
import numpy as np
from pandas._libs import lib
from pandas._typing import (
    AggFuncType,
    ArrayLike,
    Axis,
    Dtype,
    DtypeObj,
    FillnaOptions,
    FrameOrSeriesUnion,
    IndexKeyFunc,
    NpDtype,
    SingleManager,
    StorageOptions,
    TimedeltaConvertibleTypes,
    TimestampConvertibleTypes,
    ValueKeyFunc,
)
from pandas.core import base, generic
from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.missing import isna, notna
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.indexes.api import Index
from pandas.core.resample import Resampler
from pandas.core.shared_docs import _shared_docs
from pandas.util._decorators import (
    Appender,
    Substitution,
    deprecate_nonkeyword_arguments,
    doc,
)

if TYPE_CHECKING: ...
__all__ = ["Series"]
_shared_doc_kwargs = ...

class Series(base.IndexOpsMixin, generic.NDFrame):
    _typ = ...
    _HANDLED_TYPES = ...
    _name: Hashable
    _metadata: list[str] = ...
    _internal_names_set = ...
    _accessors = ...
    _hidden_attrs = ...
    hasnans = ...
    _mgr: SingleManager
    div: Callable[[Series, Any], Series]
    rdiv: Callable[[Series, Any], Series]
    def __init__(
        self,
        data=...,
        index=...,
        dtype: Dtype | None = ...,
        name=...,
        copy: bool = ...,
        fastpath: bool = ...,
    ) -> None: ...
    _index: Index | None = ...
    @property
    def dtype(self) -> DtypeObj: ...
    @property
    def dtypes(self) -> DtypeObj: ...
    @property
    def name(self) -> Hashable: ...
    @name.setter
    def name(self, value: Hashable) -> None: ...
    @property
    def values(self): ...
    @Appender(base.IndexOpsMixin.array.__doc__)
    @property
    def array(self) -> ExtensionArray: ...
    def ravel(self, order=...): ...
    def __len__(self) -> int: ...
    def view(self, dtype: Dtype | None = ...) -> Series: ...
    _HANDLED_TYPES = ...
    def __array__(self, dtype: NpDtype | None = ...) -> np.ndarray: ...
    __float__ = ...
    __long__ = ...
    __int__ = ...
    @property
    def axes(self) -> list[Index]: ...
    @Appender(generic.NDFrame.take.__doc__)
    def take(self, indices, axis=..., is_copy=..., **kwargs) -> Series: ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value) -> None: ...
    def repeat(self, repeats, axis=...) -> Series: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "level"])
    def reset_index(self, level=..., drop=..., name=..., inplace=...): ...
    def __repr__(self) -> str: ...
    def to_string(
        self,
        buf=...,
        na_rep=...,
        float_format=...,
        header=...,
        index=...,
        length=...,
        dtype=...,
        name=...,
        max_rows=...,
        min_rows=...,
    ): ...
    @doc(
        klass=_shared_doc_kwargs["klass"],
        storage_options=generic._shared_docs["storage_options"],
        examples=dedent(
            """
            Examples
            --------
            >>> s = pd.Series(["elk", "pig", "dog", "quetzal"], name="animal")
            >>> print(s.to_markdown())
            |    | animal   |
            |---:|:---------|
            |  0 | elk      |
            |  1 | pig      |
            |  2 | dog      |
            |  3 | quetzal  |
            """
        ),
    )
    def to_markdown(
        self,
        buf: IO[str] | None = ...,
        mode: str = ...,
        index: bool = ...,
        storage_options: StorageOptions = ...,
        **kwargs
    ) -> str | None: ...
    def items(self) -> Iterable[tuple[Hashable, Any]]: ...
    @Appender(items.__doc__)
    def iteritems(self) -> Iterable[tuple[Hashable, Any]]: ...
    def keys(self) -> Index: ...
    def to_dict(self, into=...): ...
    def to_frame(self, name=...) -> DataFrame: ...
    @Appender(
        """
Examples
--------
>>> ser = pd.Series([390., 350., 30., 20.],
...                 index=['Falcon', 'Falcon', 'Parrot', 'Parrot'], name="Max Speed")
>>> ser
Falcon    390.0
Falcon    350.0
Parrot     30.0
Parrot     20.0
Name: Max Speed, dtype: float64
>>> ser.groupby(["a", "b", "a", "b"]).mean()
a    210.0
b    185.0
Name: Max Speed, dtype: float64
>>> ser.groupby(level=0).mean()
Falcon    370.0
Parrot     25.0
Name: Max Speed, dtype: float64
>>> ser.groupby(ser > 100).mean()
Max Speed
False     25.0
True     370.0
Name: Max Speed, dtype: float64
**Grouping by Indexes**
We can groupby different levels of a hierarchical index
using the `level` parameter:
>>> arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot'],
...           ['Captive', 'Wild', 'Captive', 'Wild']]
>>> index = pd.MultiIndex.from_arrays(arrays, names=('Animal', 'Type'))
>>> ser = pd.Series([390., 350., 30., 20.], index=index, name="Max Speed")
>>> ser
Animal  Type
Falcon  Captive    390.0
        Wild       350.0
Parrot  Captive     30.0
        Wild        20.0
Name: Max Speed, dtype: float64
>>> ser.groupby(level=0).mean()
Animal
Falcon    370.0
Parrot     25.0
Name: Max Speed, dtype: float64
>>> ser.groupby(level="Type").mean()
Type
Captive    210.0
Wild       185.0
Name: Max Speed, dtype: float64
We can also choose to include `NA` in group keys or not by defining
`dropna` parameter, the default setting is `True`:
>>> ser = pd.Series([1, 2, 3, 3], index=["a", 'a', 'b', np.nan])
>>> ser.groupby(level=0).sum()
a    3
b    3
dtype: int64
>>> ser.groupby(level=0, dropna=False).sum()
a    3
b    3
NaN  3
dtype: int64
>>> arrays = ['Falcon', 'Falcon', 'Parrot', 'Parrot']
>>> ser = pd.Series([390., 350., 30., 20.], index=arrays, name="Max Speed")
>>> ser.groupby(["a", "b", "a", np.nan]).mean()
a    210.0
b    350.0
Name: Max Speed, dtype: float64
>>> ser.groupby(["a", "b", "a", np.nan], dropna=False).mean()
a    210.0
b    350.0
NaN   20.0
Name: Max Speed, dtype: float64
"""
    )
    @Appender(generic._shared_docs["groupby"] % _shared_doc_kwargs)
    def groupby(
        self,
        by=...,
        axis=...,
        level=...,
        as_index: bool = ...,
        sort: bool = ...,
        group_keys: bool = ...,
        squeeze: bool | lib.NoDefault = ...,
        observed: bool = ...,
        dropna: bool = ...,
    ) -> SeriesGroupBy: ...
    def count(self, level=...): ...
    def mode(self, dropna=...) -> Series: ...
    def unique(self) -> ArrayLike: ...
    @overload
    def drop_duplicates(self, keep=..., inplace: Literal[False] = ...) -> Series: ...
    @overload
    def drop_duplicates(self, keep, inplace: Literal[True]) -> None: ...
    @overload
    def drop_duplicates(self, *, inplace: Literal[True]) -> None: ...
    @overload
    def drop_duplicates(self, keep=..., inplace: bool = ...) -> Series | None: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self"])
    def drop_duplicates(self, keep=..., inplace=...) -> Series | None: ...
    def duplicated(self, keep=...) -> Series: ...
    def idxmin(self, axis=..., skipna=..., *args, **kwargs): ...
    def idxmax(self, axis=..., skipna=..., *args, **kwargs): ...
    def round(self, decimals=..., *args, **kwargs) -> Series: ...
    def quantile(self, q=..., interpolation=...): ...
    def corr(self, other, method=..., min_periods=...) -> float: ...
    def cov(
        self, other: Series, min_periods: int | None = ..., ddof: int | None = ...
    ) -> float: ...
    @doc(
        klass="Series",
        extra_params="",
        other_klass="DataFrame",
        examples=dedent(
            """
        Difference with previous row
        >>> s = pd.Series([1, 1, 2, 3, 5, 8])
        >>> s.diff()
        0    NaN
        1    0.0
        2    1.0
        3    1.0
        4    2.0
        5    3.0
        dtype: float64
        Difference with 3rd previous row
        >>> s.diff(periods=3)
        0    NaN
        1    NaN
        2    NaN
        3    2.0
        4    4.0
        5    6.0
        dtype: float64
        Difference with following row
        >>> s.diff(periods=-1)
        0    0.0
        1   -1.0
        2   -1.0
        3   -2.0
        4   -3.0
        5    NaN
        dtype: float64
        Overflow in input dtype
        >>> s = pd.Series([1, 0], dtype=np.uint8)
        >>> s.diff()
        0      NaN
        1    255.0
        dtype: float64"""
        ),
    )
    def diff(self, periods: int = ...) -> Series: ...
    def autocorr(self, lag=...) -> float: ...
    def dot(self, other): ...
    def __matmul__(self, other): ...
    def __rmatmul__(self, other): ...
    @doc(base.IndexOpsMixin.searchsorted, klass="Series")
    def searchsorted(self, value, side=..., sorter=...) -> np.ndarray: ...
    def append(
        self, to_append, ignore_index: bool = ..., verify_integrity: bool = ...
    ): ...
    @doc(
        generic._shared_docs["compare"],
        """
Returns
-------
Series or DataFrame
    If axis is 0 or 'index' the result will be a Series.
    The resulting index will be a MultiIndex with 'self' and 'other'
    stacked alternately at the inner level.
    If axis is 1 or 'columns' the result will be a DataFrame.
    It will have two columns namely 'self' and 'other'.
See Also
--------
DataFrame.compare : Compare with another DataFrame and show differences.
Notes
-----
Matching NaNs will not appear as a difference.
Examples
--------
>>> s1 = pd.Series(["a", "b", "c", "d", "e"])
>>> s2 = pd.Series(["a", "a", "c", "b", "e"])
Align the differences on columns
>>> s1.compare(s2)
  self other
1    b     a
3    d     b
Stack the differences on indices
>>> s1.compare(s2, align_axis=0)
1  self     b
   other    a
3  self     d
   other    b
dtype: object
Keep all original rows
>>> s1.compare(s2, keep_shape=True)
  self other
0  NaN   NaN
1    b     a
2  NaN   NaN
3    d     b
4  NaN   NaN
Keep all original rows and also all original values
>>> s1.compare(s2, keep_shape=True, keep_equal=True)
  self other
0    a     a
1    b     a
2    c     c
3    d     b
4    e     e
""",
        klass=_shared_doc_kwargs["klass"],
    )
    def compare(
        self,
        other: Series,
        align_axis: Axis = ...,
        keep_shape: bool = ...,
        keep_equal: bool = ...,
    ) -> FrameOrSeriesUnion: ...
    def combine(self, other, func, fill_value=...) -> Series: ...
    def combine_first(self, other) -> Series: ...
    def update(self, other) -> None: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self"])
    def sort_values(
        self,
        axis=...,
        ascending: bool | int | Sequence[bool | int] = ...,
        inplace: bool = ...,
        kind: str = ...,
        na_position: str = ...,
        ignore_index: bool = ...,
        key: ValueKeyFunc = ...,
    ): ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self"])
    def sort_index(
        self,
        axis=...,
        level=...,
        ascending: bool | int | Sequence[bool | int] = ...,
        inplace: bool = ...,
        kind: str = ...,
        na_position: str = ...,
        sort_remaining: bool = ...,
        ignore_index: bool = ...,
        key: IndexKeyFunc = ...,
    ): ...
    def argsort(self, axis=..., kind=..., order=...) -> Series: ...
    def nlargest(self, n=..., keep=...) -> Series: ...
    def nsmallest(self, n: int = ..., keep: str = ...) -> Series: ...
    @doc(
        klass=_shared_doc_kwargs["klass"],
        extra_params=dedent(
            """copy : bool, default True
            Whether to copy underlying data."""
        ),
        examples=dedent(
            """Examples
        --------
        >>> s = pd.Series(
        ...     ["A", "B", "A", "C"],
        ...     index=[
        ...         ["Final exam", "Final exam", "Coursework", "Coursework"],
        ...         ["History", "Geography", "History", "Geography"],
        ...         ["January", "February", "March", "April"],
        ...     ],
        ... )
        >>> s
        Final exam  History     January      A
                    Geography   February     B
        Coursework  History     March        A
                    Geography   April        C
        dtype: object
        In the following example, we will swap the levels of the indices.
        Here, we will swap the levels column-wise, but levels can be swapped row-wise
        in a similar manner. Note that column-wise is the default behaviour.
        By not supplying any arguments for i and j, we swap the last and second to
        last indices.
        >>> s.swaplevel()
        Final exam  January     History         A
                    February    Geography       B
        Coursework  March       History         A
                    April       Geography       C
        dtype: object
        By supplying one argument, we can choose which index to swap the last
        index with. We can for example swap the first index with the last one as
        follows.
        >>> s.swaplevel(0)
        January     History     Final exam      A
        February    Geography   Final exam      B
        March       History     Coursework      A
        April       Geography   Coursework      C
        dtype: object
        We can also define explicitly which indices we want to swap by supplying values
        for both i and j. Here, we for example swap the first and second indices.
        >>> s.swaplevel(0, 1)
        History     Final exam  January         A
        Geography   Final exam  February        B
        History     Coursework  March           A
        Geography   Coursework  April           C
        dtype: object"""
        ),
    )
    def swaplevel(self, i=..., j=..., copy=...) -> Series: ...
    def reorder_levels(self, order) -> Series: ...
    def explode(self, ignore_index: bool = ...) -> Series: ...
    def unstack(self, level=..., fill_value=...) -> DataFrame: ...
    def map(self, arg, na_action=...) -> Series: ...
    _agg_see_also_doc = ...
    _agg_examples_doc = ...
    @doc(
        generic._shared_docs["aggregate"],
        klass=_shared_doc_kwargs["klass"],
        axis=_shared_doc_kwargs["axis"],
        see_also=_agg_see_also_doc,
        examples=_agg_examples_doc,
    )
    def aggregate(self, func=..., axis=..., *args, **kwargs): ...
    agg = ...
    @doc(
        _shared_docs["transform"],
        klass=_shared_doc_kwargs["klass"],
        axis=_shared_doc_kwargs["axis"],
    )
    def transform(
        self, func: AggFuncType, axis: Axis = ..., *args, **kwargs
    ) -> FrameOrSeriesUnion: ...
    def apply(
        self,
        func: AggFuncType,
        convert_dtype: bool = ...,
        args: tuple[Any, ...] = ...,
        **kwargs
    ) -> FrameOrSeriesUnion: ...
    @doc(
        NDFrame.align,
        klass=_shared_doc_kwargs["klass"],
        axes_single_arg=_shared_doc_kwargs["axes_single_arg"],
    )
    def align(
        self,
        other,
        join=...,
        axis=...,
        level=...,
        copy=...,
        fill_value=...,
        method=...,
        limit=...,
        fill_axis=...,
        broadcast_axis=...,
    ): ...
    def rename(
        self, index=..., *, axis=..., copy=..., inplace=..., level=..., errors=...
    ): ...
    @overload
    def set_axis(
        self, labels, axis: Axis = ..., inplace: Literal[False] = ...
    ) -> Series: ...
    @overload
    def set_axis(self, labels, axis: Axis, inplace: Literal[True]) -> None: ...
    @overload
    def set_axis(self, labels, *, inplace: Literal[True]) -> None: ...
    @overload
    def set_axis(
        self, labels, axis: Axis = ..., inplace: bool = ...
    ) -> Series | None: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "labels"])
    @Appender(
        """
        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s
        0    1
        1    2
        2    3
        dtype: int64
        >>> s.set_axis(['a', 'b', 'c'], axis=0)
        a    1
        b    2
        c    3
        dtype: int64
    """
    )
    @Substitution(
        **_shared_doc_kwargs,
        extended_summary_sub="",
        axis_description_sub="",
        see_also_sub=""
    )
    @Appender(generic.NDFrame.set_axis.__doc__)
    def set_axis(self, labels, axis: Axis = ..., inplace: bool = ...): ...
    @doc(
        NDFrame.reindex,
        klass=_shared_doc_kwargs["klass"],
        axes=_shared_doc_kwargs["axes"],
        optional_labels=_shared_doc_kwargs["optional_labels"],
        optional_axis=_shared_doc_kwargs["optional_axis"],
    )
    def reindex(self, index=..., **kwargs): ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "labels"])
    def drop(
        self,
        labels=...,
        axis=...,
        index=...,
        columns=...,
        level=...,
        inplace=...,
        errors=...,
    ) -> Series: ...
    @overload
    def fillna(
        self,
        value=...,
        method: FillnaOptions | None = ...,
        axis: Axis | None = ...,
        inplace: Literal[False] = ...,
        limit=...,
        downcast=...,
    ) -> Series: ...
    @overload
    def fillna(
        self,
        value,
        method: FillnaOptions | None,
        axis: Axis | None,
        inplace: Literal[True],
        limit=...,
        downcast=...,
    ) -> None: ...
    @overload
    def fillna(self, *, inplace: Literal[True], limit=..., downcast=...) -> None: ...
    @overload
    def fillna(
        self, value, *, inplace: Literal[True], limit=..., downcast=...
    ) -> None: ...
    @overload
    def fillna(
        self,
        *,
        method: FillnaOptions | None,
        inplace: Literal[True],
        limit=...,
        downcast=...
    ) -> None: ...
    @overload
    def fillna(
        self, *, axis: Axis | None, inplace: Literal[True], limit=..., downcast=...
    ) -> None: ...
    @overload
    def fillna(
        self,
        *,
        method: FillnaOptions | None,
        axis: Axis | None,
        inplace: Literal[True],
        limit=...,
        downcast=...
    ) -> None: ...
    @overload
    def fillna(
        self,
        value,
        *,
        axis: Axis | None,
        inplace: Literal[True],
        limit=...,
        downcast=...
    ) -> None: ...
    @overload
    def fillna(
        self,
        value,
        method: FillnaOptions | None,
        *,
        inplace: Literal[True],
        limit=...,
        downcast=...
    ) -> None: ...
    @overload
    def fillna(
        self,
        value=...,
        method: FillnaOptions | None = ...,
        axis: Axis | None = ...,
        inplace: bool = ...,
        limit=...,
        downcast=...,
    ) -> Series | None: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "value"])
    @doc(NDFrame.fillna, **_shared_doc_kwargs)
    def fillna(
        self,
        value: object | ArrayLike | None = ...,
        method: FillnaOptions | None = ...,
        axis=...,
        inplace=...,
        limit=...,
        downcast=...,
    ) -> Series | None: ...
    def pop(self, item: Hashable) -> Any: ...
    @doc(
        NDFrame.replace,
        klass=_shared_doc_kwargs["klass"],
        inplace=_shared_doc_kwargs["inplace"],
        replace_iloc=_shared_doc_kwargs["replace_iloc"],
    )
    def replace(
        self, to_replace=..., value=..., inplace=..., limit=..., regex=..., method=...
    ): ...
    @doc(NDFrame.shift, klass=_shared_doc_kwargs["klass"])
    def shift(self, periods=..., freq=..., axis=..., fill_value=...) -> Series: ...
    def memory_usage(self, index: bool = ..., deep: bool = ...) -> int: ...
    def isin(self, values) -> Series: ...
    def between(self, left, right, inclusive=...) -> Series: ...
    @doc(NDFrame.isna, klass=_shared_doc_kwargs["klass"])
    def isna(self) -> Series: ...
    @doc(NDFrame.isna, klass=_shared_doc_kwargs["klass"])
    def isnull(self) -> Series: ...
    @doc(NDFrame.notna, klass=_shared_doc_kwargs["klass"])
    def notna(self) -> Series: ...
    @doc(NDFrame.notna, klass=_shared_doc_kwargs["klass"])
    def notnull(self) -> Series: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self"])
    def dropna(self, axis=..., inplace=..., how=...): ...
    @doc(NDFrame.asfreq, **_shared_doc_kwargs)
    def asfreq(
        self,
        freq,
        method=...,
        how: str | None = ...,
        normalize: bool = ...,
        fill_value=...,
    ) -> Series: ...
    @doc(NDFrame.resample, **_shared_doc_kwargs)
    def resample(
        self,
        rule,
        axis=...,
        closed: str | None = ...,
        label: str | None = ...,
        convention: str = ...,
        kind: str | None = ...,
        loffset=...,
        base: int | None = ...,
        on=...,
        level=...,
        origin: str | TimestampConvertibleTypes = ...,
        offset: TimedeltaConvertibleTypes | None = ...,
    ) -> Resampler: ...
    def to_timestamp(self, freq=..., how=..., copy=...) -> Series: ...
    def to_period(self, freq=..., copy=...) -> Series: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self"])
    def ffill(
        self: Series,
        axis: None | Axis = ...,
        inplace: bool = ...,
        limit: None | int = ...,
        downcast=...,
    ) -> Series | None: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self"])
    def bfill(
        self: Series,
        axis: None | Axis = ...,
        inplace: bool = ...,
        limit: None | int = ...,
        downcast=...,
    ) -> Series | None: ...
    @deprecate_nonkeyword_arguments(
        version=None, allowed_args=["self", "lower", "upper"]
    )
    def clip(
        self: Series,
        lower=...,
        upper=...,
        axis: Axis | None = ...,
        inplace: bool = ...,
        *args,
        **kwargs
    ) -> Series | None: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "method"])
    def interpolate(
        self: Series,
        method: str = ...,
        axis: Axis = ...,
        limit: int | None = ...,
        inplace: bool = ...,
        limit_direction: str | None = ...,
        limit_area: str | None = ...,
        downcast: str | None = ...,
        **kwargs
    ) -> Series | None: ...
    @deprecate_nonkeyword_arguments(
        version=None, allowed_args=["self", "cond", "other"]
    )
    def where(
        self,
        cond,
        other=...,
        inplace=...,
        axis=...,
        level=...,
        errors=...,
        try_cast=...,
    ): ...
    @deprecate_nonkeyword_arguments(
        version=None, allowed_args=["self", "cond", "other"]
    )
    def mask(
        self,
        cond,
        other=...,
        inplace=...,
        axis=...,
        level=...,
        errors=...,
        try_cast=...,
    ): ...
    _AXIS_ORDERS = ...
    _AXIS_REVERSED = ...
    _AXIS_LEN = ...
    _info_axis_number = ...
    _info_axis_name = ...
    index: Index = ...
    str = ...
    dt = ...
    cat = ...
    plot = ...
    sparse = ...
    hist = ...
