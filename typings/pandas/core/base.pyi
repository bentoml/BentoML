import textwrap
from typing import TYPE_CHECKING, Any, Generic, Hashable

import numpy as np
from pandas._typing import Dtype, DtypeObj, FrameOrSeries, IndexLabel, Shape, final
from pandas.core import algorithms
from pandas.core.accessor import DirNamesMixin
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.util._decorators import cache_readonly, doc

"""
Base and utility classes for pandas objects.
"""
if TYPE_CHECKING: ...
_shared_docs: dict[str, str] = ...
_indexops_doc_kwargs = ...
_T = ...

class PandasObject(DirNamesMixin):
    """
    Baseclass for various pandas objects.
    """

    _cache: dict[str, Any]
    def __repr__(self) -> str:
        """
        Return a string representation for a particular object.
        """
        ...
    def __sizeof__(self) -> int:
        """
        Generates the total memory usage for an object that returns
        either a value or Series of values
        """
        ...

class NoNewAttributesMixin:
    """
    Mixin which prevents adding new attributes.

    Prevents additional attributes via xxx.attribute = "something" after a
    call to `self.__freeze()`. Mainly used to prevent the user from using
    wrong attributes on an accessor (`Series.cat/.str/.dt`).

    If you really want to add a new attribute at a later time, you need to use
    `object.__setattr__(self, key, value)`.
    """

    def __setattr__(self, key: str, value): ...

class DataError(Exception): ...
class SpecificationError(Exception): ...

class SelectionMixin(Generic[FrameOrSeries]):
    """
    mixin implementing the selection & aggregation interface on a group-like
    object sub-classes need to define: obj, exclusions
    """

    obj: FrameOrSeries
    _selection: IndexLabel | None = ...
    exclusions: frozenset[Hashable]
    _internal_names = ...
    _internal_names_set = ...
    @final
    @cache_readonly
    def ndim(self) -> int: ...
    def __getitem__(self, key): ...
    def aggregate(self, func, *args, **kwargs): ...
    agg = ...

class IndexOpsMixin(OpsMixin):
    """
    Common ops mixin to support a unified interface / docs for Series / Index
    """

    __array_priority__ = ...
    _hidden_attrs: frozenset[str] = ...
    @property
    def dtype(self) -> DtypeObj: ...
    def transpose(self: _T, *args, **kwargs) -> _T:
        """
        Return the transpose, which is by definition self.

        Returns
        -------
        %(klass)s
        """
        ...
    T = ...
    @property
    def shape(self) -> Shape:
        """
        Return a tuple of the shape of the underlying data.
        """
        ...
    def __len__(self) -> int: ...
    @property
    def ndim(self) -> int:
        """
        Number of dimensions of the underlying data, by definition 1.
        """
        ...
    def item(self):  # -> Any:
        """
        Return the first element of the underlying data as a Python scalar.

        Returns
        -------
        scalar
            The first element of %(klass)s.

        Raises
        ------
        ValueError
            If the data is not length-1.
        """
        ...
    @property
    def nbytes(self) -> int:
        """
        Return the number of bytes in the underlying data.
        """
        ...
    @property
    def size(self) -> int:
        """
        Return the number of elements in the underlying data.
        """
        ...
    @property
    def array(self) -> ExtensionArray:
        """
        The ExtensionArray of the data backing this Series or Index.

        Returns
        -------
        ExtensionArray
            An ExtensionArray of the values stored within. For extension
            types, this is the actual array. For NumPy native types, this
            is a thin (no copy) wrapper around :class:`numpy.ndarray`.

            ``.array`` differs ``.values`` which may require converting the
            data to a different form.

        See Also
        --------
        Index.to_numpy : Similar method that always returns a NumPy array.
        Series.to_numpy : Similar method that always returns a NumPy array.

        Notes
        -----
        This table lays out the different array types for each extension
        dtype within pandas.

        ================== =============================
        dtype              array type
        ================== =============================
        category           Categorical
        period             PeriodArray
        interval           IntervalArray
        IntegerNA          IntegerArray
        string             StringArray
        boolean            BooleanArray
        datetime64[ns, tz] DatetimeArray
        ================== =============================

        For any 3rd-party extension types, the array type will be an
        ExtensionArray.

        For all remaining dtypes ``.array`` will be a
        :class:`arrays.NumpyExtensionArray` wrapping the actual ndarray
        stored within. If you absolutely need a NumPy array (possibly with
        copying / coercing data), then use :meth:`Series.to_numpy` instead.

        Examples
        --------
        For regular NumPy types like int, and float, a PandasArray
        is returned.

        >>> pd.Series([1, 2, 3]).array
        <PandasArray>
        [1, 2, 3]
        Length: 3, dtype: int64

        For extension types, like Categorical, the actual ExtensionArray
        is returned

        >>> ser = pd.Series(pd.Categorical(['a', 'b', 'a']))
        >>> ser.array
        ['a', 'b', 'a']
        Categories (2, object): ['a', 'b']
        """
        ...
    def to_numpy(
        self, dtype: Dtype | None = ..., copy: bool = ..., na_value=..., **kwargs
    ) -> np.ndarray:
        """
        A NumPy ndarray representing the values in this Series or Index.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`.
        copy : bool, default False
            Whether to ensure that the returned value is not a view on
            another array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary.
        na_value : Any, optional
            The value to use for missing values. The default value depends
            on `dtype` and the type of the array.

            .. versionadded:: 1.0.0

        **kwargs
            Additional keywords passed through to the ``to_numpy`` method
            of the underlying array (for extension arrays).

            .. versionadded:: 1.0.0

        Returns
        -------
        numpy.ndarray

        See Also
        --------
        Series.array : Get the actual data stored within.
        Index.array : Get the actual data stored within.
        DataFrame.to_numpy : Similar method for DataFrame.

        Notes
        -----
        The returned array will be the same up to equality (values equal
        in `self` will be equal in the returned array; likewise for values
        that are not equal). When `self` contains an ExtensionArray, the
        dtype may be different. For example, for a category-dtype Series,
        ``to_numpy()`` will return a NumPy array and the categorical dtype
        will be lost.

        For NumPy dtypes, this will be a reference to the actual data stored
        in this Series or Index (assuming ``copy=False``). Modifying the result
        in place will modify the data stored in the Series or Index (not that
        we recommend doing that).

        For extension types, ``to_numpy()`` *may* require copying data and
        coercing the result to a NumPy type (possibly object), which may be
        expensive. When you need a no-copy reference to the underlying data,
        :attr:`Series.array` should be used instead.

        This table lays out the different dtypes and default return types of
        ``to_numpy()`` for various dtypes within pandas.

        ================== ================================
        dtype              array type
        ================== ================================
        category[T]        ndarray[T] (same dtype as input)
        period             ndarray[object] (Periods)
        interval           ndarray[object] (Intervals)
        IntegerNA          ndarray[object]
        datetime64[ns]     datetime64[ns]
        datetime64[ns, tz] ndarray[object] (Timestamps)
        ================== ================================

        Examples
        --------
        >>> ser = pd.Series(pd.Categorical(['a', 'b', 'a']))
        >>> ser.to_numpy()
        array(['a', 'b', 'a'], dtype=object)

        Specify the `dtype` to control how datetime-aware data is represented.
        Use ``dtype=object`` to return an ndarray of pandas :class:`Timestamp`
        objects, each with the correct ``tz``.

        >>> ser = pd.Series(pd.date_range('2000', periods=2, tz="CET"))
        >>> ser.to_numpy(dtype=object)
        array([Timestamp('2000-01-01 00:00:00+0100', tz='CET'),
               Timestamp('2000-01-02 00:00:00+0100', tz='CET')],
              dtype=object)

        Or ``dtype='datetime64[ns]'`` to return an ndarray of native
        datetime64 values. The values are converted to UTC and the timezone
        info is dropped.

        >>> ser.to_numpy(dtype="datetime64[ns]")
        ... # doctest: +ELLIPSIS
        array(['1999-12-31T23:00:00.000000000', '2000-01-01T23:00:00...'],
              dtype='datetime64[ns]')
        """
        ...
    @property
    def empty(self) -> bool: ...
    def max(self, axis=..., skipna: bool = ..., *args, **kwargs):  # -> Dtype:
        """
        Return the maximum value of the Index.

        Parameters
        ----------
        axis : int, optional
            For compatibility with NumPy. Only 0 or None are allowed.
        skipna : bool, default True
            Exclude NA/null values when showing the result.
        *args, **kwargs
            Additional arguments and keywords for compatibility with NumPy.

        Returns
        -------
        scalar
            Maximum value.

        See Also
        --------
        Index.min : Return the minimum value in an Index.
        Series.max : Return the maximum value in a Series.
        DataFrame.max : Return the maximum values in a DataFrame.

        Examples
        --------
        >>> idx = pd.Index([3, 2, 1])
        >>> idx.max()
        3

        >>> idx = pd.Index(['c', 'b', 'a'])
        >>> idx.max()
        'c'

        For a MultiIndex, the maximum is determined lexicographically.

        >>> idx = pd.MultiIndex.from_product([('a', 'b'), (2, 1)])
        >>> idx.max()
        ('b', 2)
        """
        ...
    @doc(op="max", oppose="min", value="largest")
    def argmax(self, axis=..., skipna: bool = ..., *args, **kwargs) -> int:
        """
        Return int position of the {value} value in the Series.

        If the {op}imum is achieved in multiple locations,
        the first row position is returned.

        Parameters
        ----------
        axis : {{None}}
            Dummy argument for consistency with Series.
        skipna : bool, default True
            Exclude NA/null values when showing the result.
        *args, **kwargs
            Additional arguments and keywords for compatibility with NumPy.

        Returns
        -------
        int
            Row position of the {op}imum value.

        See Also
        --------
        Series.arg{op} : Return position of the {op}imum value.
        Series.arg{oppose} : Return position of the {oppose}imum value.
        numpy.ndarray.arg{op} : Equivalent method for numpy arrays.
        Series.idxmax : Return index label of the maximum values.
        Series.idxmin : Return index label of the minimum values.

        Examples
        --------
        Consider dataset containing cereal calories

        >>> s = pd.Series({{'Corn Flakes': 100.0, 'Almond Delight': 110.0,
        ...                'Cinnamon Toast Crunch': 120.0, 'Cocoa Puff': 110.0}})
        >>> s
        Corn Flakes              100.0
        Almond Delight           110.0
        Cinnamon Toast Crunch    120.0
        Cocoa Puff               110.0
        dtype: float64

        >>> s.argmax()
        2
        >>> s.argmin()
        0

        The maximum cereal calories is the third element and
        the minimum cereal calories is the first element,
        since series is zero-indexed.
        """
        ...
    def min(self, axis=..., skipna: bool = ..., *args, **kwargs):  # -> Dtype:
        """
        Return the minimum value of the Index.

        Parameters
        ----------
        axis : {None}
            Dummy argument for consistency with Series.
        skipna : bool, default True
            Exclude NA/null values when showing the result.
        *args, **kwargs
            Additional arguments and keywords for compatibility with NumPy.

        Returns
        -------
        scalar
            Minimum value.

        See Also
        --------
        Index.max : Return the maximum value of the object.
        Series.min : Return the minimum value in a Series.
        DataFrame.min : Return the minimum values in a DataFrame.

        Examples
        --------
        >>> idx = pd.Index([3, 2, 1])
        >>> idx.min()
        1

        >>> idx = pd.Index(['c', 'b', 'a'])
        >>> idx.min()
        'a'

        For a MultiIndex, the minimum is determined lexicographically.

        >>> idx = pd.MultiIndex.from_product([('a', 'b'), (2, 1)])
        >>> idx.min()
        ('a', 1)
        """
        ...
    @doc(argmax, op="min", oppose="max", value="smallest")
    def argmin(self, axis=..., skipna=..., *args, **kwargs) -> int: ...
    def tolist(self):  # -> list[Any] | Any:
        """
        Return a list of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        list

        See Also
        --------
        numpy.ndarray.tolist : Return the array as an a.ndim-levels deep
            nested list of Python scalars.
        """
        ...
    to_list = ...
    def __iter__(self):  # -> Iterator[Any] | map[Unknown]:
        """
        Return an iterator of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        iterator
        """
        ...
    @cache_readonly
    def hasnans(self) -> bool:
        """
        Return if I have any nans; enables various perf speedups.
        """
        ...
    def isna(self): ...
    def value_counts(
        self,
        normalize: bool = ...,
        sort: bool = ...,
        ascending: bool = ...,
        bins=...,
        dropna: bool = ...,
    ):  # -> Series:
        """
        Return a Series containing counts of unique values.

        The resulting object will be in descending order so that the
        first element is the most frequently-occurring element.
        Excludes NA values by default.

        Parameters
        ----------
        normalize : bool, default False
            If True then the object returned will contain the relative
            frequencies of the unique values.
        sort : bool, default True
            Sort by frequencies.
        ascending : bool, default False
            Sort in ascending order.
        bins : int, optional
            Rather than count values, group them into half-open bins,
            a convenience for ``pd.cut``, only works with numeric data.
        dropna : bool, default True
            Don't include counts of NaN.

        Returns
        -------
        Series

        See Also
        --------
        Series.count: Number of non-NA elements in a Series.
        DataFrame.count: Number of non-NA elements in a DataFrame.
        DataFrame.value_counts: Equivalent method on DataFrames.

        Examples
        --------
        >>> index = pd.Index([3, 1, 2, 3, 4, np.nan])
        >>> index.value_counts()
        3.0    2
        1.0    1
        2.0    1
        4.0    1
        dtype: int64

        With `normalize` set to `True`, returns the relative frequency by
        dividing all values by the sum of values.

        >>> s = pd.Series([3, 1, 2, 3, 4, np.nan])
        >>> s.value_counts(normalize=True)
        3.0    0.4
        1.0    0.2
        2.0    0.2
        4.0    0.2
        dtype: float64

        **bins**

        Bins can be useful for going from a continuous variable to a
        categorical variable; instead of counting unique
        apparitions of values, divide the index in the specified
        number of half-open bins.

        >>> s.value_counts(bins=3)
        (0.996, 2.0]    2
        (2.0, 3.0]      2
        (3.0, 4.0]      1
        dtype: int64

        **dropna**

        With `dropna` set to `False` we can also see NaN index values.

        >>> s.value_counts(dropna=False)
        3.0    2
        1.0    1
        2.0    1
        4.0    1
        NaN    1
        dtype: int64
        """
        ...
    def unique(self): ...
    def nunique(self, dropna: bool = ...) -> int:
        """
        Return number of unique elements in the object.

        Excludes NA values by default.

        Parameters
        ----------
        dropna : bool, default True
            Don't include NaN in the count.

        Returns
        -------
        int

        See Also
        --------
        DataFrame.nunique: Method nunique for DataFrame.
        Series.count: Count non-NA/null observations in the Series.

        Examples
        --------
        >>> s = pd.Series([1, 3, 5, 7, 7])
        >>> s
        0    1
        1    3
        2    5
        3    7
        4    7
        dtype: int64

        >>> s.nunique()
        4
        """
        ...
    @property
    def is_unique(self) -> bool:
        """
        Return boolean if values in the object are unique.

        Returns
        -------
        bool
        """
        ...
    @property
    def is_monotonic(self) -> bool:
        """
        Return boolean if values in the object are
        monotonic_increasing.

        Returns
        -------
        bool
        """
        ...
    @property
    def is_monotonic_increasing(self) -> bool:
        """
        Alias for is_monotonic.
        """
        ...
    @property
    def is_monotonic_decreasing(self) -> bool:
        """
        Return boolean if values in the object are
        monotonic_decreasing.

        Returns
        -------
        bool
        """
        ...
    @doc(
        algorithms.factorize,
        values="",
        order="",
        size_hint="",
        sort=textwrap.dedent(
            """\
            sort : bool, default False
                Sort `uniques` and shuffle `codes` to maintain the
                relationship.
            """
        ),
    )
    def factorize(self, sort: bool = ..., na_sentinel: int | None = ...): ...
    @doc(_shared_docs["searchsorted"], klass="Index")
    def searchsorted(self, value, side=..., sorter=...) -> np.ndarray: ...
    def drop_duplicates(self, keep=...): ...
