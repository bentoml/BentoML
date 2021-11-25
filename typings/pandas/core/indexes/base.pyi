from typing import TYPE_CHECKING, Any, Callable, Hashable, Literal, Sequence

import numpy as np
from pandas import DataFrame, RangeIndex, Series
from pandas._libs import index as libindex
from pandas._libs.tslibs import NaTType
from pandas._typing import AnyArrayLike, ArrayLike, Dtype, DtypeObj, Shape, final
from pandas.core.arrays import ExtensionArray
from pandas.core.base import IndexOpsMixin, PandasObject
from pandas.io.formats.printing import PrettyDict
from pandas.util._decorators import (
    Appender,
    cache_readonly,
    deprecate_nonkeyword_arguments,
    doc,
)

if TYPE_CHECKING: ...
__all__ = ["Index"]
_unsortable_types = ...
_index_doc_kwargs: dict[str, str] = ...
_index_shared_docs: dict[str, str] = ...
str_t = str
_o_dtype = ...

def disallow_kwargs(kwargs: dict[str, Any]) -> None: ...

_IndexT = ...

class Index(IndexOpsMixin, PandasObject):
    """
    Immutable sequence used for indexing and alignment. The basic object
    storing axis labels for all pandas objects.

    Parameters
    ----------
    data : array-like (1-dimensional)
    dtype : NumPy dtype (default: object)
        If dtype is None, we find the dtype that best fits the data.
        If an actual dtype is provided, we coerce to that dtype if it's safe.
        Otherwise, an error will be raised.
    copy : bool
        Make a copy of input ndarray.
    name : object
        Name to be stored in the index.
    tupleize_cols : bool (default: True)
        When True, attempt to create a MultiIndex if possible.

    See Also
    --------
    RangeIndex : Index implementing a monotonic integer range.
    CategoricalIndex : Index of :class:`Categorical` s.
    MultiIndex : A multi-level, or hierarchical Index.
    IntervalIndex : An Index of :class:`Interval` s.
    DatetimeIndex : Index of datetime64 data.
    TimedeltaIndex : Index of timedelta64 data.
    PeriodIndex : Index of Period data.
    Int64Index : A special case of :class:`Index` with purely integer labels.
    UInt64Index : A special case of :class:`Index` with purely unsigned integer labels.
    Float64Index : A special case of :class:`Index` with purely float labels.

    Notes
    -----
    An Index instance can **only** contain hashable objects

    Examples
    --------
    >>> pd.Index([1, 2, 3])
    Int64Index([1, 2, 3], dtype='int64')

    >>> pd.Index(list('abc'))
    Index(['a', 'b', 'c'], dtype='object')
    """

    _hidden_attrs: frozenset[str] = ...
    _join_precedence = ...
    _typ: str = ...
    _data: ExtensionArray | np.ndarray
    _id: object | None = ...
    _name: Hashable = ...
    _no_setting_name: bool = ...
    _comparables: list[str] = ...
    _attributes: list[str] = ...
    _is_numeric_dtype: bool = ...
    _can_hold_na: bool = ...
    _can_hold_strings: bool = ...
    _engine_type: type[libindex.IndexEngine] = ...
    _supports_partial_string_indexing = ...
    _accessors = ...
    str = ...
    def __new__(
        cls, data=..., dtype=..., copy=..., name=..., tupleize_cols=..., **kwargs
    ) -> Index: ...
    @property
    def asi8(self):  # -> None:
        """
        Integer representation of the values.

        Returns
        -------
        ndarray
            An ndarray with int64 dtype.
        """
        ...
    @final
    def is_(self, other) -> bool:
        """
        More flexible, faster check like ``is`` but that works through views.

        Note: this is *not* the same as ``Index.identical()``, which checks
        that metadata is also the same.

        Parameters
        ----------
        other : object
            Other object to compare against.

        Returns
        -------
        bool
            True if both have same underlying data, False otherwise.

        See Also
        --------
        Index.identical : Works like ``Index.is_`` but also checks metadata.
        """
        ...
    def __len__(self) -> int:
        """
        Return the length of the Index.
        """
        ...
    def __array__(self, dtype=...) -> np.ndarray:
        """
        The array interface, return my values.
        """
        ...
    def __array_wrap__(self, result, context=...):  # -> object | Index:
        """
        Gets called after a ufunc and other functions.
        """
        ...
    @cache_readonly
    def dtype(self) -> DtypeObj:
        """
        Return the dtype object of the underlying data.
        """
        ...
    @final
    def ravel(self, order=...):  # -> ndarray:
        """
        Return an ndarray of the flattened values of the underlying data.

        Returns
        -------
        numpy.ndarray
            Flattened array.

        See Also
        --------
        numpy.ndarray.ravel : Return a flattened array.
        """
        ...
    def view(self, cls=...): ...
    def astype(self, dtype, copy=...):  # -> Self@Index | Index:
        """
        Create an Index with values cast to dtypes.

        The class of a new Index is determined by dtype. When conversion is
        impossible, a TypeError exception is raised.

        Parameters
        ----------
        dtype : numpy dtype or pandas type
            Note that any signed integer `dtype` is treated as ``'int64'``,
            and any unsigned integer `dtype` is treated as ``'uint64'``,
            regardless of the size.
        copy : bool, default True
            By default, astype always returns a newly allocated object.
            If copy is set to False and internal requirements on dtype are
            satisfied, the original data is used to create a new Index
            or the original Index is returned.

        Returns
        -------
        Index
            Index with values cast to specified dtype.
        """
        ...
    @Appender(_index_shared_docs["take"] % _index_doc_kwargs)
    def take(
        self, indices, axis: int = ..., allow_fill: bool = ..., fill_value=..., **kwargs
    ): ...
    @Appender(_index_shared_docs["repeat"] % _index_doc_kwargs)
    def repeat(self, repeats, axis=...): ...
    def copy(
        self: _IndexT,
        name: Hashable | None = ...,
        deep: bool = ...,
        dtype: Dtype | None = ...,
        names: Sequence[Hashable] | None = ...,
    ) -> _IndexT:
        """
        Make a copy of this object.

        Name and dtype sets those attributes on the new object.

        Parameters
        ----------
        name : Label, optional
            Set name for new object.
        deep : bool, default False
        dtype : numpy dtype or pandas type, optional
            Set dtype for new object.

            .. deprecated:: 1.2.0
                use ``astype`` method instead.
        names : list-like, optional
            Kept for compatibility with MultiIndex. Should not be used.

        Returns
        -------
        Index
            Index refer to new object which is a copy of this object.

        Notes
        -----
        In most cases, there should be no functional difference from using
        ``deep``, but if ``deep`` is passed it will attempt to deepcopy.
        """
        ...
    @final
    def __copy__(self: _IndexT, **kwargs) -> _IndexT: ...
    @final
    def __deepcopy__(self: _IndexT, memo=...) -> _IndexT:
        """
        Parameters
        ----------
        memo, default None
            Standard signature. Unused
        """
        ...
    @final
    def __repr__(self) -> str_t:
        """
        Return a string representation for this object.
        """
        ...
    def format(
        self, name: bool = ..., formatter: Callable | None = ..., na_rep: str_t = ...
    ) -> list[str_t]:
        """
        Render a string representation of the Index.
        """
        ...
    @final
    def to_native_types(self, slicer=..., **kwargs) -> np.ndarray:
        """
        Format specified values of `self` and return them.

        .. deprecated:: 1.2.0

        Parameters
        ----------
        slicer : int, array-like
            An indexer into `self` that specifies which values
            are used in the formatting process.
        kwargs : dict
            Options for specifying how the values should be formatted.
            These options include the following:

            1) na_rep : str
                The value that serves as a placeholder for NULL values
            2) quoting : bool or None
                Whether or not there are quoted values in `self`
            3) date_format : str
                The format used to represent date-like values.

        Returns
        -------
        numpy.ndarray
            Formatted values.
        """
        ...
    def to_flat_index(self):  # -> Self@Index:
        """
        Identity method.

        This is implemented for compatibility with subclass implementations
        when chaining.

        Returns
        -------
        pd.Index
            Caller.

        See Also
        --------
        MultiIndex.to_flat_index : Subclass implementation.
        """
        ...
    def to_series(self, index=..., name: Hashable = ...) -> Series:
        """
        Create a Series with both index and values equal to the index keys.

        Useful with map for returning an indexer based on an index.

        Parameters
        ----------
        index : Index, optional
            Index of resulting Series. If None, defaults to original index.
        name : str, optional
            Name of resulting Series. If None, defaults to name of original
            index.

        Returns
        -------
        Series
            The dtype will be based on the type of the Index values.

        See Also
        --------
        Index.to_frame : Convert an Index to a DataFrame.
        Series.to_frame : Convert Series to DataFrame.

        Examples
        --------
        >>> idx = pd.Index(['Ant', 'Bear', 'Cow'], name='animal')

        By default, the original Index and original name is reused.

        >>> idx.to_series()
        animal
        Ant      Ant
        Bear    Bear
        Cow      Cow
        Name: animal, dtype: object

        To enforce a new Index, specify new labels to ``index``:

        >>> idx.to_series(index=[0, 1, 2])
        0     Ant
        1    Bear
        2     Cow
        Name: animal, dtype: object

        To override the name of the resulting column, specify `name`:

        >>> idx.to_series(name='zoo')
        animal
        Ant      Ant
        Bear    Bear
        Cow      Cow
        Name: zoo, dtype: object
        """
        ...
    def to_frame(self, index: bool = ..., name: Hashable = ...) -> DataFrame:
        """
        Create a DataFrame with a column containing the Index.

        Parameters
        ----------
        index : bool, default True
            Set the index of the returned DataFrame as the original Index.

        name : object, default None
            The passed name should substitute for the index name (if it has
            one).

        Returns
        -------
        DataFrame
            DataFrame containing the original Index data.

        See Also
        --------
        Index.to_series : Convert an Index to a Series.
        Series.to_frame : Convert Series to DataFrame.

        Examples
        --------
        >>> idx = pd.Index(['Ant', 'Bear', 'Cow'], name='animal')
        >>> idx.to_frame()
               animal
        animal
        Ant       Ant
        Bear     Bear
        Cow       Cow

        By default, the original Index is reused. To enforce a new Index:

        >>> idx.to_frame(index=False)
            animal
        0   Ant
        1  Bear
        2   Cow

        To override the name of the resulting column, specify `name`:

        >>> idx.to_frame(index=False, name='zoo')
            zoo
        0   Ant
        1  Bear
        2   Cow
        """
        ...
    @property
    def name(self):  # -> Hashable:
        """
        Return Index or MultiIndex name.
        """
        ...
    @name.setter
    def name(self, value: Hashable): ...
    names = ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "names"])
    def set_names(
        self, names, level=..., inplace: bool = ...
    ):  # -> ABCMultiIndex | Self@Index | None:
        """
        Set Index or MultiIndex name.

        Able to set new names partially and by level.

        Parameters
        ----------

        names : label or list of label or dict-like for MultiIndex
            Name(s) to set.

            .. versionchanged:: 1.3.0

        level : int, label or list of int or label, optional
            If the index is a MultiIndex and names is not dict-like, level(s) to set
            (None for all levels). Otherwise level must be None.

            .. versionchanged:: 1.3.0

        inplace : bool, default False
            Modifies the object directly, instead of creating a new Index or
            MultiIndex.

        Returns
        -------
        Index or None
            The same type as the caller or None if ``inplace=True``.

        See Also
        --------
        Index.rename : Able to set new names without level.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx
        Int64Index([1, 2, 3, 4], dtype='int64')
        >>> idx.set_names('quarter')
        Int64Index([1, 2, 3, 4], dtype='int64', name='quarter')

        >>> idx = pd.MultiIndex.from_product([['python', 'cobra'],
        ...                                   [2018, 2019]])
        >>> idx
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   )
        >>> idx.set_names(['kind', 'year'], inplace=True)
        >>> idx
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   names=['kind', 'year'])
        >>> idx.set_names('species', level=0)
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   names=['species', 'year'])

        When renaming levels with a dict, levels can not be passed.

        >>> idx.set_names({'kind': 'snake'})
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   names=['snake', 'year'])
        """
        ...
    def rename(self, name, inplace=...):  # -> ABCMultiIndex | Self@Index | None:
        """
        Alter Index or MultiIndex name.

        Able to set new names without level. Defaults to returning new index.
        Length of names must match number of levels in MultiIndex.

        Parameters
        ----------
        name : label or list of labels
            Name(s) to set.
        inplace : bool, default False
            Modifies the object directly, instead of creating a new Index or
            MultiIndex.

        Returns
        -------
        Index or None
            The same type as the caller or None if ``inplace=True``.

        See Also
        --------
        Index.set_names : Able to set new names partially and by level.

        Examples
        --------
        >>> idx = pd.Index(['A', 'C', 'A', 'B'], name='score')
        >>> idx.rename('grade')
        Index(['A', 'C', 'A', 'B'], dtype='object', name='grade')

        >>> idx = pd.MultiIndex.from_product([['python', 'cobra'],
        ...                                   [2018, 2019]],
        ...                                   names=['kind', 'year'])
        >>> idx
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   names=['kind', 'year'])
        >>> idx.rename(['species', 'year'])
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   names=['species', 'year'])
        >>> idx.rename('species')
        Traceback (most recent call last):
        TypeError: Must pass list-like as `names`.
        """
        ...
    @property
    def nlevels(self) -> int:
        """
        Number of levels.
        """
        ...
    def sortlevel(
        self, level=..., ascending=..., sort_remaining=...
    ):  # -> tuple[Self@Index | MultiIndex, Unknown | ndarray | Any] | Self@Index | MultiIndex:
        """
        For internal compatibility with the Index API.

        Sort the Index. This is for compat with MultiIndex

        Parameters
        ----------
        ascending : bool, default True
            False to sort in descending order

        level, sort_remaining are compat parameters

        Returns
        -------
        Index
        """
        ...
    get_level_values = ...
    @final
    def droplevel(self, level=...):  # -> Self@Index | MultiIndex:
        """
        Return index with requested level(s) removed.

        If resulting index has only 1 level left, the result will be
        of Index type, not MultiIndex.

        Parameters
        ----------
        level : int, str, or list-like, default 0
            If a string is given, must be the name of a level
            If list-like, elements must be names or indexes of levels.

        Returns
        -------
        Index or MultiIndex

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays(
        ... [[1, 2], [3, 4], [5, 6]], names=['x', 'y', 'z'])
        >>> mi
        MultiIndex([(1, 3, 5),
                    (2, 4, 6)],
                   names=['x', 'y', 'z'])

        >>> mi.droplevel()
        MultiIndex([(3, 5),
                    (4, 6)],
                   names=['y', 'z'])

        >>> mi.droplevel(2)
        MultiIndex([(1, 3),
                    (2, 4)],
                   names=['x', 'y'])

        >>> mi.droplevel('z')
        MultiIndex([(1, 3),
                    (2, 4)],
                   names=['x', 'y'])

        >>> mi.droplevel(['x', 'y'])
        Int64Index([5, 6], dtype='int64', name='z')
        """
        ...
    @final
    @property
    def is_monotonic(self) -> bool:
        """
        Alias for is_monotonic_increasing.
        """
        ...
    @property
    def is_monotonic_increasing(self) -> bool:
        """
        Return if the index is monotonic increasing (only equal or
        increasing) values.

        Examples
        --------
        >>> Index([1, 2, 3]).is_monotonic_increasing
        True
        >>> Index([1, 2, 2]).is_monotonic_increasing
        True
        >>> Index([1, 3, 2]).is_monotonic_increasing
        False
        """
        ...
    @property
    def is_monotonic_decreasing(self) -> bool:
        """
        Return if the index is monotonic decreasing (only equal or
        decreasing) values.

        Examples
        --------
        >>> Index([3, 2, 1]).is_monotonic_decreasing
        True
        >>> Index([3, 2, 2]).is_monotonic_decreasing
        True
        >>> Index([3, 1, 2]).is_monotonic_decreasing
        False
        """
        ...
    @cache_readonly
    def is_unique(self) -> bool:
        """
        Return if the index has unique values.
        """
        ...
    @final
    @property
    def has_duplicates(self) -> bool:
        """
        Check if the Index has duplicate values.

        Returns
        -------
        bool
            Whether or not the Index has duplicate values.

        Examples
        --------
        >>> idx = pd.Index([1, 5, 7, 7])
        >>> idx.has_duplicates
        True

        >>> idx = pd.Index([1, 5, 7])
        >>> idx.has_duplicates
        False

        >>> idx = pd.Index(["Watermelon", "Orange", "Apple",
        ...                 "Watermelon"]).astype("category")
        >>> idx.has_duplicates
        True

        >>> idx = pd.Index(["Orange", "Apple",
        ...                 "Watermelon"]).astype("category")
        >>> idx.has_duplicates
        False
        """
        ...
    @final
    def is_boolean(self) -> bool:
        """
        Check if the Index only consists of booleans.

        Returns
        -------
        bool
            Whether or not the Index only consists of booleans.

        See Also
        --------
        is_integer : Check if the Index only consists of integers.
        is_floating : Check if the Index is a floating type.
        is_numeric : Check if the Index only consists of numeric data.
        is_object : Check if the Index is of the object dtype.
        is_categorical : Check if the Index holds categorical data.
        is_interval : Check if the Index holds Interval objects.
        is_mixed : Check if the Index holds data with mixed data types.

        Examples
        --------
        >>> idx = pd.Index([True, False, True])
        >>> idx.is_boolean()
        True

        >>> idx = pd.Index(["True", "False", "True"])
        >>> idx.is_boolean()
        False

        >>> idx = pd.Index([True, False, "True"])
        >>> idx.is_boolean()
        False
        """
        ...
    @final
    def is_integer(self) -> bool:
        """
        Check if the Index only consists of integers.

        Returns
        -------
        bool
            Whether or not the Index only consists of integers.

        See Also
        --------
        is_boolean : Check if the Index only consists of booleans.
        is_floating : Check if the Index is a floating type.
        is_numeric : Check if the Index only consists of numeric data.
        is_object : Check if the Index is of the object dtype.
        is_categorical : Check if the Index holds categorical data.
        is_interval : Check if the Index holds Interval objects.
        is_mixed : Check if the Index holds data with mixed data types.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx.is_integer()
        True

        >>> idx = pd.Index([1.0, 2.0, 3.0, 4.0])
        >>> idx.is_integer()
        False

        >>> idx = pd.Index(["Apple", "Mango", "Watermelon"])
        >>> idx.is_integer()
        False
        """
        ...
    @final
    def is_floating(self) -> bool:
        """
        Check if the Index is a floating type.

        The Index may consist of only floats, NaNs, or a mix of floats,
        integers, or NaNs.

        Returns
        -------
        bool
            Whether or not the Index only consists of only consists of floats, NaNs, or
            a mix of floats, integers, or NaNs.

        See Also
        --------
        is_boolean : Check if the Index only consists of booleans.
        is_integer : Check if the Index only consists of integers.
        is_numeric : Check if the Index only consists of numeric data.
        is_object : Check if the Index is of the object dtype.
        is_categorical : Check if the Index holds categorical data.
        is_interval : Check if the Index holds Interval objects.
        is_mixed : Check if the Index holds data with mixed data types.

        Examples
        --------
        >>> idx = pd.Index([1.0, 2.0, 3.0, 4.0])
        >>> idx.is_floating()
        True

        >>> idx = pd.Index([1.0, 2.0, np.nan, 4.0])
        >>> idx.is_floating()
        True

        >>> idx = pd.Index([1, 2, 3, 4, np.nan])
        >>> idx.is_floating()
        True

        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx.is_floating()
        False
        """
        ...
    @final
    def is_numeric(self) -> bool:
        """
        Check if the Index only consists of numeric data.

        Returns
        -------
        bool
            Whether or not the Index only consists of numeric data.

        See Also
        --------
        is_boolean : Check if the Index only consists of booleans.
        is_integer : Check if the Index only consists of integers.
        is_floating : Check if the Index is a floating type.
        is_object : Check if the Index is of the object dtype.
        is_categorical : Check if the Index holds categorical data.
        is_interval : Check if the Index holds Interval objects.
        is_mixed : Check if the Index holds data with mixed data types.

        Examples
        --------
        >>> idx = pd.Index([1.0, 2.0, 3.0, 4.0])
        >>> idx.is_numeric()
        True

        >>> idx = pd.Index([1, 2, 3, 4.0])
        >>> idx.is_numeric()
        True

        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx.is_numeric()
        True

        >>> idx = pd.Index([1, 2, 3, 4.0, np.nan])
        >>> idx.is_numeric()
        True

        >>> idx = pd.Index([1, 2, 3, 4.0, np.nan, "Apple"])
        >>> idx.is_numeric()
        False
        """
        ...
    @final
    def is_object(self) -> bool:
        """
        Check if the Index is of the object dtype.

        Returns
        -------
        bool
            Whether or not the Index is of the object dtype.

        See Also
        --------
        is_boolean : Check if the Index only consists of booleans.
        is_integer : Check if the Index only consists of integers.
        is_floating : Check if the Index is a floating type.
        is_numeric : Check if the Index only consists of numeric data.
        is_categorical : Check if the Index holds categorical data.
        is_interval : Check if the Index holds Interval objects.
        is_mixed : Check if the Index holds data with mixed data types.

        Examples
        --------
        >>> idx = pd.Index(["Apple", "Mango", "Watermelon"])
        >>> idx.is_object()
        True

        >>> idx = pd.Index(["Apple", "Mango", 2.0])
        >>> idx.is_object()
        True

        >>> idx = pd.Index(["Watermelon", "Orange", "Apple",
        ...                 "Watermelon"]).astype("category")
        >>> idx.is_object()
        False

        >>> idx = pd.Index([1.0, 2.0, 3.0, 4.0])
        >>> idx.is_object()
        False
        """
        ...
    @final
    def is_categorical(self) -> bool:
        """
        Check if the Index holds categorical data.

        Returns
        -------
        bool
            True if the Index is categorical.

        See Also
        --------
        CategoricalIndex : Index for categorical data.
        is_boolean : Check if the Index only consists of booleans.
        is_integer : Check if the Index only consists of integers.
        is_floating : Check if the Index is a floating type.
        is_numeric : Check if the Index only consists of numeric data.
        is_object : Check if the Index is of the object dtype.
        is_interval : Check if the Index holds Interval objects.
        is_mixed : Check if the Index holds data with mixed data types.

        Examples
        --------
        >>> idx = pd.Index(["Watermelon", "Orange", "Apple",
        ...                 "Watermelon"]).astype("category")
        >>> idx.is_categorical()
        True

        >>> idx = pd.Index([1, 3, 5, 7])
        >>> idx.is_categorical()
        False

        >>> s = pd.Series(["Peter", "Victor", "Elisabeth", "Mar"])
        >>> s
        0        Peter
        1       Victor
        2    Elisabeth
        3          Mar
        dtype: object
        >>> s.index.is_categorical()
        False
        """
        ...
    @final
    def is_interval(self) -> bool:
        """
        Check if the Index holds Interval objects.

        Returns
        -------
        bool
            Whether or not the Index holds Interval objects.

        See Also
        --------
        IntervalIndex : Index for Interval objects.
        is_boolean : Check if the Index only consists of booleans.
        is_integer : Check if the Index only consists of integers.
        is_floating : Check if the Index is a floating type.
        is_numeric : Check if the Index only consists of numeric data.
        is_object : Check if the Index is of the object dtype.
        is_categorical : Check if the Index holds categorical data.
        is_mixed : Check if the Index holds data with mixed data types.

        Examples
        --------
        >>> idx = pd.Index([pd.Interval(left=0, right=5),
        ...                 pd.Interval(left=5, right=10)])
        >>> idx.is_interval()
        True

        >>> idx = pd.Index([1, 3, 5, 7])
        >>> idx.is_interval()
        False
        """
        ...
    @final
    def is_mixed(self) -> bool:
        """
        Check if the Index holds data with mixed data types.

        Returns
        -------
        bool
            Whether or not the Index holds data with mixed data types.

        See Also
        --------
        is_boolean : Check if the Index only consists of booleans.
        is_integer : Check if the Index only consists of integers.
        is_floating : Check if the Index is a floating type.
        is_numeric : Check if the Index only consists of numeric data.
        is_object : Check if the Index is of the object dtype.
        is_categorical : Check if the Index holds categorical data.
        is_interval : Check if the Index holds Interval objects.

        Examples
        --------
        >>> idx = pd.Index(['a', np.nan, 'b'])
        >>> idx.is_mixed()
        True

        >>> idx = pd.Index([1.0, 2.0, 3.0, 5.0])
        >>> idx.is_mixed()
        False
        """
        ...
    @final
    def holds_integer(self) -> bool:
        """
        Whether the type is an integer type.
        """
        ...
    @cache_readonly
    def inferred_type(self) -> str_t:
        """
        Return a string of the type inferred from the values.
        """
        ...
    @cache_readonly
    @final
    def is_all_dates(self) -> bool:
        """
        Whether or not the index values only consist of dates.
        """
        ...
    def __reduce__(self): ...
    _na_value: float | NaTType = ...
    @cache_readonly
    def hasnans(self) -> bool:
        """
        Return if I have any nans; enables various perf speedups.
        """
        ...
    @final
    def isna(self) -> np.ndarray:
        """
        Detect missing values.

        Return a boolean same-sized object indicating if the values are NA.
        NA values, such as ``None``, :attr:`numpy.NaN` or :attr:`pd.NaT`, get
        mapped to ``True`` values.
        Everything else get mapped to ``False`` values. Characters such as
        empty strings `''` or :attr:`numpy.inf` are not considered NA values
        (unless you set ``pandas.options.mode.use_inf_as_na = True``).

        Returns
        -------
        numpy.ndarray[bool]
            A boolean array of whether my values are NA.

        See Also
        --------
        Index.notna : Boolean inverse of isna.
        Index.dropna : Omit entries with missing values.
        isna : Top-level isna.
        Series.isna : Detect missing values in Series object.

        Examples
        --------
        Show which entries in a pandas.Index are NA. The result is an
        array.

        >>> idx = pd.Index([5.2, 6.0, np.NaN])
        >>> idx
        Float64Index([5.2, 6.0, nan], dtype='float64')
        >>> idx.isna()
        array([False, False,  True])

        Empty strings are not considered NA values. None is considered an NA
        value.

        >>> idx = pd.Index(['black', '', 'red', None])
        >>> idx
        Index(['black', '', 'red', None], dtype='object')
        >>> idx.isna()
        array([False, False, False,  True])

        For datetimes, `NaT` (Not a Time) is considered as an NA value.

        >>> idx = pd.DatetimeIndex([pd.Timestamp('1940-04-25'),
        ...                         pd.Timestamp(''), None, pd.NaT])
        >>> idx
        DatetimeIndex(['1940-04-25', 'NaT', 'NaT', 'NaT'],
                      dtype='datetime64[ns]', freq=None)
        >>> idx.isna()
        array([False,  True,  True,  True])
        """
        ...
    isnull = ...
    @final
    def notna(self) -> np.ndarray:
        """
        Detect existing (non-missing) values.

        Return a boolean same-sized object indicating if the values are not NA.
        Non-missing values get mapped to ``True``. Characters such as empty
        strings ``''`` or :attr:`numpy.inf` are not considered NA values
        (unless you set ``pandas.options.mode.use_inf_as_na = True``).
        NA values, such as None or :attr:`numpy.NaN`, get mapped to ``False``
        values.

        Returns
        -------
        numpy.ndarray[bool]
            Boolean array to indicate which entries are not NA.

        See Also
        --------
        Index.notnull : Alias of notna.
        Index.isna: Inverse of notna.
        notna : Top-level notna.

        Examples
        --------
        Show which entries in an Index are not NA. The result is an
        array.

        >>> idx = pd.Index([5.2, 6.0, np.NaN])
        >>> idx
        Float64Index([5.2, 6.0, nan], dtype='float64')
        >>> idx.notna()
        array([ True,  True, False])

        Empty strings are not considered NA values. None is considered a NA
        value.

        >>> idx = pd.Index(['black', '', 'red', None])
        >>> idx
        Index(['black', '', 'red', None], dtype='object')
        >>> idx.notna()
        array([ True,  True,  True, False])
        """
        ...
    notnull = ...
    def fillna(self, value=..., downcast=...):  # -> Index | Self@Index:
        """
        Fill NA/NaN values with the specified value.

        Parameters
        ----------
        value : scalar
            Scalar value to use to fill holes (e.g. 0).
            This value cannot be a list-likes.
        downcast : dict, default is None
            A dict of item->dtype of what to downcast if possible,
            or the string 'infer' which will try to downcast to an appropriate
            equal type (e.g. float64 to int64 if possible).

        Returns
        -------
        Index

        See Also
        --------
        DataFrame.fillna : Fill NaN values of a DataFrame.
        Series.fillna : Fill NaN Values of a Series.
        """
        ...
    def dropna(self: _IndexT, how: str_t = ...) -> _IndexT:
        """
        Return Index without NA/NaN values.

        Parameters
        ----------
        how : {'any', 'all'}, default 'any'
            If the Index is a MultiIndex, drop the value when any or all levels
            are NaN.

        Returns
        -------
        Index
        """
        ...
    def unique(self: _IndexT, level: Hashable | None = ...) -> _IndexT:
        """
        Return unique values in the index.

        Unique values are returned in order of appearance, this does NOT sort.

        Parameters
        ----------
        level : int or hashable, optional
            Only return values from specified level (for MultiIndex).
            If int, gets the level by integer position, else by level name.

        Returns
        -------
        Index

        See Also
        --------
        unique : Numpy array of unique values in that column.
        Series.unique : Return unique values of Series object.
        """
        ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self"])
    def drop_duplicates(self: _IndexT, keep: str_t | bool = ...) -> _IndexT:
        """
        Return Index with duplicate values removed.

        Parameters
        ----------
        keep : {'first', 'last', ``False``}, default 'first'
            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - ``False`` : Drop all duplicates.

        Returns
        -------
        deduplicated : Index

        See Also
        --------
        Series.drop_duplicates : Equivalent method on Series.
        DataFrame.drop_duplicates : Equivalent method on DataFrame.
        Index.duplicated : Related method on Index, indicating duplicate
            Index values.

        Examples
        --------
        Generate an pandas.Index with duplicate values.

        >>> idx = pd.Index(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'])

        The `keep` parameter controls  which duplicate values are removed.
        The value 'first' keeps the first occurrence for each
        set of duplicated entries. The default value of keep is 'first'.

        >>> idx.drop_duplicates(keep='first')
        Index(['lama', 'cow', 'beetle', 'hippo'], dtype='object')

        The value 'last' keeps the last occurrence for each set of duplicated
        entries.

        >>> idx.drop_duplicates(keep='last')
        Index(['cow', 'beetle', 'lama', 'hippo'], dtype='object')

        The value ``False`` discards all sets of duplicated entries.

        >>> idx.drop_duplicates(keep=False)
        Index(['cow', 'beetle', 'hippo'], dtype='object')
        """
        ...
    def duplicated(self, keep: Literal["first", "last", False] = ...) -> np.ndarray:
        """
        Indicate duplicate index values.

        Duplicated values are indicated as ``True`` values in the resulting
        array. Either all duplicates, all except the first, or all except the
        last occurrence of duplicates can be indicated.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            The value or values in a set of duplicates to mark as missing.

            - 'first' : Mark duplicates as ``True`` except for the first
              occurrence.
            - 'last' : Mark duplicates as ``True`` except for the last
              occurrence.
            - ``False`` : Mark all duplicates as ``True``.

        Returns
        -------
        np.ndarray[bool]

        See Also
        --------
        Series.duplicated : Equivalent method on pandas.Series.
        DataFrame.duplicated : Equivalent method on pandas.DataFrame.
        Index.drop_duplicates : Remove duplicate values from Index.

        Examples
        --------
        By default, for each set of duplicated values, the first occurrence is
        set to False and all others to True:

        >>> idx = pd.Index(['lama', 'cow', 'lama', 'beetle', 'lama'])
        >>> idx.duplicated()
        array([False, False,  True, False,  True])

        which is equivalent to

        >>> idx.duplicated(keep='first')
        array([False, False,  True, False,  True])

        By using 'last', the last occurrence of each set of duplicated values
        is set on False and all others on True:

        >>> idx.duplicated(keep='last')
        array([ True, False,  True, False, False])

        By setting keep on ``False``, all duplicates are True:

        >>> idx.duplicated(keep=False)
        array([ True, False,  True, False,  True])
        """
        ...
    def __iadd__(self, other): ...
    @final
    def __and__(self, other): ...
    @final
    def __or__(self, other): ...
    @final
    def __xor__(self, other): ...
    @final
    def __nonzero__(self): ...
    __bool__ = ...
    @final
    def union(self, other, sort=...):
        """
        Form the union of two Index objects.

        If the Index objects are incompatible, both Index objects will be
        cast to dtype('object') first.

            .. versionchanged:: 0.25.0

        Parameters
        ----------
        other : Index or array-like
        sort : bool or None, default None
            Whether to sort the resulting Index.

            * None : Sort the result, except when

              1. `self` and `other` are equal.
              2. `self` or `other` has length 0.
              3. Some values in `self` or `other` cannot be compared.
                 A RuntimeWarning is issued in this case.

            * False : do not sort the result.

        Returns
        -------
        union : Index

        Examples
        --------
        Union matching dtypes

        >>> idx1 = pd.Index([1, 2, 3, 4])
        >>> idx2 = pd.Index([3, 4, 5, 6])
        >>> idx1.union(idx2)
        Int64Index([1, 2, 3, 4, 5, 6], dtype='int64')

        Union mismatched dtypes

        >>> idx1 = pd.Index(['a', 'b', 'c', 'd'])
        >>> idx2 = pd.Index([1, 2, 3, 4])
        >>> idx1.union(idx2)
        Index(['a', 'b', 'c', 'd', 1, 2, 3, 4], dtype='object')

        MultiIndex case

        >>> idx1 = pd.MultiIndex.from_arrays(
        ...     [[1, 1, 2, 2], ["Red", "Blue", "Red", "Blue"]]
        ... )
        >>> idx1
        MultiIndex([(1,  'Red'),
            (1, 'Blue'),
            (2,  'Red'),
            (2, 'Blue')],
           )
        >>> idx2 = pd.MultiIndex.from_arrays(
        ...     [[3, 3, 2, 2], ["Red", "Green", "Red", "Green"]]
        ... )
        >>> idx2
        MultiIndex([(3,   'Red'),
            (3, 'Green'),
            (2,   'Red'),
            (2, 'Green')],
           )
        >>> idx1.union(idx2)
        MultiIndex([(1,  'Blue'),
            (1,   'Red'),
            (2,  'Blue'),
            (2, 'Green'),
            (2,   'Red'),
            (3, 'Green'),
            (3,   'Red')],
           )
        >>> idx1.union(idx2, sort=False)
        MultiIndex([(1,   'Red'),
            (1,  'Blue'),
            (2,   'Red'),
            (2,  'Blue'),
            (3,   'Red'),
            (3, 'Green'),
            (2, 'Green')],
           )
        """
        ...
    @final
    def intersection(
        self, other, sort=...
    ):  # -> ABCMultiIndex | Self@Index | Index | None:
        """
        Form the intersection of two Index objects.

        This returns a new Index with elements common to the index and `other`.

        Parameters
        ----------
        other : Index or array-like
        sort : False or None, default False
            Whether to sort the resulting index.

            * False : do not sort the result.
            * None : sort the result, except when `self` and `other` are equal
              or when the values cannot be compared.

        Returns
        -------
        intersection : Index

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3, 4])
        >>> idx2 = pd.Index([3, 4, 5, 6])
        >>> idx1.intersection(idx2)
        Int64Index([3, 4], dtype='int64')
        """
        ...
    @final
    def difference(
        self, other, sort=...
    ):  # -> Any | ABCMultiIndex | Index | Self@Index | None:
        """
        Return a new Index with elements of index not in `other`.

        This is the set difference of two Index objects.

        Parameters
        ----------
        other : Index or array-like
        sort : False or None, default None
            Whether to sort the resulting index. By default, the
            values are attempted to be sorted, but any TypeError from
            incomparable elements is caught by pandas.

            * None : Attempt to sort the result, but catch any TypeErrors
              from comparing incomparable elements.
            * False : Do not sort the result.

        Returns
        -------
        difference : Index

        Examples
        --------
        >>> idx1 = pd.Index([2, 1, 3, 4])
        >>> idx2 = pd.Index([3, 4, 5, 6])
        >>> idx1.difference(idx2)
        Int64Index([1, 2], dtype='int64')
        >>> idx1.difference(idx2, sort=False)
        Int64Index([2, 1], dtype='int64')
        """
        ...
    def symmetric_difference(
        self, other, result_name=..., sort=...
    ):  # -> ABCMultiIndex | Index | None:
        """
        Compute the symmetric difference of two Index objects.

        Parameters
        ----------
        other : Index or array-like
        result_name : str
        sort : False or None, default None
            Whether to sort the resulting index. By default, the
            values are attempted to be sorted, but any TypeError from
            incomparable elements is caught by pandas.

            * None : Attempt to sort the result, but catch any TypeErrors
              from comparing incomparable elements.
            * False : Do not sort the result.

        Returns
        -------
        symmetric_difference : Index

        Notes
        -----
        ``symmetric_difference`` contains elements that appear in either
        ``idx1`` or ``idx2`` but not both. Equivalent to the Index created by
        ``idx1.difference(idx2) | idx2.difference(idx1)`` with duplicates
        dropped.

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3, 4])
        >>> idx2 = pd.Index([2, 3, 4, 5])
        >>> idx1.symmetric_difference(idx2)
        Int64Index([1, 5], dtype='int64')
        """
        ...
    def get_loc(self, key, method=..., tolerance=...):  # -> Any:
        """
        Get integer location, slice or boolean mask for requested label.

        Parameters
        ----------
        key : label
        method : {None, 'pad'/'ffill', 'backfill'/'bfill', 'nearest'}, optional
            * default: exact matches only.
            * pad / ffill: find the PREVIOUS index value if no exact match.
            * backfill / bfill: use NEXT index value if no exact match
            * nearest: use the NEAREST index value if no exact match. Tied
              distances are broken by preferring the larger index value.
        tolerance : int or float, optional
            Maximum distance from index value for inexact matches. The value of
            the index at the matching location must satisfy the equation
            ``abs(index[loc] - key) <= tolerance``.

        Returns
        -------
        loc : int if unique index, slice if monotonic index, else mask

        Examples
        --------
        >>> unique_index = pd.Index(list('abc'))
        >>> unique_index.get_loc('b')
        1

        >>> monotonic_index = pd.Index(list('abbc'))
        >>> monotonic_index.get_loc('b')
        slice(1, 3, None)

        >>> non_monotonic_index = pd.Index(list('abcb'))
        >>> non_monotonic_index.get_loc('b')
        array([False,  True, False,  True])
        """
        ...
    @Appender(_index_shared_docs["get_indexer"] % _index_doc_kwargs)
    @final
    def get_indexer(
        self, target, method: str_t | None = ..., limit: int | None = ..., tolerance=...
    ) -> np.ndarray: ...
    def reindex(
        self, target, method=..., level=..., limit=..., tolerance=...
    ) -> tuple[Index, np.ndarray | None]:
        """
        Create index with target's values.

        Parameters
        ----------
        target : an iterable

        Returns
        -------
        new_index : pd.Index
            Resulting index.
        indexer : np.ndarray[np.intp] or None
            Indices of output values in original index.
        """
        ...
    @_maybe_return_indexers
    def join(
        self,
        other,
        how: str_t = ...,
        level=...,
        return_indexers: bool = ...,
        sort: bool = ...,
    ):  # -> tuple[MultiIndex, Unknown, Unknown] | tuple[MultiIndex, ndarray | None, ndarray | None] | tuple[Self@Index, None, ndarray] | tuple[Index, ndarray, None] | tuple[Unknown, Unknown, Unknown] | tuple[Index, ndarray, ndarray] | tuple[Index | Self@Index, None, None] | tuple[Self@Index | Index | Unbound, ndarray | None, ndarray | None] | tuple[tuple[Self@Index | MultiIndex, Unknown | ndarray | Any] | Self@Index | MultiIndex | Unknown | Index | Unbound | None, ndarray | None, ndarray | None]:
        """
        Compute join_index and indexers to conform data
        structures to the new index.

        Parameters
        ----------
        other : Index
        how : {'left', 'right', 'inner', 'outer'}
        level : int or level name, default None
        return_indexers : bool, default False
        sort : bool, default False
            Sort the join keys lexicographically in the result Index. If False,
            the order of the join keys depends on the join type (how keyword).

        Returns
        -------
        join_index, (left_indexer, right_indexer)
        """
        ...
    @property
    def values(self) -> ArrayLike:
        """
        Return an array representing the data in the Index.

        .. warning::

           We recommend using :attr:`Index.array` or
           :meth:`Index.to_numpy`, depending on whether you need
           a reference to the underlying data or a NumPy array.

        Returns
        -------
        array: numpy.ndarray or ExtensionArray

        See Also
        --------
        Index.array : Reference to the underlying data.
        Index.to_numpy : A NumPy array representing the underlying data.
        """
        ...
    @cache_readonly
    @doc(IndexOpsMixin.array)
    def array(self) -> ExtensionArray: ...
    @doc(IndexOpsMixin._memory_usage)
    def memory_usage(self, deep: bool = ...) -> int: ...
    @final
    def where(self, cond, other=...) -> Index:
        """
        Replace values where the condition is False.

        The replacement is taken from other.

        Parameters
        ----------
        cond : bool array-like with the same length as self
            Condition to select the values on.
        other : scalar, or array-like, default None
            Replacement if the condition is False.

        Returns
        -------
        pandas.Index
            A copy of self with values replaced from other
            where the condition is False.

        See Also
        --------
        Series.where : Same method for Series.
        DataFrame.where : Same method for DataFrame.

        Examples
        --------
        >>> idx = pd.Index(['car', 'bike', 'train', 'tractor'])
        >>> idx
        Index(['car', 'bike', 'train', 'tractor'], dtype='object')
        >>> idx.where(idx.isin(['car', 'train']), 'other')
        Index(['car', 'other', 'train', 'other'], dtype='object')
        """
        ...
    def is_type_compatible(self, kind: str_t) -> bool:
        """
        Whether the index type is compatible with the provided type.
        """
        ...
    def __contains__(self, key: Any) -> bool:
        """
        Return a boolean indicating whether the provided key is in the index.

        Parameters
        ----------
        key : label
            The key to check if it is present in the index.

        Returns
        -------
        bool
            Whether the key search is in the index.

        Raises
        ------
        TypeError
            If the key is not hashable.

        See Also
        --------
        Index.isin : Returns an ndarray of boolean dtype indicating whether the
            list-like key is in the index.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx
        Int64Index([1, 2, 3, 4], dtype='int64')

        >>> 2 in idx
        True
        >>> 6 in idx
        False
        """
        ...
    __hash__: None
    @final
    def __setitem__(self, key, value): ...
    def __getitem__(self, key):  # -> ExtensionArray | Any | Self@Index:
        """
        Override numpy.ndarray's __getitem__ method to work as desired.

        This function adds lists and Series as valid boolean indexers
        (ndarrays only supports ndarray with dtype=bool).

        If resulting ndim != 1, plain ndarray is returned instead of
        corresponding `Index` subclass.

        """
        ...
    def append(self, other: Index | Sequence[Index]) -> Index:
        """
        Append a collection of Index options together.

        Parameters
        ----------
        other : Index or list/tuple of indices

        Returns
        -------
        Index
        """
        ...
    def putmask(self, mask, value) -> Index:
        """
        Return a new Index of the values set with the mask.

        Returns
        -------
        Index

        See Also
        --------
        numpy.ndarray.putmask : Changes elements of an array
            based on conditional and input values.
        """
        ...
    def equals(self, other: Any) -> bool:
        """
        Determine if two Index object are equal.

        The things that are being compared are:

        * The elements inside the Index object.
        * The order of the elements inside the Index object.

        Parameters
        ----------
        other : Any
            The other object to compare against.

        Returns
        -------
        bool
            True if "other" is an Index and it has the same elements and order
            as the calling index; False otherwise.

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3])
        >>> idx1
        Int64Index([1, 2, 3], dtype='int64')
        >>> idx1.equals(pd.Index([1, 2, 3]))
        True

        The elements inside are compared

        >>> idx2 = pd.Index(["1", "2", "3"])
        >>> idx2
        Index(['1', '2', '3'], dtype='object')

        >>> idx1.equals(idx2)
        False

        The order is compared

        >>> ascending_idx = pd.Index([1, 2, 3])
        >>> ascending_idx
        Int64Index([1, 2, 3], dtype='int64')
        >>> descending_idx = pd.Index([3, 2, 1])
        >>> descending_idx
        Int64Index([3, 2, 1], dtype='int64')
        >>> ascending_idx.equals(descending_idx)
        False

        The dtype is *not* compared

        >>> int64_idx = pd.Int64Index([1, 2, 3])
        >>> int64_idx
        Int64Index([1, 2, 3], dtype='int64')
        >>> uint64_idx = pd.UInt64Index([1, 2, 3])
        >>> uint64_idx
        UInt64Index([1, 2, 3], dtype='uint64')
        >>> int64_idx.equals(uint64_idx)
        True
        """
        ...
    @final
    def identical(self, other) -> bool:
        """
        Similar to equals, but checks that object attributes and types are also equal.

        Returns
        -------
        bool
            If two Index objects have equal elements and same type True,
            otherwise False.
        """
        ...
    @final
    def asof(self, label):  # -> ExtensionArray | Any | Index | float | NaTType:
        """
        Return the label from the index, or, if not present, the previous one.

        Assuming that the index is sorted, return the passed index label if it
        is in the index, or return the previous index label if the passed one
        is not in the index.

        Parameters
        ----------
        label : object
            The label up to which the method returns the latest index label.

        Returns
        -------
        object
            The passed label if it is in the index. The previous label if the
            passed label is not in the sorted index or `NaN` if there is no
            such label.

        See Also
        --------
        Series.asof : Return the latest value in a Series up to the
            passed index.
        merge_asof : Perform an asof merge (similar to left join but it
            matches on nearest key rather than equal key).
        Index.get_loc : An `asof` is a thin wrapper around `get_loc`
            with method='pad'.

        Examples
        --------
        `Index.asof` returns the latest index label up to the passed label.

        >>> idx = pd.Index(['2013-12-31', '2014-01-02', '2014-01-03'])
        >>> idx.asof('2014-01-01')
        '2013-12-31'

        If the label is in the index, the method returns the passed label.

        >>> idx.asof('2014-01-02')
        '2014-01-02'

        If all of the labels in the index are later than the passed label,
        NaN is returned.

        >>> idx.asof('1999-01-02')
        nan

        If the index is not sorted, an error is raised.

        >>> idx_not_sorted = pd.Index(['2013-12-31', '2015-01-02',
        ...                            '2014-01-03'])
        >>> idx_not_sorted.asof('2013-12-31')
        Traceback (most recent call last):
        ValueError: index must be monotonic increasing or decreasing
        """
        ...
    def asof_locs(self, where: Index, mask: np.ndarray) -> np.ndarray:
        """
        Return the locations (indices) of labels in the index.

        As in the `asof` function, if the label (a particular entry in
        `where`) is not in the index, the latest index label up to the
        passed label is chosen and its index returned.

        If all of the labels in the index are later than a label in `where`,
        -1 is returned.

        `mask` is used to ignore NA values in the index during calculation.

        Parameters
        ----------
        where : Index
            An Index consisting of an array of timestamps.
        mask : np.ndarray[bool]
            Array of booleans denoting where values in the original
            data are not NA.

        Returns
        -------
        np.ndarray[np.intp]
            An array of locations (indices) of the labels from the Index
            which correspond to the return values of the `asof` function
            for every element in `where`.
        """
        ...
    @final
    def sort_values(
        self,
        return_indexer: bool = ...,
        ascending: bool = ...,
        na_position: str_t = ...,
        key: Callable | None = ...,
    ):  # -> tuple[Self@Index | MultiIndex, Unknown | ndarray | Any] | Self@Index | MultiIndex:
        """
        Return a sorted copy of the index.

        Return a sorted copy of the index, and optionally return the indices
        that sorted the index itself.

        Parameters
        ----------
        return_indexer : bool, default False
            Should the indices that would sort the index be returned.
        ascending : bool, default True
            Should the index values be sorted in an ascending order.
        na_position : {'first' or 'last'}, default 'last'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
            the end.

            .. versionadded:: 1.2.0

        key : callable, optional
            If not None, apply the key function to the index values
            before sorting. This is similar to the `key` argument in the
            builtin :meth:`sorted` function, with the notable difference that
            this `key` function should be *vectorized*. It should expect an
            ``Index`` and return an ``Index`` of the same shape.

            .. versionadded:: 1.1.0

        Returns
        -------
        sorted_index : pandas.Index
            Sorted copy of the index.
        indexer : numpy.ndarray, optional
            The indices that the index itself was sorted by.

        See Also
        --------
        Series.sort_values : Sort values of a Series.
        DataFrame.sort_values : Sort values in a DataFrame.

        Examples
        --------
        >>> idx = pd.Index([10, 100, 1, 1000])
        >>> idx
        Int64Index([10, 100, 1, 1000], dtype='int64')

        Sort values in ascending order (default behavior).

        >>> idx.sort_values()
        Int64Index([1, 10, 100, 1000], dtype='int64')

        Sort values in descending order, and also get the indices `idx` was
        sorted by.

        >>> idx.sort_values(ascending=False, return_indexer=True)
        (Int64Index([1000, 100, 10, 1], dtype='int64'), array([3, 1, 0, 2]))
        """
        ...
    @final
    def sort(self, *args, **kwargs):  # -> NoReturn:
        """
        Use sort_values instead.
        """
        ...
    def shift(self, periods=..., freq=...):
        """
        Shift index by desired number of time frequency increments.

        This method is for shifting the values of datetime-like indexes
        by a specified time increment a given number of times.

        Parameters
        ----------
        periods : int, default 1
            Number of periods (or increments) to shift by,
            can be positive or negative.
        freq : pandas.DateOffset, pandas.Timedelta or str, optional
            Frequency increment to shift by.
            If None, the index is shifted by its own `freq` attribute.
            Offset aliases are valid strings, e.g., 'D', 'W', 'M' etc.

        Returns
        -------
        pandas.Index
            Shifted index.

        See Also
        --------
        Series.shift : Shift values of Series.

        Notes
        -----
        This method is only implemented for datetime-like index classes,
        i.e., DatetimeIndex, PeriodIndex and TimedeltaIndex.

        Examples
        --------
        Put the first 5 month starts of 2011 into an index.

        >>> month_starts = pd.date_range('1/1/2011', periods=5, freq='MS')
        >>> month_starts
        DatetimeIndex(['2011-01-01', '2011-02-01', '2011-03-01', '2011-04-01',
                       '2011-05-01'],
                      dtype='datetime64[ns]', freq='MS')

        Shift the index by 10 days.

        >>> month_starts.shift(10, freq='D')
        DatetimeIndex(['2011-01-11', '2011-02-11', '2011-03-11', '2011-04-11',
                       '2011-05-11'],
                      dtype='datetime64[ns]', freq=None)

        The default value of `freq` is the `freq` attribute of the index,
        which is 'MS' (month start) in this example.

        >>> month_starts.shift(10)
        DatetimeIndex(['2011-11-01', '2011-12-01', '2012-01-01', '2012-02-01',
                       '2012-03-01'],
                      dtype='datetime64[ns]', freq='MS')
        """
        ...
    def argsort(self, *args, **kwargs) -> np.ndarray:
        """
        Return the integer indices that would sort the index.

        Parameters
        ----------
        *args
            Passed to `numpy.ndarray.argsort`.
        **kwargs
            Passed to `numpy.ndarray.argsort`.

        Returns
        -------
        np.ndarray[np.intp]
            Integer indices that would sort the index if used as
            an indexer.

        See Also
        --------
        numpy.argsort : Similar method for NumPy arrays.
        Index.sort_values : Return sorted copy of Index.

        Examples
        --------
        >>> idx = pd.Index(['b', 'a', 'd', 'c'])
        >>> idx
        Index(['b', 'a', 'd', 'c'], dtype='object')

        >>> order = idx.argsort()
        >>> order
        array([1, 0, 3, 2])

        >>> idx[order]
        Index(['a', 'b', 'c', 'd'], dtype='object')
        """
        ...
    @final
    def get_value(self, series: Series, key):  # -> Any | ExtensionArray:
        """
        Fast lookup of value from 1-dimensional ndarray.

        Only use this if you know what you're doing.

        Returns
        -------
        scalar or Series
        """
        ...
    @final
    def set_value(self, arr, key, value):  # -> None:
        """
        Fast lookup of value from 1-dimensional ndarray.

        .. deprecated:: 1.0

        Notes
        -----
        Only use this if you know what you're doing.
        """
        ...
    @Appender(_index_shared_docs["get_indexer_non_unique"] % _index_doc_kwargs)
    def get_indexer_non_unique(self, target) -> tuple[np.ndarray, np.ndarray]: ...
    @final
    def get_indexer_for(self, target, **kwargs) -> np.ndarray:
        """
        Guaranteed return of an indexer even when non-unique.

        This dispatches to get_indexer or get_indexer_non_unique
        as appropriate.

        Returns
        -------
        np.ndarray[np.intp]
            List of indices.
        """
        ...
    _requires_unique_msg = ...
    @final
    def groupby(self, values) -> PrettyDict[Hashable, np.ndarray]:
        """
        Group the index labels by a given array of values.

        Parameters
        ----------
        values : array
            Values used to determine the groups.

        Returns
        -------
        dict
            {group name -> group labels}
        """
        ...
    def map(self, mapper, na_action=...):  # -> MultiIndex | Index:
        """
        Map values using input correspondence (a dict, Series, or function).

        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}
            If 'ignore', propagate NA values, without passing them to the
            mapping correspondence.

        Returns
        -------
        applied : Union[Index, MultiIndex], inferred
            The output of the mapping function applied to the index.
            If the function returns a tuple with more than one element
            a MultiIndex will be returned.
        """
        ...
    def isin(self, values, level=...) -> np.ndarray:
        """
        Return a boolean array where the index values are in `values`.

        Compute boolean array of whether each index value is found in the
        passed set of values. The length of the returned boolean array matches
        the length of the index.

        Parameters
        ----------
        values : set or list-like
            Sought values.
        level : str or int, optional
            Name or position of the index level to use (if the index is a
            `MultiIndex`).

        Returns
        -------
        np.ndarray[bool]
            NumPy array of boolean values.

        See Also
        --------
        Series.isin : Same for Series.
        DataFrame.isin : Same method for DataFrames.

        Notes
        -----
        In the case of `MultiIndex` you must either specify `values` as a
        list-like object containing tuples that are the same length as the
        number of levels, or specify `level`. Otherwise it will raise a
        ``ValueError``.

        If `level` is specified:

        - if it is the name of one *and only one* index level, use that level;
        - otherwise it should be a number indicating level position.

        Examples
        --------
        >>> idx = pd.Index([1,2,3])
        >>> idx
        Int64Index([1, 2, 3], dtype='int64')

        Check whether each index value in a list of values.

        >>> idx.isin([1, 4])
        array([ True, False, False])

        >>> midx = pd.MultiIndex.from_arrays([[1,2,3],
        ...                                  ['red', 'blue', 'green']],
        ...                                  names=('number', 'color'))
        >>> midx
        MultiIndex([(1,   'red'),
                    (2,  'blue'),
                    (3, 'green')],
                   names=['number', 'color'])

        Check whether the strings in the 'color' level of the MultiIndex
        are in a list of colors.

        >>> midx.isin(['red', 'orange', 'yellow'], level='color')
        array([ True, False, False])

        To check across the levels of a MultiIndex, pass a list of tuples:

        >>> midx.isin([(1, 'red'), (3, 'red')])
        array([ True, False, False])

        For a DatetimeIndex, string values in `values` are converted to
        Timestamps.

        >>> dates = ['2000-03-11', '2000-03-12', '2000-03-13']
        >>> dti = pd.to_datetime(dates)
        >>> dti
        DatetimeIndex(['2000-03-11', '2000-03-12', '2000-03-13'],
        dtype='datetime64[ns]', freq=None)

        >>> dti.isin(['2000-03-11'])
        array([ True, False, False])
        """
        ...
    def slice_indexer(
        self,
        start: Hashable | None = ...,
        end: Hashable | None = ...,
        step: int | None = ...,
        kind: str_t | None = ...,
    ) -> slice:
        """
        Compute the slice indexer for input labels and step.

        Index needs to be ordered and unique.

        Parameters
        ----------
        start : label, default None
            If None, defaults to the beginning.
        end : label, default None
            If None, defaults to the end.
        step : int, default None
        kind : str, default None

        Returns
        -------
        indexer : slice

        Raises
        ------
        KeyError : If key does not exist, or key is not unique and index is
            not ordered.

        Notes
        -----
        This function assumes that the data is sorted, so use at your own peril

        Examples
        --------
        This is a method on all index types. For example you can do:

        >>> idx = pd.Index(list('abcd'))
        >>> idx.slice_indexer(start='b', end='c')
        slice(1, 3, None)

        >>> idx = pd.MultiIndex.from_arrays([list('abcd'), list('efgh')])
        >>> idx.slice_indexer(start='b', end=('c', 'g'))
        slice(1, 3, None)
        """
        ...
    def get_slice_bound(self, label, side: str_t, kind=...) -> int:
        """
        Calculate slice bound that corresponds to given label.

        Returns leftmost (one-past-the-rightmost if ``side=='right'``) position
        of given label.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}
        kind : {'loc', 'getitem'} or None

        Returns
        -------
        int
            Index of label.
        """
        ...
    def slice_locs(self, start=..., end=..., step=..., kind=...):
        """
        Compute slice locations for input labels.

        Parameters
        ----------
        start : label, default None
            If None, defaults to the beginning.
        end : label, default None
            If None, defaults to the end.
        step : int, defaults None
            If None, defaults to 1.
        kind : {'loc', 'getitem'} or None

        Returns
        -------
        start, end : int

        See Also
        --------
        Index.get_loc : Get location for a single label.

        Notes
        -----
        This method only works if the index is monotonic or unique.

        Examples
        --------
        >>> idx = pd.Index(list('abcd'))
        >>> idx.slice_locs(start='b', end='c')
        (1, 3)
        """
        ...
    def delete(self: _IndexT, loc) -> _IndexT:
        """
        Make new Index with passed location(-s) deleted.

        Parameters
        ----------
        loc : int or list of int
            Location of item(-s) which will be deleted.
            Use a list of locations to delete more than one value at the same time.

        Returns
        -------
        Index
            Will be same type as self, except for RangeIndex.

        See Also
        --------
        numpy.delete : Delete any rows and column from NumPy array (ndarray).

        Examples
        --------
        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx.delete(1)
        Index(['a', 'c'], dtype='object')

        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx.delete([0, 2])
        Index(['b'], dtype='object')
        """
        ...
    def insert(self, loc: int, item) -> Index:
        """
        Make new Index inserting new item at location.

        Follows Python list.append semantics for negative values.

        Parameters
        ----------
        loc : int
        item : object

        Returns
        -------
        new_index : Index
        """
        ...
    def drop(self, labels, errors: str_t = ...) -> Index:
        """
        Make new Index with passed list of labels deleted.

        Parameters
        ----------
        labels : array-like
        errors : {'ignore', 'raise'}, default 'raise'
            If 'ignore', suppress error and existing labels are dropped.

        Returns
        -------
        dropped : Index
            Will be same type as self, except for RangeIndex.

        Raises
        ------
        KeyError
            If not all of the labels are found in the selected axis
        """
        ...
    def __abs__(self): ...
    def __neg__(self): ...
    def __pos__(self): ...
    def __inv__(self): ...
    def any(self, *args, **kwargs):
        """
        Return whether any element is Truthy.

        Parameters
        ----------
        *args
            Required for compatibility with numpy.
        **kwargs
            Required for compatibility with numpy.

        Returns
        -------
        any : bool or array-like (if axis is specified)
            A single element array-like may be converted to bool.

        See Also
        --------
        Index.all : Return whether all elements are True.
        Series.all : Return whether all elements are True.

        Notes
        -----
        Not a Number (NaN), positive infinity and negative infinity
        evaluate to True because these are not equal to zero.

        Examples
        --------
        >>> index = pd.Index([0, 1, 2])
        >>> index.any()
        True

        >>> index = pd.Index([0, 0, 0])
        >>> index.any()
        False
        """
        ...
    def all(self, *args, **kwargs):
        """
        Return whether all elements are Truthy.

        Parameters
        ----------
        *args
            Required for compatibility with numpy.
        **kwargs
            Required for compatibility with numpy.

        Returns
        -------
        all : bool or array-like (if axis is specified)
            A single element array-like may be converted to bool.

        See Also
        --------
        Index.any : Return whether any element in an Index is True.
        Series.any : Return whether any element in a Series is True.
        Series.all : Return whether all elements in a Series are True.

        Notes
        -----
        Not a Number (NaN), positive infinity and negative infinity
        evaluate to True because these are not equal to zero.

        Examples
        --------
        **all**

        True, because nonzero integers are considered True.

        >>> pd.Index([1, 2, 3]).all()
        True

        False, because ``0`` is considered False.

        >>> pd.Index([0, 1, 2]).all()
        False

        **any**

        True, because ``1`` is considered True.

        >>> pd.Index([0, 0, 1]).any()
        True

        False, because ``0`` is considered False.

        >>> pd.Index([0, 0, 0]).any()
        False
        """
        ...
    @final
    @property
    def shape(self) -> Shape:
        """
        Return a tuple of the shape of the underlying data.
        """
        ...

def ensure_index_from_sequences(sequences, names=...):  # -> Index | MultiIndex:
    """
    Construct an index from sequences of data.

    A single sequence returns an Index. Many sequences returns a
    MultiIndex.

    Parameters
    ----------
    sequences : sequence of sequences
    names : sequence of str

    Returns
    -------
    index : Index or MultiIndex

    Examples
    --------
    >>> ensure_index_from_sequences([[1, 2, 3]], names=["name"])
    Int64Index([1, 2, 3], dtype='int64', name='name')

    >>> ensure_index_from_sequences([["a", "a"], ["a", "b"]], names=["L1", "L2"])
    MultiIndex([('a', 'a'),
                ('a', 'b')],
               names=['L1', 'L2'])

    See Also
    --------
    ensure_index
    """
    ...

def ensure_index(index_like: AnyArrayLike | Sequence, copy: bool = ...) -> Index:
    """
    Ensure that we have an index from some index-like object.

    Parameters
    ----------
    index_like : sequence
        An Index or other sequence
    copy : bool, default False

    Returns
    -------
    index : Index or MultiIndex

    See Also
    --------
    ensure_index_from_sequences

    Examples
    --------
    >>> ensure_index(['a', 'b'])
    Index(['a', 'b'], dtype='object')

    >>> ensure_index([('a', 'a'),  ('b', 'c')])
    Index([('a', 'a'), ('b', 'c')], dtype='object')

    >>> ensure_index([['a', 'a'], ['b', 'c']])
    MultiIndex([('a', 'b'),
            ('a', 'c')],
           )
    """
    ...

def ensure_has_len(seq):  # -> list[Unknown]:
    """
    If seq is an iterator, put its values into a list.
    """
    ...

def trim_front(strings: list[str]) -> list[str]:
    """
    Trims zeros and decimal points.

    Examples
    --------
    >>> trim_front([" a", " b"])
    ['a', 'b']

    >>> trim_front([" a", " "])
    ['a', '']
    """
    ...

def default_index(n: int) -> RangeIndex: ...
def maybe_extract_name(name, obj, cls) -> Hashable:
    """
    If no name is passed, then extract it from data, validating hashability.
    """
    ...

def get_unanimous_names(*indexes: Index) -> tuple[Hashable, ...]:
    """
    Return common name if all indices agree, otherwise None (level-by-level).

    Parameters
    ----------
    indexes : list of Index objects

    Returns
    -------
    list
        A list representing the unanimous 'names' found.
    """
    ...

def unpack_nested_dtype(other: _IndexT) -> _IndexT:
    """
    When checking if our dtype is comparable with another, we need
    to unpack CategoricalDtype to look at its categories.dtype.

    Parameters
    ----------
    other : Index

    Returns
    -------
    Index
    """
    ...
