from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable, Sequence

import numpy as np
from pandas import DataFrame, Series
from pandas._libs import index as libindex
from pandas._libs.hashtable import duplicated
from pandas._typing import Shape
from pandas.core.indexes.base import Index, _index_shared_docs
from pandas.core.indexes.frozen import FrozenList
from pandas.util._decorators import (
    Appender,
    cache_readonly,
    deprecate_nonkeyword_arguments,
    doc,
)

if TYPE_CHECKING: ...
_index_doc_kwargs = ...

class MultiIndexUIntEngine(libindex.BaseMultiIndexCodesEngine, libindex.UInt64Engine):
    """
    This class manages a MultiIndex by mapping label combinations to positive
    integers.
    """

    _base = libindex.UInt64Engine

class MultiIndexPyIntEngine(libindex.BaseMultiIndexCodesEngine, libindex.ObjectEngine):
    """
    This class manages those (extreme) cases in which the number of possible
    label combinations overflows the 64 bits integers, and uses an ObjectEngine
    containing Python integers.
    """

    _base = libindex.ObjectEngine

def names_compat(
    meth,
):  # -> (self_or_cls: Unknown, *args: Unknown, **kwargs: Unknown) -> Unknown:
    """
    A decorator to allow either `name` or `names` keyword but not both.

    This makes it easier to share code with base class.
    """
    ...

class MultiIndex(Index):
    """
    A multi-level, or hierarchical, index object for pandas objects.

    Parameters
    ----------
    levels : sequence of arrays
        The unique labels for each level.
    codes : sequence of arrays
        Integers for each level designating which label at each location.
    sortorder : optional int
        Level of sortedness (must be lexicographically sorted by that
        level).
    names : optional sequence of objects
        Names for each of the index levels. (name is accepted for compat).
    copy : bool, default False
        Copy the meta-data.
    verify_integrity : bool, default True
        Check that the levels/codes are consistent and valid.

    Attributes
    ----------
    names
    levels
    codes
    nlevels
    levshape

    Methods
    -------
    from_arrays
    from_tuples
    from_product
    from_frame
    set_levels
    set_codes
    to_frame
    to_flat_index
    sortlevel
    droplevel
    swaplevel
    reorder_levels
    remove_unused_levels
    get_locs

    See Also
    --------
    MultiIndex.from_arrays  : Convert list of arrays to MultiIndex.
    MultiIndex.from_product : Create a MultiIndex from the cartesian product
                              of iterables.
    MultiIndex.from_tuples  : Convert list of tuples to a MultiIndex.
    MultiIndex.from_frame   : Make a MultiIndex from a DataFrame.
    Index : The base pandas Index type.

    Notes
    -----
    See the `user guide
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html>`__
    for more.

    Examples
    --------
    A new ``MultiIndex`` is typically constructed using one of the helper
    methods :meth:`MultiIndex.from_arrays`, :meth:`MultiIndex.from_product`
    and :meth:`MultiIndex.from_tuples`. For example (using ``.from_arrays``):

    >>> arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
    >>> pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))
    MultiIndex([(1,  'red'),
                (1, 'blue'),
                (2,  'red'),
                (2, 'blue')],
               names=['number', 'color'])

    See further examples for how to construct a MultiIndex in the doc strings
    of the mentioned helper methods.
    """

    _hidden_attrs = ...
    _typ = ...
    _names = ...
    _levels = ...
    _codes = ...
    _comparables = ...
    sortorder: int | None
    def __new__(
        cls,
        levels=...,
        codes=...,
        sortorder=...,
        names=...,
        dtype=...,
        copy=...,
        name=...,
        verify_integrity: bool = ...,
    ): ...
    @classmethod
    def from_arrays(cls, arrays, sortorder=..., names=...) -> MultiIndex:
        """
        Convert arrays to MultiIndex.

        Parameters
        ----------
        arrays : list / sequence of array-likes
            Each array-like gives one level's value for each data point.
            len(arrays) is the number of levels.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
        >>> pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
        ...
    @classmethod
    @names_compat
    def from_tuples(
        cls,
        tuples: Iterable[tuple[Hashable, ...]],
        sortorder: int | None = ...,
        names: Sequence[Hashable] | None = ...,
    ) -> MultiIndex:
        """
        Convert list of tuples to MultiIndex.

        Parameters
        ----------
        tuples : list / sequence of tuple-likes
            Each tuple is the index of one row/column.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> tuples = [(1, 'red'), (1, 'blue'),
        ...           (2, 'red'), (2, 'blue')]
        >>> pd.MultiIndex.from_tuples(tuples, names=('number', 'color'))
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
        ...
    @classmethod
    def from_product(cls, iterables, sortorder=..., names=...) -> MultiIndex:
        """
        Make a MultiIndex from the cartesian product of multiple iterables.

        Parameters
        ----------
        iterables : list / sequence of iterables
            Each iterable has unique labels for each level of the index.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

            .. versionchanged:: 1.0.0

               If not explicitly provided, names will be inferred from the
               elements of iterables if an element has a name attribute

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> numbers = [0, 1, 2]
        >>> colors = ['green', 'purple']
        >>> pd.MultiIndex.from_product([numbers, colors],
        ...                            names=['number', 'color'])
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        """
        ...
    @classmethod
    def from_frame(cls, df: DataFrame, sortorder=..., names=...) -> MultiIndex:
        """
        Make a MultiIndex from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            DataFrame to be converted to MultiIndex.
        sortorder : int, optional
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list-like, optional
            If no names are provided, use the column names, or tuple of column
            names if the columns is a MultiIndex. If a sequence, overwrite
            names with the given sequence.

        Returns
        -------
        MultiIndex
            The MultiIndex representation of the given DataFrame.

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.

        Examples
        --------
        >>> df = pd.DataFrame([['HI', 'Temp'], ['HI', 'Precip'],
        ...                    ['NJ', 'Temp'], ['NJ', 'Precip']],
        ...                   columns=['a', 'b'])
        >>> df
              a       b
        0    HI    Temp
        1    HI  Precip
        2    NJ    Temp
        3    NJ  Precip

        >>> pd.MultiIndex.from_frame(df)
        MultiIndex([('HI',   'Temp'),
                    ('HI', 'Precip'),
                    ('NJ',   'Temp'),
                    ('NJ', 'Precip')],
                   names=['a', 'b'])

        Using explicit names, instead of the column names

        >>> pd.MultiIndex.from_frame(df, names=['state', 'observation'])
        MultiIndex([('HI',   'Temp'),
                    ('HI', 'Precip'),
                    ('NJ',   'Temp'),
                    ('NJ', 'Precip')],
                   names=['state', 'observation'])
        """
        ...
    @property
    def values(self) -> np.ndarray: ...
    @property
    def array(self):  # -> NoReturn:
        """
        Raises a ValueError for `MultiIndex` because there's no single
        array backing a MultiIndex.

        Raises
        ------
        ValueError
        """
        ...
    @cache_readonly
    def dtypes(self) -> Series:
        """
        Return the dtypes as a Series for the underlying MultiIndex
        """
        ...
    def __len__(self) -> int: ...
    @cache_readonly
    def levels(self) -> FrozenList: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "levels"])
    def set_levels(
        self, levels, level=..., inplace=..., verify_integrity: bool = ...
    ):  # -> Self@MultiIndex | MultiIndex | None:
        """
        Set new levels on MultiIndex. Defaults to returning new index.

        Parameters
        ----------
        levels : sequence or list of sequence
            New level(s) to apply.
        level : int, level name, or sequence of int/level names (default None)
            Level(s) to set (None for all levels).
        inplace : bool
            If True, mutates in place.

            .. deprecated:: 1.2.0
        verify_integrity : bool, default True
            If True, checks that levels and codes are compatible.

        Returns
        -------
        new index (of same type and class...etc) or None
            The same type as the caller or None if ``inplace=True``.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_tuples(
        ...     [
        ...         (1, "one"),
        ...         (1, "two"),
        ...         (2, "one"),
        ...         (2, "two"),
        ...         (3, "one"),
        ...         (3, "two")
        ...     ],
        ...     names=["foo", "bar"]
        ... )
        >>> idx
        MultiIndex([(1, 'one'),
            (1, 'two'),
            (2, 'one'),
            (2, 'two'),
            (3, 'one'),
            (3, 'two')],
           names=['foo', 'bar'])

        >>> idx.set_levels([['a', 'b', 'c'], [1, 2]])
        MultiIndex([('a', 1),
                    ('a', 2),
                    ('b', 1),
                    ('b', 2),
                    ('c', 1),
                    ('c', 2)],
                   names=['foo', 'bar'])
        >>> idx.set_levels(['a', 'b', 'c'], level=0)
        MultiIndex([('a', 'one'),
                    ('a', 'two'),
                    ('b', 'one'),
                    ('b', 'two'),
                    ('c', 'one'),
                    ('c', 'two')],
                   names=['foo', 'bar'])
        >>> idx.set_levels(['a', 'b'], level='bar')
        MultiIndex([(1, 'a'),
                    (1, 'b'),
                    (2, 'a'),
                    (2, 'b'),
                    (3, 'a'),
                    (3, 'b')],
                   names=['foo', 'bar'])

        If any of the levels passed to ``set_levels()`` exceeds the
        existing length, all of the values from that argument will
        be stored in the MultiIndex levels, though the values will
        be truncated in the MultiIndex output.

        >>> idx.set_levels([['a', 'b', 'c'], [1, 2, 3, 4]], level=[0, 1])
        MultiIndex([('a', 1),
            ('a', 2),
            ('b', 1),
            ('b', 2),
            ('c', 1),
            ('c', 2)],
           names=['foo', 'bar'])
        >>> idx.set_levels([['a', 'b', 'c'], [1, 2, 3, 4]], level=[0, 1]).levels
        FrozenList([['a', 'b', 'c'], [1, 2, 3, 4]])
        """
        ...
    @property
    def nlevels(self) -> int:
        """
        Integer number of levels in this MultiIndex.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a'], ['b'], ['c']])
        >>> mi
        MultiIndex([('a', 'b', 'c')],
                   )
        >>> mi.nlevels
        3
        """
        ...
    @property
    def levshape(self) -> Shape:
        """
        A tuple with the length of each level.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a'], ['b'], ['c']])
        >>> mi
        MultiIndex([('a', 'b', 'c')],
                   )
        >>> mi.levshape
        (1, 1, 1)
        """
        ...
    @property
    def codes(self): ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "codes"])
    def set_codes(
        self, codes, level=..., inplace=..., verify_integrity: bool = ...
    ):  # -> Self@MultiIndex | MultiIndex | None:
        """
        Set new codes on MultiIndex. Defaults to returning new index.

        Parameters
        ----------
        codes : sequence or list of sequence
            New codes to apply.
        level : int, level name, or sequence of int/level names (default None)
            Level(s) to set (None for all levels).
        inplace : bool
            If True, mutates in place.

            .. deprecated:: 1.2.0
        verify_integrity : bool, default True
            If True, checks that levels and codes are compatible.

        Returns
        -------
        new index (of same type and class...etc) or None
            The same type as the caller or None if ``inplace=True``.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_tuples(
        ...     [(1, "one"), (1, "two"), (2, "one"), (2, "two")], names=["foo", "bar"]
        ... )
        >>> idx
        MultiIndex([(1, 'one'),
            (1, 'two'),
            (2, 'one'),
            (2, 'two')],
           names=['foo', 'bar'])

        >>> idx.set_codes([[1, 0, 1, 0], [0, 0, 1, 1]])
        MultiIndex([(2, 'one'),
                    (1, 'one'),
                    (2, 'two'),
                    (1, 'two')],
                   names=['foo', 'bar'])
        >>> idx.set_codes([1, 0, 1, 0], level=0)
        MultiIndex([(2, 'one'),
                    (1, 'two'),
                    (2, 'one'),
                    (1, 'two')],
                   names=['foo', 'bar'])
        >>> idx.set_codes([0, 0, 1, 1], level='bar')
        MultiIndex([(1, 'one'),
                    (1, 'one'),
                    (2, 'two'),
                    (2, 'two')],
                   names=['foo', 'bar'])
        >>> idx.set_codes([[1, 0, 1, 0], [0, 0, 1, 1]], level=[0, 1])
        MultiIndex([(2, 'one'),
                    (1, 'one'),
                    (2, 'two'),
                    (1, 'two')],
                   names=['foo', 'bar'])
        """
        ...
    def copy(self, names=..., dtype=..., levels=..., codes=..., deep=..., name=...):
        """
        Make a copy of this object. Names, dtype, levels and codes can be
        passed and will be set on new copy.

        Parameters
        ----------
        names : sequence, optional
        dtype : numpy dtype or pandas type, optional

            .. deprecated:: 1.2.0
        levels : sequence, optional

            .. deprecated:: 1.2.0
        codes : sequence, optional

            .. deprecated:: 1.2.0
        deep : bool, default False
        name : Label
            Kept for compatibility with 1-dimensional Index. Should not be used.

        Returns
        -------
        MultiIndex

        Notes
        -----
        In most cases, there should be no functional difference from using
        ``deep``, but if ``deep`` is passed it will attempt to deepcopy.
        This could be potentially expensive on large MultiIndex objects.
        """
        ...
    def __array__(self, dtype=...) -> np.ndarray:
        """the array interface, return my values"""
        ...
    def view(self, cls=...):  # -> MultiIndex | Self@MultiIndex:
        """this is defined as a copy with the same identity"""
        ...
    @doc(Index.__contains__)
    def __contains__(self, key: Any) -> bool: ...
    @cache_readonly
    def dtype(self) -> np.dtype: ...
    @doc(Index.memory_usage)
    def memory_usage(self, deep: bool = ...) -> int: ...
    @cache_readonly
    def nbytes(self) -> int:
        """return the number of bytes in the underlying data"""
        ...
    def format(
        self,
        name: bool | None = ...,
        formatter: Callable | None = ...,
        na_rep: str | None = ...,
        names: bool = ...,
        space: int = ...,
        sparsify=...,
        adjoin: bool = ...,
    ) -> list: ...
    names = ...
    @cache_readonly
    def inferred_type(self) -> str: ...
    @cache_readonly
    def is_monotonic_increasing(self) -> bool:
        """
        return if the index is monotonic increasing (only equal or
        increasing) values.
        """
        ...
    @cache_readonly
    def is_monotonic_decreasing(self) -> bool:
        """
        return if the index is monotonic decreasing (only equal or
        decreasing) values.
        """
        ...
    @doc(Index.duplicated)
    def duplicated(self, keep=...) -> np.ndarray: ...
    _duplicated = ...
    def fillna(self, value=..., downcast=...):
        """
        fillna is not implemented for MultiIndex
        """
        ...
    @doc(Index.dropna)
    def dropna(self, how: str = ...) -> MultiIndex: ...
    def get_level_values(self, level):  # -> Index:
        """
        Return vector of label values for requested level.

        Length of returned vector is equal to the length of the index.

        Parameters
        ----------
        level : int or str
            ``level`` is either the integer position of the level in the
            MultiIndex, or the name of the level.

        Returns
        -------
        values : Index
            Values is a level of this MultiIndex converted to
            a single :class:`Index` (or subclass thereof).

        Examples
        --------
        Create a MultiIndex:

        >>> mi = pd.MultiIndex.from_arrays((list('abc'), list('def')))
        >>> mi.names = ['level_1', 'level_2']

        Get level values by supplying level as either integer or name:

        >>> mi.get_level_values(0)
        Index(['a', 'b', 'c'], dtype='object', name='level_1')
        >>> mi.get_level_values('level_2')
        Index(['d', 'e', 'f'], dtype='object', name='level_2')
        """
        ...
    @doc(Index.unique)
    def unique(self, level=...): ...
    def to_frame(self, index: bool = ..., name=...) -> DataFrame:
        """
        Create a DataFrame with the levels of the MultiIndex as columns.

        Column ordering is determined by the DataFrame constructor with data as
        a dict.

        Parameters
        ----------
        index : bool, default True
            Set the index of the returned DataFrame as the original MultiIndex.

        name : list / sequence of str, optional
            The passed names should substitute index level names.

        Returns
        -------
        DataFrame : a DataFrame containing the original MultiIndex data.

        See Also
        --------
        DataFrame : Two-dimensional, size-mutable, potentially heterogeneous
            tabular data.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a', 'b'], ['c', 'd']])
        >>> mi
        MultiIndex([('a', 'c'),
                    ('b', 'd')],
                   )

        >>> df = mi.to_frame()
        >>> df
             0  1
        a c  a  c
        b d  b  d

        >>> df = mi.to_frame(index=False)
        >>> df
           0  1
        0  a  c
        1  b  d

        >>> df = mi.to_frame(name=['x', 'y'])
        >>> df
             x  y
        a c  a  c
        b d  b  d
        """
        ...
    def to_flat_index(self) -> Index:
        """
        Convert a MultiIndex to an Index of Tuples containing the level values.

        Returns
        -------
        pd.Index
            Index with the MultiIndex data represented in Tuples.

        See Also
        --------
        MultiIndex.from_tuples : Convert flat index back to MultiIndex.

        Notes
        -----
        This method will simply return the caller if called by anything other
        than a MultiIndex.

        Examples
        --------
        >>> index = pd.MultiIndex.from_product(
        ...     [['foo', 'bar'], ['baz', 'qux']],
        ...     names=['a', 'b'])
        >>> index.to_flat_index()
        Index([('foo', 'baz'), ('foo', 'qux'),
               ('bar', 'baz'), ('bar', 'qux')],
              dtype='object')
        """
        ...
    def is_lexsorted(self) -> bool: ...
    @property
    def lexsort_depth(self): ...
    def remove_unused_levels(self) -> MultiIndex:
        """
        Create new MultiIndex from current that removes unused levels.

        Unused level(s) means levels that are not expressed in the
        labels. The resulting MultiIndex will have the same outward
        appearance, meaning the same .values and ordering. It will
        also be .equals() to the original.

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> mi = pd.MultiIndex.from_product([range(2), list('ab')])
        >>> mi
        MultiIndex([(0, 'a'),
                    (0, 'b'),
                    (1, 'a'),
                    (1, 'b')],
                   )

        >>> mi[2:]
        MultiIndex([(1, 'a'),
                    (1, 'b')],
                   )

        The 0 from the first level is not represented
        and can be removed

        >>> mi2 = mi[2:].remove_unused_levels()
        >>> mi2.levels
        FrozenList([[1], ['a', 'b']])
        """
        ...
    def __reduce__(
        self,
    ):  # -> tuple[(cls: Unknown, d: Unknown) -> (NDArrayBackedExtensionIndex | PeriodIndex | ABCMultiIndex | Unknown), tuple[Type[Self@MultiIndex], dict[str, list[Unknown] | int | list[Any] | None]], None]:
        """Necessary for making this object picklable"""
        ...
    def __getitem__(self, key): ...
    @Appender(_index_shared_docs["take"] % _index_doc_kwargs)
    def take(
        self: MultiIndex,
        indices,
        axis: int = ...,
        allow_fill: bool = ...,
        fill_value=...,
        **kwargs
    ) -> MultiIndex: ...
    def append(self, other):
        """
        Append a collection of Index options together

        Parameters
        ----------
        other : Index or list/tuple of indices

        Returns
        -------
        appended : Index
        """
        ...
    def argsort(self, *args, **kwargs) -> np.ndarray: ...
    @Appender(_index_shared_docs["repeat"] % _index_doc_kwargs)
    def repeat(self, repeats: int, axis=...) -> MultiIndex: ...
    def drop(self, codes, level=..., errors=...):
        """
        Make new MultiIndex with passed list of codes deleted

        Parameters
        ----------
        codes : array-like
            Must be a list of tuples when level is not specified
        level : int or level name, default None
        errors : str, default 'raise'

        Returns
        -------
        dropped : MultiIndex
        """
        ...
    def swaplevel(self, i=..., j=...) -> MultiIndex:
        """
        Swap level i with level j.

        Calling this method does not change the ordering of the values.

        Parameters
        ----------
        i : int, str, default -2
            First level of index to be swapped. Can pass level name as string.
            Type of parameters can be mixed.
        j : int, str, default -1
            Second level of index to be swapped. Can pass level name as string.
            Type of parameters can be mixed.

        Returns
        -------
        MultiIndex
            A new MultiIndex.

        See Also
        --------
        Series.swaplevel : Swap levels i and j in a MultiIndex.
        Dataframe.swaplevel : Swap levels i and j in a MultiIndex on a
            particular axis.

        Examples
        --------
        >>> mi = pd.MultiIndex(levels=[['a', 'b'], ['bb', 'aa']],
        ...                    codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        >>> mi
        MultiIndex([('a', 'bb'),
                    ('a', 'aa'),
                    ('b', 'bb'),
                    ('b', 'aa')],
                   )
        >>> mi.swaplevel(0, 1)
        MultiIndex([('bb', 'a'),
                    ('aa', 'a'),
                    ('bb', 'b'),
                    ('aa', 'b')],
                   )
        """
        ...
    def reorder_levels(self, order) -> MultiIndex:
        """
        Rearrange levels using input order. May not drop or duplicate levels.

        Parameters
        ----------
        order : list of int or list of str
            List representing new level order. Reference level by number
            (position) or by key (label).

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([[1, 2], [3, 4]], names=['x', 'y'])
        >>> mi
        MultiIndex([(1, 3),
                    (2, 4)],
                   names=['x', 'y'])

        >>> mi.reorder_levels(order=[1, 0])
        MultiIndex([(3, 1),
                    (4, 2)],
                   names=['y', 'x'])

        >>> mi.reorder_levels(order=['y', 'x'])
        MultiIndex([(3, 1),
                    (4, 2)],
                   names=['y', 'x'])
        """
        ...
    def sortlevel(
        self, level=..., ascending: bool = ..., sort_remaining: bool = ...
    ) -> tuple[MultiIndex, np.ndarray]:
        """
        Sort MultiIndex at the requested level.

        The result will respect the original ordering of the associated
        factor at that level.

        Parameters
        ----------
        level : list-like, int or str, default 0
            If a string is given, must be a name of the level.
            If list-like must be names or ints of levels.
        ascending : bool, default True
            False to sort in descending order.
            Can also be a list to specify a directed ordering.
        sort_remaining : sort by the remaining levels after level

        Returns
        -------
        sorted_index : pd.MultiIndex
            Resulting index.
        indexer : np.ndarray
            Indices of output values in original index.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([[0, 0], [2, 1]])
        >>> mi
        MultiIndex([(0, 2),
                    (0, 1)],
                   )

        >>> mi.sortlevel()
        (MultiIndex([(0, 1),
                    (0, 2)],
                   ), array([1, 0]))

        >>> mi.sortlevel(sort_remaining=False)
        (MultiIndex([(0, 2),
                    (0, 1)],
                   ), array([0, 1]))

        >>> mi.sortlevel(1)
        (MultiIndex([(0, 1),
                    (0, 2)],
                   ), array([1, 0]))

        >>> mi.sortlevel(1, ascending=False)
        (MultiIndex([(0, 2),
                    (0, 1)],
                   ), array([0, 1]))
        """
        ...
    def reindex(
        self, target, method=..., level=..., limit=..., tolerance=...
    ) -> tuple[MultiIndex, np.ndarray | None]:
        """
        Create index with target's values (move/add/delete values as necessary)

        Returns
        -------
        new_index : pd.MultiIndex
            Resulting index
        indexer : np.ndarray[np.intp] or None
            Indices of output values in original index.

        """
        ...
    def get_slice_bound(
        self, label: Hashable | Sequence[Hashable], side: str, kind: str | None = ...
    ) -> int:
        """
        For an ordered MultiIndex, compute slice bound
        that corresponds to given label.

        Returns leftmost (one-past-the-rightmost if `side=='right') position
        of given label.

        Parameters
        ----------
        label : object or tuple of objects
        side : {'left', 'right'}
        kind : {'loc', 'getitem', None}

        Returns
        -------
        int
            Index of label.

        Notes
        -----
        This method only works if level 0 index of the MultiIndex is lexsorted.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abbc'), list('gefd')])

        Get the locations from the leftmost 'b' in the first level
        until the end of the multiindex:

        >>> mi.get_slice_bound('b', side="left")
        1

        Like above, but if you get the locations from the rightmost
        'b' in the first level and 'f' in the second level:

        >>> mi.get_slice_bound(('b','f'), side="right")
        3

        See Also
        --------
        MultiIndex.get_loc : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.
        """
        ...
    def slice_locs(self, start=..., end=..., step=..., kind=...):  # -> tuple[int, int]:
        """
        For an ordered MultiIndex, compute the slice locations for input
        labels.

        The input labels can be tuples representing partial levels, e.g. for a
        MultiIndex with 3 levels, you can pass a single value (corresponding to
        the first level), or a 1-, 2-, or 3-tuple.

        Parameters
        ----------
        start : label or tuple, default None
            If None, defaults to the beginning
        end : label or tuple
            If None, defaults to the end
        step : int or None
            Slice step
        kind : string, optional, defaults None

        Returns
        -------
        (start, end) : (int, int)

        Notes
        -----
        This method only works if the MultiIndex is properly lexsorted. So,
        if only the first 2 levels of a 3-level MultiIndex are lexsorted,
        you can only pass two levels to ``.slice_locs``.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abbd'), list('deff')],
        ...                                names=['A', 'B'])

        Get the slice locations from the beginning of 'b' in the first level
        until the end of the multiindex:

        >>> mi.slice_locs(start='b')
        (1, 4)

        Like above, but stop at the end of 'b' in the first level and 'f' in
        the second level:

        >>> mi.slice_locs(start='b', end=('b', 'f'))
        (1, 3)

        See Also
        --------
        MultiIndex.get_loc : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.
        """
        ...
    def get_loc(self, key, method=...):
        """
        Get location for a label or a tuple of labels.

        The location is returned as an integer/slice or boolean
        mask.

        Parameters
        ----------
        key : label or tuple of labels (one for each level)
        method : None

        Returns
        -------
        loc : int, slice object or boolean mask
            If the key is past the lexsort depth, the return may be a
            boolean mask array, otherwise it is always a slice or int.

        See Also
        --------
        Index.get_loc : The get_loc method for (single-level) index.
        MultiIndex.slice_locs : Get slice location given start label(s) and
                                end label(s).
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.

        Notes
        -----
        The key cannot be a slice, list of same-level labels, a boolean mask,
        or a sequence of such. If you want to use those, use
        :meth:`MultiIndex.get_locs` instead.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abb'), list('def')])

        >>> mi.get_loc('b')
        slice(1, 3, None)

        >>> mi.get_loc(('b', 'e'))
        1
        """
        ...
    def get_loc_level(
        self, key, level=..., drop_level: bool = ...
    ):  # -> tuple[ndarray | None, Unknown] | tuple[ndarray | slice, Unknown] | tuple[Any, None] | tuple[Unknown, Unknown] | tuple[slice | ndarray | NoReturn, Unknown]:
        """
        Get location and sliced index for requested label(s)/level(s).

        Parameters
        ----------
        key : label or sequence of labels
        level : int/level name or list thereof, optional
        drop_level : bool, default True
            If ``False``, the resulting index will not drop any level.

        Returns
        -------
        loc : A 2-tuple where the elements are:
              Element 0: int, slice object or boolean array
              Element 1: The resulting sliced multiindex/index. If the key
              contains all levels, this will be ``None``.

        See Also
        --------
        MultiIndex.get_loc  : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abb'), list('def')],
        ...                                names=['A', 'B'])

        >>> mi.get_loc_level('b')
        (slice(1, 3, None), Index(['e', 'f'], dtype='object', name='B'))

        >>> mi.get_loc_level('e', level='B')
        (array([False,  True, False]), Index(['b'], dtype='object', name='A'))

        >>> mi.get_loc_level(['b', 'e'])
        (1, None)
        """
        ...
    def get_locs(self, seq):
        """
        Get location for a sequence of labels.

        Parameters
        ----------
        seq : label, slice, list, mask or a sequence of such
           You should use one of the above for each level.
           If a level should not be used, set it to ``slice(None)``.

        Returns
        -------
        numpy.ndarray
            NumPy array of integers suitable for passing to iloc.

        See Also
        --------
        MultiIndex.get_loc : Get location for a label or a tuple of labels.
        MultiIndex.slice_locs : Get slice location given start label(s) and
                                end label(s).

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abb'), list('def')])

        >>> mi.get_locs('b')  # doctest: +SKIP
        array([1, 2], dtype=int64)

        >>> mi.get_locs([slice(None), ['e', 'f']])  # doctest: +SKIP
        array([1, 2], dtype=int64)

        >>> mi.get_locs([[True, False, True], slice('e', 'f')])  # doctest: +SKIP
        array([2], dtype=int64)
        """
        ...
    def truncate(self, before=..., after=...) -> MultiIndex:
        """
        Slice index between two labels / tuples, return new MultiIndex

        Parameters
        ----------
        before : label or tuple, can be partial. Default None
            None defaults to start
        after : label or tuple, can be partial. Default None
            None defaults to end

        Returns
        -------
        truncated : MultiIndex
        """
        ...
    def equals(self, other: object) -> bool:
        """
        Determines if two MultiIndex objects have the same labeling information
        (the levels themselves do not necessarily have to be the same)

        See Also
        --------
        equal_levels
        """
        ...
    def equal_levels(self, other: MultiIndex) -> bool:
        """
        Return True if the levels of both MultiIndex objects are the same

        """
        ...
    @doc(Index.astype)
    def astype(self, dtype, copy: bool = ...): ...
    def insert(self, loc: int, item) -> MultiIndex:
        """
        Make new MultiIndex inserting new item at location

        Parameters
        ----------
        loc : int
        item : tuple
            Must be same length as number of levels in the MultiIndex

        Returns
        -------
        new_index : Index
        """
        ...
    def delete(self, loc) -> MultiIndex:
        """
        Make new index with passed location deleted

        Returns
        -------
        new_index : MultiIndex
        """
        ...
    @doc(Index.isin)
    def isin(self, values, level=...) -> np.ndarray: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "names"])
    def set_names(self, names, level=..., inplace: bool = ...) -> MultiIndex | None: ...
    rename = ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self"])
    def drop_duplicates(self, keep: str | bool = ...) -> MultiIndex: ...
    __add__ = ...
    __radd__ = ...
    __iadd__ = ...
    __sub__ = ...
    __rsub__ = ...
    __isub__ = ...
    __pow__ = ...
    __rpow__ = ...
    __mul__ = ...
    __rmul__ = ...
    __floordiv__ = ...
    __rfloordiv__ = ...
    __truediv__ = ...
    __rtruediv__ = ...
    __mod__ = ...
    __rmod__ = ...
    __divmod__ = ...
    __rdivmod__ = ...
    __neg__ = ...
    __pos__ = ...
    __abs__ = ...
    __inv__ = ...

def sparsify_labels(label_list, start: int = ..., sentinel=...): ...
def maybe_droplevels(index: Index, key) -> Index:
    """
    Attempt to drop level or levels from the given index.

    Parameters
    ----------
    index: Index
    key : scalar or tuple

    Returns
    -------
    Index
    """
    ...
