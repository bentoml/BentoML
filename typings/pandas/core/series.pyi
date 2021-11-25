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

"""
Data structure for 1-dimensional cross-sectional and time series data
"""
if TYPE_CHECKING: ...
__all__ = ["Series"]
_shared_doc_kwargs = ...

class Series(base.IndexOpsMixin, generic.NDFrame):
    """
    One-dimensional ndarray with axis labels (including time series).

    Labels need not be unique but must be a hashable type. The object
    supports both integer- and label-based indexing and provides a host of
    methods for performing operations involving the index. Statistical
    methods from ndarray have been overridden to automatically exclude
    missing data (currently represented as NaN).

    Operations between Series (+, -, /, *, **) align values based on their
    associated index values-- they need not be the same length. The result
    index will be the sorted union of the two indexes.

    Parameters
    ----------
    data : array-like, Iterable, dict, or scalar value
        Contains data stored in Series. If data is a dict, argument order is
        maintained.
    index : array-like or Index (1d)
        Values must be hashable and have the same length as `data`.
        Non-unique index values are allowed. Will default to
        RangeIndex (0, 1, 2, ..., n) if not provided. If data is dict-like
        and index is None, then the keys in the data are used as the index. If the
        index is not None, the resulting Series is reindexed with the index values.
    dtype : str, numpy.dtype, or ExtensionDtype, optional
        Data type for the output Series. If not specified, this will be
        inferred from `data`.
        See the :ref:`user guide <basics.dtypes>` for more usages.
    name : str, optional
        The name to give to the Series.
    copy : bool, default False
        Copy input data. Only affects Series or 1d ndarray input. See examples.

    Examples
    --------
    Constructing Series from a dictionary with an Index specified

    >>> d = {'a': 1, 'b': 2, 'c': 3}
    >>> ser = pd.Series(data=d, index=['a', 'b', 'c'])
    >>> ser
    a   1
    b   2
    c   3
    dtype: int64

    The keys of the dictionary match with the Index values, hence the Index
    values have no effect.

    >>> d = {'a': 1, 'b': 2, 'c': 3}
    >>> ser = pd.Series(data=d, index=['x', 'y', 'z'])
    >>> ser
    x   NaN
    y   NaN
    z   NaN
    dtype: float64

    Note that the Index is first build with the keys from the dictionary.
    After this the Series is reindexed with the given Index values, hence we
    get all NaN as a result.

    Constructing Series from a list with `copy=False`.

    >>> r = [1, 2]
    >>> ser = pd.Series(r, copy=False)
    >>> ser.iloc[0] = 999
    >>> r
    [1, 2]
    >>> ser
    0    999
    1      2
    dtype: int64

    Due to input data type the Series has a `copy` of
    the original data even though `copy=False`, so
    the data is unchanged.

    Constructing Series from a 1d ndarray with `copy=False`.

    >>> r = np.array([1, 2])
    >>> ser = pd.Series(r, copy=False)
    >>> ser.iloc[0] = 999
    >>> r
    array([999,   2])
    >>> ser
    0    999
    1      2
    dtype: int64

    Due to input data type the Series has a `view` on
    the original data, so
    the data is changed as well.
    """

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
    def dtype(self) -> DtypeObj:
        """
        Return the dtype object of the underlying data.
        """
        ...
    @property
    def dtypes(self) -> DtypeObj:
        """
        Return the dtype object of the underlying data.
        """
        ...
    @property
    def name(self) -> Hashable:
        """
        Return the name of the Series.

        The name of a Series becomes its index or column name if it is used
        to form a DataFrame. It is also used whenever displaying the Series
        using the interpreter.

        Returns
        -------
        label (hashable object)
            The name of the Series, also the column name if part of a DataFrame.

        See Also
        --------
        Series.rename : Sets the Series name when given a scalar input.
        Index.name : Corresponding Index property.

        Examples
        --------
        The Series name can be set initially when calling the constructor.

        >>> s = pd.Series([1, 2, 3], dtype=np.int64, name='Numbers')
        >>> s
        0    1
        1    2
        2    3
        Name: Numbers, dtype: int64
        >>> s.name = "Integers"
        >>> s
        0    1
        1    2
        2    3
        Name: Integers, dtype: int64

        The name of a Series within a DataFrame is its column name.

        >>> df = pd.DataFrame([[1, 2], [3, 4], [5, 6]],
        ...                   columns=["Odd Numbers", "Even Numbers"])
        >>> df
           Odd Numbers  Even Numbers
        0            1             2
        1            3             4
        2            5             6
        >>> df["Even Numbers"].name
        'Even Numbers'
        """
        ...
    @name.setter
    def name(self, value: Hashable) -> None: ...
    @property
    def values(self):  # -> ArrayLike:
        """
        Return Series as ndarray or ndarray-like depending on the dtype.

        .. warning::

           We recommend using :attr:`Series.array` or
           :meth:`Series.to_numpy`, depending on whether you need
           a reference to the underlying data or a NumPy array.

        Returns
        -------
        numpy.ndarray or ndarray-like

        See Also
        --------
        Series.array : Reference to the underlying data.
        Series.to_numpy : A NumPy array representing the underlying data.

        Examples
        --------
        >>> pd.Series([1, 2, 3]).values
        array([1, 2, 3])

        >>> pd.Series(list('aabc')).values
        array(['a', 'a', 'b', 'c'], dtype=object)

        >>> pd.Series(list('aabc')).astype('category').values
        ['a', 'a', 'b', 'c']
        Categories (3, object): ['a', 'b', 'c']

        Timezone aware datetime data is converted to UTC:

        >>> pd.Series(pd.date_range('20130101', periods=3,
        ...                         tz='US/Eastern')).values
        array(['2013-01-01T05:00:00.000000000',
               '2013-01-02T05:00:00.000000000',
               '2013-01-03T05:00:00.000000000'], dtype='datetime64[ns]')
        """
        ...
    @Appender(base.IndexOpsMixin.array.__doc__)
    @property
    def array(self) -> ExtensionArray: ...
    def ravel(self, order=...):  # -> ndarray | ExtensionArray:
        """
        Return the flattened underlying data as an ndarray.

        Returns
        -------
        numpy.ndarray or ndarray-like
            Flattened data of the Series.

        See Also
        --------
        numpy.ndarray.ravel : Return a flattened array.
        """
        ...
    def __len__(self) -> int:
        """
        Return the length of the Series.
        """
        ...
    def view(self, dtype: Dtype | None = ...) -> Series:
        """
        Create a new view of the Series.

        This function will return a new Series with a view of the same
        underlying values in memory, optionally reinterpreted with a new data
        type. The new data type must preserve the same size in bytes as to not
        cause index misalignment.

        Parameters
        ----------
        dtype : data type
            Data type object or one of their string representations.

        Returns
        -------
        Series
            A new Series object as a view of the same data in memory.

        See Also
        --------
        numpy.ndarray.view : Equivalent numpy function to create a new view of
            the same data in memory.

        Notes
        -----
        Series are instantiated with ``dtype=float64`` by default. While
        ``numpy.ndarray.view()`` will return a view with the same data type as
        the original array, ``Series.view()`` (without specified dtype)
        will try using ``float64`` and may fail if the original data type size
        in bytes is not the same.

        Examples
        --------
        >>> s = pd.Series([-2, -1, 0, 1, 2], dtype='int8')
        >>> s
        0   -2
        1   -1
        2    0
        3    1
        4    2
        dtype: int8

        The 8 bit signed integer representation of `-1` is `0b11111111`, but
        the same bytes represent 255 if read as an 8 bit unsigned integer:

        >>> us = s.view('uint8')
        >>> us
        0    254
        1    255
        2      0
        3      1
        4      2
        dtype: uint8

        The views share the same underlying values:

        >>> us[0] = 128
        >>> s
        0   -128
        1     -1
        2      0
        3      1
        4      2
        dtype: int8
        """
        ...
    _HANDLED_TYPES = ...
    def __array__(self, dtype: NpDtype | None = ...) -> np.ndarray:
        """
        Return the values as a NumPy array.

        Users should not call this directly. Rather, it is invoked by
        :func:`numpy.array` and :func:`numpy.asarray`.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to use for the resulting NumPy array. By default,
            the dtype is inferred from the data.

        Returns
        -------
        numpy.ndarray
            The values in the series converted to a :class:`numpy.ndarray`
            with the specified `dtype`.

        See Also
        --------
        array : Create a new array from data.
        Series.array : Zero-copy view to the array backing the Series.
        Series.to_numpy : Series method for similar behavior.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3])
        >>> np.asarray(ser)
        array([1, 2, 3])

        For timezone-aware data, the timezones may be retained with
        ``dtype='object'``

        >>> tzser = pd.Series(pd.date_range('2000', periods=2, tz="CET"))
        >>> np.asarray(tzser, dtype="object")
        array([Timestamp('2000-01-01 00:00:00+0100', tz='CET'),
               Timestamp('2000-01-02 00:00:00+0100', tz='CET')],
              dtype=object)

        Or the values may be localized to UTC and the tzinfo discarded with
        ``dtype='datetime64[ns]'``

        >>> np.asarray(tzser, dtype="datetime64[ns]")  # doctest: +ELLIPSIS
        array(['1999-12-31T23:00:00.000000000', ...],
              dtype='datetime64[ns]')
        """
        ...
    __float__ = ...
    __long__ = ...
    __int__ = ...
    @property
    def axes(self) -> list[Index]:
        """
        Return a list of the row axis labels.
        """
        ...
    @Appender(generic.NDFrame.take.__doc__)
    def take(self, indices, axis=..., is_copy=..., **kwargs) -> Series: ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value) -> None: ...
    def repeat(self, repeats, axis=...) -> Series:
        """
        Repeat elements of a Series.

        Returns a new Series where each element of the current Series
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            Series.
        axis : None
            Must be ``None``. Has no effect but is accepted for compatibility
            with numpy.

        Returns
        -------
        Series
            Newly created Series with repeated elements.

        See Also
        --------
        Index.repeat : Equivalent function for Index.
        numpy.repeat : Similar method for :class:`numpy.ndarray`.

        Examples
        --------
        >>> s = pd.Series(['a', 'b', 'c'])
        >>> s
        0    a
        1    b
        2    c
        dtype: object
        >>> s.repeat(2)
        0    a
        0    a
        1    b
        1    b
        2    c
        2    c
        dtype: object
        >>> s.repeat([1, 2, 3])
        0    a
        1    b
        1    b
        2    c
        2    c
        2    c
        dtype: object
        """
        ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "level"])
    def reset_index(
        self, level=..., drop=..., name=..., inplace=...
    ):  # -> Series | DataFrame | None:
        """
        Generate a new DataFrame or Series with the index reset.

        This is useful when the index needs to be treated as a column, or
        when the index is meaningless and needs to be reset to the default
        before another operation.

        Parameters
        ----------
        level : int, str, tuple, or list, default optional
            For a Series with a MultiIndex, only remove the specified levels
            from the index. Removes all levels by default.
        drop : bool, default False
            Just reset the index, without inserting it as a column in
            the new DataFrame.
        name : object, optional
            The name to use for the column containing the original Series
            values. Uses ``self.name`` by default. This argument is ignored
            when `drop` is True.
        inplace : bool, default False
            Modify the Series in place (do not create a new object).

        Returns
        -------
        Series or DataFrame or None
            When `drop` is False (the default), a DataFrame is returned.
            The newly created columns will come first in the DataFrame,
            followed by the original Series values.
            When `drop` is True, a `Series` is returned.
            In either case, if ``inplace=True``, no value is returned.

        See Also
        --------
        DataFrame.reset_index: Analogous function for DataFrame.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4], name='foo',
        ...               index=pd.Index(['a', 'b', 'c', 'd'], name='idx'))

        Generate a DataFrame with default index.

        >>> s.reset_index()
          idx  foo
        0   a    1
        1   b    2
        2   c    3
        3   d    4

        To specify the name of the new column use `name`.

        >>> s.reset_index(name='values')
          idx  values
        0   a       1
        1   b       2
        2   c       3
        3   d       4

        To generate a new Series with the default set `drop` to True.

        >>> s.reset_index(drop=True)
        0    1
        1    2
        2    3
        3    4
        Name: foo, dtype: int64

        To update the Series in place, without generating a new one
        set `inplace` to True. Note that it also requires ``drop=True``.

        >>> s.reset_index(inplace=True, drop=True)
        >>> s
        0    1
        1    2
        2    3
        3    4
        Name: foo, dtype: int64

        The `level` parameter is interesting for Series with a multi-level
        index.

        >>> arrays = [np.array(['bar', 'bar', 'baz', 'baz']),
        ...           np.array(['one', 'two', 'one', 'two'])]
        >>> s2 = pd.Series(
        ...     range(4), name='foo',
        ...     index=pd.MultiIndex.from_arrays(arrays,
        ...                                     names=['a', 'b']))

        To remove a specific level from the Index, use `level`.

        >>> s2.reset_index(level='a')
               a  foo
        b
        one  bar    0
        two  bar    1
        one  baz    2
        two  baz    3

        If `level` is not set, all levels are removed from the Index.

        >>> s2.reset_index()
             a    b  foo
        0  bar  one    0
        1  bar  two    1
        2  baz  one    2
        3  baz  two    3
        """
        ...
    def __repr__(self) -> str:
        """
        Return a string representation for a particular Series.
        """
        ...
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
    ):  # -> str | None:
        """
        Render a string representation of the Series.

        Parameters
        ----------
        buf : StringIO-like, optional
            Buffer to write to.
        na_rep : str, optional
            String representation of NaN to use, default 'NaN'.
        float_format : one-parameter function, optional
            Formatter function to apply to columns' elements if they are
            floats, default None.
        header : bool, default True
            Add the Series header (index name).
        index : bool, optional
            Add index (row) labels, default True.
        length : bool, default False
            Add the Series length.
        dtype : bool, default False
            Add the Series dtype.
        name : bool, default False
            Add the Series name if not None.
        max_rows : int, optional
            Maximum number of rows to show before truncating. If None, show
            all.
        min_rows : int, optional
            The number of rows to display in a truncated repr (when number
            of rows is above `max_rows`).

        Returns
        -------
        str or None
            String representation of Series if ``buf=None``, otherwise None.
        """
        ...
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
    ) -> str | None:
        """
        Print {klass} in Markdown-friendly format.

        .. versionadded:: 1.0.0

        Parameters
        ----------
        buf : str, Path or StringIO-like, optional, default None
            Buffer to write to. If None, the output is returned as a string.
        mode : str, optional
            Mode in which file is opened, "wt" by default.
        index : bool, optional, default True
            Add index (row) labels.

            .. versionadded:: 1.1.0
        {storage_options}

            .. versionadded:: 1.2.0

        **kwargs
            These parameters will be passed to `tabulate \
                <https://pypi.org/project/tabulate>`_.

        Returns
        -------
        str
            {klass} in Markdown-friendly format.

        Notes
        -----
        Requires the `tabulate <https://pypi.org/project/tabulate>`_ package.

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

        Output markdown with a tabulate option.

        >>> print(s.to_markdown(tablefmt="grid"))
        +----+----------+
        |    | animal   |
        +====+==========+
        |  0 | elk      |
        +----+----------+
        |  1 | pig      |
        +----+----------+
        |  2 | dog      |
        +----+----------+
        |  3 | quetzal  |
        +----+----------+
        """
        ...
    def items(self) -> Iterable[tuple[Hashable, Any]]:
        """
        Lazily iterate over (index, value) tuples.

        This method returns an iterable tuple (index, value). This is
        convenient if you want to create a lazy iterator.

        Returns
        -------
        iterable
            Iterable of tuples containing the (index, value) pairs from a
            Series.

        See Also
        --------
        DataFrame.items : Iterate over (column name, Series) pairs.
        DataFrame.iterrows : Iterate over DataFrame rows as (index, Series) pairs.

        Examples
        --------
        >>> s = pd.Series(['A', 'B', 'C'])
        >>> for index, value in s.items():
        ...     print(f"Index : {index}, Value : {value}")
        Index : 0, Value : A
        Index : 1, Value : B
        Index : 2, Value : C
        """
        ...
    @Appender(items.__doc__)
    def iteritems(self) -> Iterable[tuple[Hashable, Any]]: ...
    def keys(self) -> Index:
        """
        Return alias for index.

        Returns
        -------
        Index
            Index of the Series.
        """
        ...
    def to_dict(
        self, into=...
    ):  # -> defaultdict[Unknown, Unknown] | Mapping[Unknown, Unknown]:
        """
        Convert Series to {label -> value} dict or dict-like object.

        Parameters
        ----------
        into : class, default dict
            The collections.abc.Mapping subclass to use as the return
            object. Can be the actual class or an empty
            instance of the mapping type you want.  If you want a
            collections.defaultdict, you must pass it initialized.

        Returns
        -------
        collections.abc.Mapping
            Key-value representation of Series.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.to_dict()
        {0: 1, 1: 2, 2: 3, 3: 4}
        >>> from collections import OrderedDict, defaultdict
        >>> s.to_dict(OrderedDict)
        OrderedDict([(0, 1), (1, 2), (2, 3), (3, 4)])
        >>> dd = defaultdict(list)
        >>> s.to_dict(dd)
        defaultdict(<class 'list'>, {0: 1, 1: 2, 2: 3, 3: 4})
        """
        ...
    def to_frame(self, name=...) -> DataFrame:
        """
        Convert Series to DataFrame.

        Parameters
        ----------
        name : object, default None
            The passed name should substitute for the series name (if it has
            one).

        Returns
        -------
        DataFrame
            DataFrame representation of Series.

        Examples
        --------
        >>> s = pd.Series(["a", "b", "c"],
        ...               name="vals")
        >>> s.to_frame()
          vals
        0    a
        1    b
        2    c
        """
        ...
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
    def count(self, level=...):  # -> Any | Series:
        """
        Return number of non-NA/null observations in the Series.

        Parameters
        ----------
        level : int or level name, default None
            If the axis is a MultiIndex (hierarchical), count along a
            particular level, collapsing into a smaller Series.

        Returns
        -------
        int or Series (if level specified)
            Number of non-null values in the Series.

        See Also
        --------
        DataFrame.count : Count non-NA cells for each column or row.

        Examples
        --------
        >>> s = pd.Series([0.0, 1.0, np.nan])
        >>> s.count()
        2
        """
        ...
    def mode(self, dropna=...) -> Series:
        """
        Return the mode(s) of the Series.

        The mode is the value that appears most often. There can be multiple modes.

        Always returns Series even if only one value is returned.

        Parameters
        ----------
        dropna : bool, default True
            Don't consider counts of NaN/NaT.

        Returns
        -------
        Series
            Modes of the Series in sorted order.
        """
        ...
    def unique(self) -> ArrayLike:
        """
        Return unique values of Series object.

        Uniques are returned in order of appearance. Hash table-based unique,
        therefore does NOT sort.

        Returns
        -------
        ndarray or ExtensionArray
            The unique values returned as a NumPy array. See Notes.

        See Also
        --------
        unique : Top-level unique method for any 1-d array-like object.
        Index.unique : Return Index with unique values from an Index object.

        Notes
        -----
        Returns the unique values as a NumPy array. In case of an
        extension-array backed Series, a new
        :class:`~api.extensions.ExtensionArray` of that type with just
        the unique values is returned. This includes

            * Categorical
            * Period
            * Datetime with Timezone
            * Interval
            * Sparse
            * IntegerNA

        See Examples section.

        Examples
        --------
        >>> pd.Series([2, 1, 3, 3], name='A').unique()
        array([2, 1, 3])

        >>> pd.Series([pd.Timestamp('2016-01-01') for _ in range(3)]).unique()
        array(['2016-01-01T00:00:00.000000000'], dtype='datetime64[ns]')

        >>> pd.Series([pd.Timestamp('2016-01-01', tz='US/Eastern')
        ...            for _ in range(3)]).unique()
        <DatetimeArray>
        ['2016-01-01 00:00:00-05:00']
        Length: 1, dtype: datetime64[ns, US/Eastern]

        An Categorical will return categories in the order of
        appearance and with the same dtype.

        >>> pd.Series(pd.Categorical(list('baabc'))).unique()
        ['b', 'a', 'c']
        Categories (3, object): ['a', 'b', 'c']
        >>> pd.Series(pd.Categorical(list('baabc'), categories=list('abc'),
        ...                          ordered=True)).unique()
        ['b', 'a', 'c']
        Categories (3, object): ['a' < 'b' < 'c']
        """
        ...
    @overload
    def drop_duplicates(self, keep=..., inplace: Literal[False] = ...) -> Series: ...
    @overload
    def drop_duplicates(self, keep, inplace: Literal[True]) -> None: ...
    @overload
    def drop_duplicates(self, *, inplace: Literal[True]) -> None: ...
    @overload
    def drop_duplicates(self, keep=..., inplace: bool = ...) -> Series | None: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self"])
    def drop_duplicates(self, keep=..., inplace=...) -> Series | None:
        """
        Return Series with duplicate values removed.

        Parameters
        ----------
        keep : {'first', 'last', ``False``}, default 'first'
            Method to handle dropping duplicates:

            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - ``False`` : Drop all duplicates.

        inplace : bool, default ``False``
            If ``True``, performs operation inplace and returns None.

        Returns
        -------
        Series or None
            Series with duplicates dropped or None if ``inplace=True``.

        See Also
        --------
        Index.drop_duplicates : Equivalent method on Index.
        DataFrame.drop_duplicates : Equivalent method on DataFrame.
        Series.duplicated : Related method on Series, indicating duplicate
            Series values.

        Examples
        --------
        Generate a Series with duplicated entries.

        >>> s = pd.Series(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'],
        ...               name='animal')
        >>> s
        0      lama
        1       cow
        2      lama
        3    beetle
        4      lama
        5     hippo
        Name: animal, dtype: object

        With the 'keep' parameter, the selection behaviour of duplicated values
        can be changed. The value 'first' keeps the first occurrence for each
        set of duplicated entries. The default value of keep is 'first'.

        >>> s.drop_duplicates()
        0      lama
        1       cow
        3    beetle
        5     hippo
        Name: animal, dtype: object

        The value 'last' for parameter 'keep' keeps the last occurrence for
        each set of duplicated entries.

        >>> s.drop_duplicates(keep='last')
        1       cow
        3    beetle
        4      lama
        5     hippo
        Name: animal, dtype: object

        The value ``False`` for parameter 'keep' discards all sets of
        duplicated entries. Setting the value of 'inplace' to ``True`` performs
        the operation inplace and returns ``None``.

        >>> s.drop_duplicates(keep=False, inplace=True)
        >>> s
        1       cow
        3    beetle
        5     hippo
        Name: animal, dtype: object
        """
        ...
    def duplicated(self, keep=...) -> Series:
        """
        Indicate duplicate Series values.

        Duplicated values are indicated as ``True`` values in the resulting
        Series. Either all duplicates, all except the first or all except the
        last occurrence of duplicates can be indicated.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            Method to handle dropping duplicates:

            - 'first' : Mark duplicates as ``True`` except for the first
              occurrence.
            - 'last' : Mark duplicates as ``True`` except for the last
              occurrence.
            - ``False`` : Mark all duplicates as ``True``.

        Returns
        -------
        Series[bool]
            Series indicating whether each value has occurred in the
            preceding values.

        See Also
        --------
        Index.duplicated : Equivalent method on pandas.Index.
        DataFrame.duplicated : Equivalent method on pandas.DataFrame.
        Series.drop_duplicates : Remove duplicate values from Series.

        Examples
        --------
        By default, for each set of duplicated values, the first occurrence is
        set on False and all others on True:

        >>> animals = pd.Series(['lama', 'cow', 'lama', 'beetle', 'lama'])
        >>> animals.duplicated()
        0    False
        1    False
        2     True
        3    False
        4     True
        dtype: bool

        which is equivalent to

        >>> animals.duplicated(keep='first')
        0    False
        1    False
        2     True
        3    False
        4     True
        dtype: bool

        By using 'last', the last occurrence of each set of duplicated values
        is set on False and all others on True:

        >>> animals.duplicated(keep='last')
        0     True
        1    False
        2     True
        3    False
        4    False
        dtype: bool

        By setting keep on ``False``, all duplicates are True:

        >>> animals.duplicated(keep=False)
        0     True
        1    False
        2     True
        3    False
        4     True
        dtype: bool
        """
        ...
    def idxmin(
        self, axis=..., skipna=..., *args, **kwargs
    ):  # -> float | ExtensionArray | Any | Index:
        """
        Return the row label of the minimum value.

        If multiple values equal the minimum, the first row label with that
        value is returned.

        Parameters
        ----------
        axis : int, default 0
            For compatibility with DataFrame.idxmin. Redundant for application
            on Series.
        skipna : bool, default True
            Exclude NA/null values. If the entire Series is NA, the result
            will be NA.
        *args, **kwargs
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        Index
            Label of the minimum value.

        Raises
        ------
        ValueError
            If the Series is empty.

        See Also
        --------
        numpy.argmin : Return indices of the minimum values
            along the given axis.
        DataFrame.idxmin : Return index of first occurrence of minimum
            over requested axis.
        Series.idxmax : Return index *label* of the first occurrence
            of maximum of values.

        Notes
        -----
        This method is the Series version of ``ndarray.argmin``. This method
        returns the label of the minimum, while ``ndarray.argmin`` returns
        the position. To get the position, use ``series.values.argmin()``.

        Examples
        --------
        >>> s = pd.Series(data=[1, None, 4, 1],
        ...               index=['A', 'B', 'C', 'D'])
        >>> s
        A    1.0
        B    NaN
        C    4.0
        D    1.0
        dtype: float64

        >>> s.idxmin()
        'A'

        If `skipna` is False and there is an NA value in the data,
        the function returns ``nan``.

        >>> s.idxmin(skipna=False)
        nan
        """
        ...
    def idxmax(
        self, axis=..., skipna=..., *args, **kwargs
    ):  # -> float | ExtensionArray | Any | Index:
        """
        Return the row label of the maximum value.

        If multiple values equal the maximum, the first row label with that
        value is returned.

        Parameters
        ----------
        axis : int, default 0
            For compatibility with DataFrame.idxmax. Redundant for application
            on Series.
        skipna : bool, default True
            Exclude NA/null values. If the entire Series is NA, the result
            will be NA.
        *args, **kwargs
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        Index
            Label of the maximum value.

        Raises
        ------
        ValueError
            If the Series is empty.

        See Also
        --------
        numpy.argmax : Return indices of the maximum values
            along the given axis.
        DataFrame.idxmax : Return index of first occurrence of maximum
            over requested axis.
        Series.idxmin : Return index *label* of the first occurrence
            of minimum of values.

        Notes
        -----
        This method is the Series version of ``ndarray.argmax``. This method
        returns the label of the maximum, while ``ndarray.argmax`` returns
        the position. To get the position, use ``series.values.argmax()``.

        Examples
        --------
        >>> s = pd.Series(data=[1, None, 4, 3, 4],
        ...               index=['A', 'B', 'C', 'D', 'E'])
        >>> s
        A    1.0
        B    NaN
        C    4.0
        D    3.0
        E    4.0
        dtype: float64

        >>> s.idxmax()
        'C'

        If `skipna` is False and there is an NA value in the data,
        the function returns ``nan``.

        >>> s.idxmax(skipna=False)
        nan
        """
        ...
    def round(self, decimals=..., *args, **kwargs) -> Series:
        """
        Round each value in a Series to the given number of decimals.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point.
        *args, **kwargs
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        Series
            Rounded values of the Series.

        See Also
        --------
        numpy.around : Round values of an np.array.
        DataFrame.round : Round values of a DataFrame.

        Examples
        --------
        >>> s = pd.Series([0.1, 1.3, 2.7])
        >>> s.round()
        0    0.0
        1    1.0
        2    3.0
        dtype: float64
        """
        ...
    def quantile(self, q=..., interpolation=...):  # -> Series:
        """
        Return value at the given quantile.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            The quantile(s) to compute, which can lie in range: 0 <= q <= 1.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points `i` and `j`:

                * linear: `i + (j - i) * fraction`, where `fraction` is the
                  fractional part of the index surrounded by `i` and `j`.
                * lower: `i`.
                * higher: `j`.
                * nearest: `i` or `j` whichever is nearest.
                * midpoint: (`i` + `j`) / 2.

        Returns
        -------
        float or Series
            If ``q`` is an array, a Series will be returned where the
            index is ``q`` and the values are the quantiles, otherwise
            a float will be returned.

        See Also
        --------
        core.window.Rolling.quantile : Calculate the rolling quantile.
        numpy.percentile : Returns the q-th percentile(s) of the array elements.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.quantile(.5)
        2.5
        >>> s.quantile([.25, .5, .75])
        0.25    1.75
        0.50    2.50
        0.75    3.25
        dtype: float64
        """
        ...
    def corr(self, other, method=..., min_periods=...) -> float:
        """
        Compute correlation with `other` Series, excluding missing values.

        Parameters
        ----------
        other : Series
            Series with which to compute the correlation.
        method : {'pearson', 'kendall', 'spearman'} or callable
            Method used to compute correlation:

            - pearson : Standard correlation coefficient
            - kendall : Kendall Tau correlation coefficient
            - spearman : Spearman rank correlation
            - callable: Callable with input two 1d ndarrays and returning a float.

            .. warning::
                Note that the returned matrix from corr will have 1 along the
                diagonals and will be symmetric regardless of the callable's
                behavior.
        min_periods : int, optional
            Minimum number of observations needed to have a valid result.

        Returns
        -------
        float
            Correlation with other.

        See Also
        --------
        DataFrame.corr : Compute pairwise correlation between columns.
        DataFrame.corrwith : Compute pairwise correlation with another
            DataFrame or Series.

        Examples
        --------
        >>> def histogram_intersection(a, b):
        ...     v = np.minimum(a, b).sum().round(decimals=1)
        ...     return v
        >>> s1 = pd.Series([.2, .0, .6, .2])
        >>> s2 = pd.Series([.3, .6, .0, .1])
        >>> s1.corr(s2, method=histogram_intersection)
        0.3
        """
        ...
    def cov(
        self, other: Series, min_periods: int | None = ..., ddof: int | None = ...
    ) -> float:
        """
        Compute covariance with Series, excluding missing values.

        Parameters
        ----------
        other : Series
            Series with which to compute the covariance.
        min_periods : int, optional
            Minimum number of observations needed to have a valid result.
        ddof : int, default 1
            Delta degrees of freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.

            .. versionadded:: 1.1.0

        Returns
        -------
        float
            Covariance between Series and other normalized by N-1
            (unbiased estimator).

        See Also
        --------
        DataFrame.cov : Compute pairwise covariance of columns.

        Examples
        --------
        >>> s1 = pd.Series([0.90010907, 0.13484424, 0.62036035])
        >>> s2 = pd.Series([0.12528585, 0.26962463, 0.51111198])
        >>> s1.cov(s2)
        -0.01685762652715874
        """
        ...
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
    def diff(self, periods: int = ...) -> Series:
        """
        First discrete difference of element.

        Calculates the difference of a {klass} element compared with another
        element in the {klass} (default is element in previous row).

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference, accepts negative
            values.
        {extra_params}
        Returns
        -------
        {klass}
            First differences of the Series.

        See Also
        --------
        {klass}.pct_change: Percent change over given number of periods.
        {klass}.shift: Shift index by desired number of periods with an
            optional time freq.
        {other_klass}.diff: First discrete difference of object.

        Notes
        -----
        For boolean dtypes, this uses :meth:`operator.xor` rather than
        :meth:`operator.sub`.
        The result is calculated according to current dtype in {klass},
        however dtype of the result is always float64.

        Examples
        --------
        {examples}
        """
        ...
    def autocorr(self, lag=...) -> float:
        """
        Compute the lag-N autocorrelation.

        This method computes the Pearson correlation between
        the Series and its shifted self.

        Parameters
        ----------
        lag : int, default 1
            Number of lags to apply before performing autocorrelation.

        Returns
        -------
        float
            The Pearson correlation between self and self.shift(lag).

        See Also
        --------
        Series.corr : Compute the correlation between two Series.
        Series.shift : Shift index by desired number of periods.
        DataFrame.corr : Compute pairwise correlation of columns.
        DataFrame.corrwith : Compute pairwise correlation between rows or
            columns of two DataFrame objects.

        Notes
        -----
        If the Pearson correlation is not well defined return 'NaN'.

        Examples
        --------
        >>> s = pd.Series([0.25, 0.5, 0.2, -0.05])
        >>> s.autocorr()  # doctest: +ELLIPSIS
        0.10355...
        >>> s.autocorr(lag=2)  # doctest: +ELLIPSIS
        -0.99999...

        If the Pearson correlation is not well defined, then 'NaN' is returned.

        >>> s = pd.Series([1, 0, 0, 0])
        >>> s.autocorr()
        nan
        """
        ...
    def dot(self, other):  # -> Series:
        """
        Compute the dot product between the Series and the columns of other.

        This method computes the dot product between the Series and another
        one, or the Series and each columns of a DataFrame, or the Series and
        each columns of an array.

        It can also be called using `self @ other` in Python >= 3.5.

        Parameters
        ----------
        other : Series, DataFrame or array-like
            The other object to compute the dot product with its columns.

        Returns
        -------
        scalar, Series or numpy.ndarray
            Return the dot product of the Series and other if other is a
            Series, the Series of the dot product of Series and each rows of
            other if other is a DataFrame or a numpy.ndarray between the Series
            and each columns of the numpy array.

        See Also
        --------
        DataFrame.dot: Compute the matrix product with the DataFrame.
        Series.mul: Multiplication of series and other, element-wise.

        Notes
        -----
        The Series and other has to share the same index if other is a Series
        or a DataFrame.

        Examples
        --------
        >>> s = pd.Series([0, 1, 2, 3])
        >>> other = pd.Series([-1, 2, -3, 4])
        >>> s.dot(other)
        8
        >>> s @ other
        8
        >>> df = pd.DataFrame([[0, 1], [-2, 3], [4, -5], [6, 7]])
        >>> s.dot(df)
        0    24
        1    14
        dtype: int64
        >>> arr = np.array([[0, 1], [-2, 3], [4, -5], [6, 7]])
        >>> s.dot(arr)
        array([24, 14])
        """
        ...
    def __matmul__(self, other):  # -> Series:
        """
        Matrix multiplication using binary `@` operator in Python>=3.5.
        """
        ...
    def __rmatmul__(self, other):  # -> Series:
        """
        Matrix multiplication using binary `@` operator in Python>=3.5.
        """
        ...
    @doc(base.IndexOpsMixin.searchsorted, klass="Series")
    def searchsorted(self, value, side=..., sorter=...) -> np.ndarray: ...
    def append(
        self, to_append, ignore_index: bool = ..., verify_integrity: bool = ...
    ):  # -> FrameOrSeriesUnion:
        """
        Concatenate two or more Series.

        Parameters
        ----------
        to_append : Series or list/tuple of Series
            Series to append with self.
        ignore_index : bool, default False
            If True, the resulting axis will be labeled 0, 1, …, n - 1.
        verify_integrity : bool, default False
            If True, raise Exception on creating index with duplicates.

        Returns
        -------
        Series
            Concatenated Series.

        See Also
        --------
        concat : General function to concatenate DataFrame or Series objects.

        Notes
        -----
        Iteratively appending to a Series can be more computationally intensive
        than a single concatenate. A better solution is to append values to a
        list and then concatenate the list with the original Series all at
        once.

        Examples
        --------
        >>> s1 = pd.Series([1, 2, 3])
        >>> s2 = pd.Series([4, 5, 6])
        >>> s3 = pd.Series([4, 5, 6], index=[3, 4, 5])
        >>> s1.append(s2)
        0    1
        1    2
        2    3
        0    4
        1    5
        2    6
        dtype: int64

        >>> s1.append(s3)
        0    1
        1    2
        2    3
        3    4
        4    5
        5    6
        dtype: int64

        With `ignore_index` set to True:

        >>> s1.append(s2, ignore_index=True)
        0    1
        1    2
        2    3
        3    4
        4    5
        5    6
        dtype: int64

        With `verify_integrity` set to True:

        >>> s1.append(s2, verify_integrity=True)
        Traceback (most recent call last):
        ...
        ValueError: Indexes have overlapping values: [0, 1, 2]
        """
        ...
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
    def combine(self, other, func, fill_value=...) -> Series:
        """
        Combine the Series with a Series or scalar according to `func`.

        Combine the Series and `other` using `func` to perform elementwise
        selection for combined Series.
        `fill_value` is assumed when value is missing at some index
        from one of the two objects being combined.

        Parameters
        ----------
        other : Series or scalar
            The value(s) to be combined with the `Series`.
        func : function
            Function that takes two scalars as inputs and returns an element.
        fill_value : scalar, optional
            The value to assume when an index is missing from
            one Series or the other. The default specifies to use the
            appropriate NaN value for the underlying dtype of the Series.

        Returns
        -------
        Series
            The result of combining the Series with the other object.

        See Also
        --------
        Series.combine_first : Combine Series values, choosing the calling
            Series' values first.

        Examples
        --------
        Consider 2 Datasets ``s1`` and ``s2`` containing
        highest clocked speeds of different birds.

        >>> s1 = pd.Series({'falcon': 330.0, 'eagle': 160.0})
        >>> s1
        falcon    330.0
        eagle     160.0
        dtype: float64
        >>> s2 = pd.Series({'falcon': 345.0, 'eagle': 200.0, 'duck': 30.0})
        >>> s2
        falcon    345.0
        eagle     200.0
        duck       30.0
        dtype: float64

        Now, to combine the two datasets and view the highest speeds
        of the birds across the two datasets

        >>> s1.combine(s2, max)
        duck        NaN
        eagle     200.0
        falcon    345.0
        dtype: float64

        In the previous example, the resulting value for duck is missing,
        because the maximum of a NaN and a float is a NaN.
        So, in the example, we set ``fill_value=0``,
        so the maximum value returned will be the value from some dataset.

        >>> s1.combine(s2, max, fill_value=0)
        duck       30.0
        eagle     200.0
        falcon    345.0
        dtype: float64
        """
        ...
    def combine_first(self, other) -> Series:
        """
        Update null elements with value in the same location in 'other'.

        Combine two Series objects by filling null values in one Series with
        non-null values from the other Series. Result index will be the union
        of the two indexes.

        Parameters
        ----------
        other : Series
            The value(s) to be used for filling null values.

        Returns
        -------
        Series
            The result of combining the provided Series with the other object.

        See Also
        --------
        Series.combine : Perform element-wise operation on two Series
            using a given function.

        Examples
        --------
        >>> s1 = pd.Series([1, np.nan])
        >>> s2 = pd.Series([3, 4, 5])
        >>> s1.combine_first(s2)
        0    1.0
        1    4.0
        2    5.0
        dtype: float64

        Null values still persist if the location of that null value
        does not exist in `other`

        >>> s1 = pd.Series({'falcon': np.nan, 'eagle': 160.0})
        >>> s2 = pd.Series({'eagle': 200.0, 'duck': 30.0})
        >>> s1.combine_first(s2)
        duck       30.0
        eagle     160.0
        falcon      NaN
        dtype: float64
        """
        ...
    def update(self, other) -> None:
        """
        Modify Series in place using values from passed Series.

        Uses non-NA values from passed Series to make updates. Aligns
        on index.

        Parameters
        ----------
        other : Series, or object coercible into Series

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.update(pd.Series([4, 5, 6]))
        >>> s
        0    4
        1    5
        2    6
        dtype: int64

        >>> s = pd.Series(['a', 'b', 'c'])
        >>> s.update(pd.Series(['d', 'e'], index=[0, 2]))
        >>> s
        0    d
        1    b
        2    e
        dtype: object

        >>> s = pd.Series([1, 2, 3])
        >>> s.update(pd.Series([4, 5, 6, 7, 8]))
        >>> s
        0    4
        1    5
        2    6
        dtype: int64

        If ``other`` contains NaNs the corresponding values are not updated
        in the original Series.

        >>> s = pd.Series([1, 2, 3])
        >>> s.update(pd.Series([4, np.nan, 6]))
        >>> s
        0    4
        1    2
        2    6
        dtype: int64

        ``other`` can also be a non-Series object type
        that is coercible into a Series

        >>> s = pd.Series([1, 2, 3])
        >>> s.update([4, np.nan, 6])
        >>> s
        0    4
        1    2
        2    6
        dtype: int64

        >>> s = pd.Series([1, 2, 3])
        >>> s.update({1: 9})
        >>> s
        0    1
        1    9
        2    3
        dtype: int64
        """
        ...
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
    ):  # -> Series | None:
        """
        Sort by the values.

        Sort a Series in ascending or descending order by some
        criterion.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Axis to direct sorting. The value 'index' is accepted for
            compatibility with DataFrame.sort_values.
        ascending : bool or list of bools, default True
            If True, sort values in ascending order, otherwise descending.
        inplace : bool, default False
            If True, perform operation in-place.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort'
            Choice of sorting algorithm. See also :func:`numpy.sort` for more
            information. 'mergesort' and 'stable' are the only stable  algorithms.
        na_position : {'first' or 'last'}, default 'last'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
            the end.
        ignore_index : bool, default False
            If True, the resulting axis will be labeled 0, 1, …, n - 1.

            .. versionadded:: 1.0.0

        key : callable, optional
            If not None, apply the key function to the series values
            before sorting. This is similar to the `key` argument in the
            builtin :meth:`sorted` function, with the notable difference that
            this `key` function should be *vectorized*. It should expect a
            ``Series`` and return an array-like.

            .. versionadded:: 1.1.0

        Returns
        -------
        Series or None
            Series ordered by values or None if ``inplace=True``.

        See Also
        --------
        Series.sort_index : Sort by the Series indices.
        DataFrame.sort_values : Sort DataFrame by the values along either axis.
        DataFrame.sort_index : Sort DataFrame by indices.

        Examples
        --------
        >>> s = pd.Series([np.nan, 1, 3, 10, 5])
        >>> s
        0     NaN
        1     1.0
        2     3.0
        3     10.0
        4     5.0
        dtype: float64

        Sort values ascending order (default behaviour)

        >>> s.sort_values(ascending=True)
        1     1.0
        2     3.0
        4     5.0
        3    10.0
        0     NaN
        dtype: float64

        Sort values descending order

        >>> s.sort_values(ascending=False)
        3    10.0
        4     5.0
        2     3.0
        1     1.0
        0     NaN
        dtype: float64

        Sort values inplace

        >>> s.sort_values(ascending=False, inplace=True)
        >>> s
        3    10.0
        4     5.0
        2     3.0
        1     1.0
        0     NaN
        dtype: float64

        Sort values putting NAs first

        >>> s.sort_values(na_position='first')
        0     NaN
        1     1.0
        2     3.0
        4     5.0
        3    10.0
        dtype: float64

        Sort a series of strings

        >>> s = pd.Series(['z', 'b', 'd', 'a', 'c'])
        >>> s
        0    z
        1    b
        2    d
        3    a
        4    c
        dtype: object

        >>> s.sort_values()
        3    a
        1    b
        4    c
        2    d
        0    z
        dtype: object

        Sort using a key function. Your `key` function will be
        given the ``Series`` of values and should return an array-like.

        >>> s = pd.Series(['a', 'B', 'c', 'D', 'e'])
        >>> s.sort_values()
        1    B
        3    D
        0    a
        2    c
        4    e
        dtype: object
        >>> s.sort_values(key=lambda x: x.str.lower())
        0    a
        1    B
        2    c
        3    D
        4    e
        dtype: object

        NumPy ufuncs work well here. For example, we can
        sort by the ``sin`` of the value

        >>> s = pd.Series([-4, -2, 0, 2, 4])
        >>> s.sort_values(key=np.sin)
        1   -2
        4    4
        2    0
        0   -4
        3    2
        dtype: int64

        More complicated user-defined functions can be used,
        as long as they expect a Series and return an array-like

        >>> s.sort_values(key=lambda x: (np.tan(x.cumsum())))
        0   -4
        3    2
        4    4
        1   -2
        2    0
        dtype: int64
        """
        ...
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
    ):  # -> Series | DataFrame | None:
        """
        Sort Series by index labels.

        Returns a new Series sorted by label if `inplace` argument is
        ``False``, otherwise updates the original series and returns None.

        Parameters
        ----------
        axis : int, default 0
            Axis to direct sorting. This can only be 0 for Series.
        level : int, optional
            If not None, sort on values in specified index level(s).
        ascending : bool or list-like of bools, default True
            Sort ascending vs. descending. When the index is a MultiIndex the
            sort direction can be controlled for each level individually.
        inplace : bool, default False
            If True, perform operation in-place.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort'
            Choice of sorting algorithm. See also :func:`numpy.sort` for more
            information. 'mergesort' and 'stable' are the only stable algorithms. For
            DataFrames, this option is only applied when sorting on a single
            column or label.
        na_position : {'first', 'last'}, default 'last'
            If 'first' puts NaNs at the beginning, 'last' puts NaNs at the end.
            Not implemented for MultiIndex.
        sort_remaining : bool, default True
            If True and sorting by level and index is multilevel, sort by other
            levels too (in order) after sorting by specified level.
        ignore_index : bool, default False
            If True, the resulting axis will be labeled 0, 1, …, n - 1.

            .. versionadded:: 1.0.0

        key : callable, optional
            If not None, apply the key function to the index values
            before sorting. This is similar to the `key` argument in the
            builtin :meth:`sorted` function, with the notable difference that
            this `key` function should be *vectorized*. It should expect an
            ``Index`` and return an ``Index`` of the same shape.

            .. versionadded:: 1.1.0

        Returns
        -------
        Series or None
            The original Series sorted by the labels or None if ``inplace=True``.

        See Also
        --------
        DataFrame.sort_index: Sort DataFrame by the index.
        DataFrame.sort_values: Sort DataFrame by the value.
        Series.sort_values : Sort Series by the value.

        Examples
        --------
        >>> s = pd.Series(['a', 'b', 'c', 'd'], index=[3, 2, 1, 4])
        >>> s.sort_index()
        1    c
        2    b
        3    a
        4    d
        dtype: object

        Sort Descending

        >>> s.sort_index(ascending=False)
        4    d
        3    a
        2    b
        1    c
        dtype: object

        Sort Inplace

        >>> s.sort_index(inplace=True)
        >>> s
        1    c
        2    b
        3    a
        4    d
        dtype: object

        By default NaNs are put at the end, but use `na_position` to place
        them at the beginning

        >>> s = pd.Series(['a', 'b', 'c', 'd'], index=[3, 2, 1, np.nan])
        >>> s.sort_index(na_position='first')
        NaN     d
         1.0    c
         2.0    b
         3.0    a
        dtype: object

        Specify index level to sort

        >>> arrays = [np.array(['qux', 'qux', 'foo', 'foo',
        ...                     'baz', 'baz', 'bar', 'bar']),
        ...           np.array(['two', 'one', 'two', 'one',
        ...                     'two', 'one', 'two', 'one'])]
        >>> s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=arrays)
        >>> s.sort_index(level=1)
        bar  one    8
        baz  one    6
        foo  one    4
        qux  one    2
        bar  two    7
        baz  two    5
        foo  two    3
        qux  two    1
        dtype: int64

        Does not sort by remaining levels when sorting by levels

        >>> s.sort_index(level=1, sort_remaining=False)
        qux  one    2
        foo  one    4
        baz  one    6
        bar  one    8
        qux  two    1
        foo  two    3
        baz  two    5
        bar  two    7
        dtype: int64

        Apply a key function before sorting

        >>> s = pd.Series([1, 2, 3, 4], index=['A', 'b', 'C', 'd'])
        >>> s.sort_index(key=lambda x : x.str.lower())
        A    1
        b    2
        C    3
        d    4
        dtype: int64
        """
        ...
    def argsort(self, axis=..., kind=..., order=...) -> Series:
        """
        Return the integer indices that would sort the Series values.

        Override ndarray.argsort. Argsorts the value, omitting NA/null values,
        and places the result in the same locations as the non-NA values.

        Parameters
        ----------
        axis : {0 or "index"}
            Has no effect but is accepted for compatibility with numpy.
        kind : {'mergesort', 'quicksort', 'heapsort', 'stable'}, default 'quicksort'
            Choice of sorting algorithm. See :func:`numpy.sort` for more
            information. 'mergesort' and 'stable' are the only stable algorithms.
        order : None
            Has no effect but is accepted for compatibility with numpy.

        Returns
        -------
        Series[np.intp]
            Positions of values within the sort order with -1 indicating
            nan values.

        See Also
        --------
        numpy.ndarray.argsort : Returns the indices that would sort this array.
        """
        ...
    def nlargest(self, n=..., keep=...) -> Series:
        """
        Return the largest `n` elements.

        Parameters
        ----------
        n : int, default 5
            Return this many descending sorted values.
        keep : {'first', 'last', 'all'}, default 'first'
            When there are duplicate values that cannot all fit in a
            Series of `n` elements:

            - ``first`` : return the first `n` occurrences in order
                of appearance.
            - ``last`` : return the last `n` occurrences in reverse
                order of appearance.
            - ``all`` : keep all occurrences. This can result in a Series of
                size larger than `n`.

        Returns
        -------
        Series
            The `n` largest values in the Series, sorted in decreasing order.

        See Also
        --------
        Series.nsmallest: Get the `n` smallest elements.
        Series.sort_values: Sort Series by values.
        Series.head: Return the first `n` rows.

        Notes
        -----
        Faster than ``.sort_values(ascending=False).head(n)`` for small `n`
        relative to the size of the ``Series`` object.

        Examples
        --------
        >>> countries_population = {"Italy": 59000000, "France": 65000000,
        ...                         "Malta": 434000, "Maldives": 434000,
        ...                         "Brunei": 434000, "Iceland": 337000,
        ...                         "Nauru": 11300, "Tuvalu": 11300,
        ...                         "Anguilla": 11300, "Montserrat": 5200}
        >>> s = pd.Series(countries_population)
        >>> s
        Italy       59000000
        France      65000000
        Malta         434000
        Maldives      434000
        Brunei        434000
        Iceland       337000
        Nauru          11300
        Tuvalu         11300
        Anguilla       11300
        Montserrat      5200
        dtype: int64

        The `n` largest elements where ``n=5`` by default.

        >>> s.nlargest()
        France      65000000
        Italy       59000000
        Malta         434000
        Maldives      434000
        Brunei        434000
        dtype: int64

        The `n` largest elements where ``n=3``. Default `keep` value is 'first'
        so Malta will be kept.

        >>> s.nlargest(3)
        France    65000000
        Italy     59000000
        Malta       434000
        dtype: int64

        The `n` largest elements where ``n=3`` and keeping the last duplicates.
        Brunei will be kept since it is the last with value 434000 based on
        the index order.

        >>> s.nlargest(3, keep='last')
        France      65000000
        Italy       59000000
        Brunei        434000
        dtype: int64

        The `n` largest elements where ``n=3`` with all duplicates kept. Note
        that the returned Series has five elements due to the three duplicates.

        >>> s.nlargest(3, keep='all')
        France      65000000
        Italy       59000000
        Malta         434000
        Maldives      434000
        Brunei        434000
        dtype: int64
        """
        ...
    def nsmallest(self, n: int = ..., keep: str = ...) -> Series:
        """
        Return the smallest `n` elements.

        Parameters
        ----------
        n : int, default 5
            Return this many ascending sorted values.
        keep : {'first', 'last', 'all'}, default 'first'
            When there are duplicate values that cannot all fit in a
            Series of `n` elements:

            - ``first`` : return the first `n` occurrences in order
                of appearance.
            - ``last`` : return the last `n` occurrences in reverse
                order of appearance.
            - ``all`` : keep all occurrences. This can result in a Series of
                size larger than `n`.

        Returns
        -------
        Series
            The `n` smallest values in the Series, sorted in increasing order.

        See Also
        --------
        Series.nlargest: Get the `n` largest elements.
        Series.sort_values: Sort Series by values.
        Series.head: Return the first `n` rows.

        Notes
        -----
        Faster than ``.sort_values().head(n)`` for small `n` relative to
        the size of the ``Series`` object.

        Examples
        --------
        >>> countries_population = {"Italy": 59000000, "France": 65000000,
        ...                         "Brunei": 434000, "Malta": 434000,
        ...                         "Maldives": 434000, "Iceland": 337000,
        ...                         "Nauru": 11300, "Tuvalu": 11300,
        ...                         "Anguilla": 11300, "Montserrat": 5200}
        >>> s = pd.Series(countries_population)
        >>> s
        Italy       59000000
        France      65000000
        Brunei        434000
        Malta         434000
        Maldives      434000
        Iceland       337000
        Nauru          11300
        Tuvalu         11300
        Anguilla       11300
        Montserrat      5200
        dtype: int64

        The `n` smallest elements where ``n=5`` by default.

        >>> s.nsmallest()
        Montserrat    5200
        Nauru        11300
        Tuvalu       11300
        Anguilla     11300
        Iceland     337000
        dtype: int64

        The `n` smallest elements where ``n=3``. Default `keep` value is
        'first' so Nauru and Tuvalu will be kept.

        >>> s.nsmallest(3)
        Montserrat   5200
        Nauru       11300
        Tuvalu      11300
        dtype: int64

        The `n` smallest elements where ``n=3`` and keeping the last
        duplicates. Anguilla and Tuvalu will be kept since they are the last
        with value 11300 based on the index order.

        >>> s.nsmallest(3, keep='last')
        Montserrat   5200
        Anguilla    11300
        Tuvalu      11300
        dtype: int64

        The `n` smallest elements where ``n=3`` with all duplicates kept. Note
        that the returned Series has four elements due to the three duplicates.

        >>> s.nsmallest(3, keep='all')
        Montserrat   5200
        Nauru       11300
        Tuvalu      11300
        Anguilla    11300
        dtype: int64
        """
        ...
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
    def swaplevel(self, i=..., j=..., copy=...) -> Series:
        """
        Swap levels i and j in a :class:`MultiIndex`.

        Default is to swap the two innermost levels of the index.

        Parameters
        ----------
        i, j : int or str
            Levels of the indices to be swapped. Can pass level name as string.
        {extra_params}

        Returns
        -------
        {klass}
            {klass} with levels swapped in MultiIndex.

        {examples}
        """
        ...
    def reorder_levels(self, order) -> Series:
        """
        Rearrange index levels using input order.

        May not drop or duplicate levels.

        Parameters
        ----------
        order : list of int representing new level order
            Reference level by number or key.

        Returns
        -------
        type of caller (new object)
        """
        ...
    def explode(self, ignore_index: bool = ...) -> Series:
        """
        Transform each element of a list-like to a row.

        .. versionadded:: 0.25.0

        Parameters
        ----------
        ignore_index : bool, default False
            If True, the resulting index will be labeled 0, 1, …, n - 1.

            .. versionadded:: 1.1.0

        Returns
        -------
        Series
            Exploded lists to rows; index will be duplicated for these rows.

        See Also
        --------
        Series.str.split : Split string values on specified separator.
        Series.unstack : Unstack, a.k.a. pivot, Series with MultiIndex
            to produce DataFrame.
        DataFrame.melt : Unpivot a DataFrame from wide format to long format.
        DataFrame.explode : Explode a DataFrame from list-like
            columns to long format.

        Notes
        -----
        This routine will explode list-likes including lists, tuples, sets,
        Series, and np.ndarray. The result dtype of the subset rows will
        be object. Scalars will be returned unchanged, and empty list-likes will
        result in a np.nan for that row. In addition, the ordering of elements in
        the output will be non-deterministic when exploding sets.

        Examples
        --------
        >>> s = pd.Series([[1, 2, 3], 'foo', [], [3, 4]])
        >>> s
        0    [1, 2, 3]
        1          foo
        2           []
        3       [3, 4]
        dtype: object

        >>> s.explode()
        0      1
        0      2
        0      3
        1    foo
        2    NaN
        3      3
        3      4
        dtype: object
        """
        ...
    def unstack(self, level=..., fill_value=...) -> DataFrame:
        """
        Unstack, also known as pivot, Series with MultiIndex to produce DataFrame.

        Parameters
        ----------
        level : int, str, or list of these, default last level
            Level(s) to unstack, can pass level name.
        fill_value : scalar value, default None
            Value to use when replacing NaN values.

        Returns
        -------
        DataFrame
            Unstacked Series.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4],
        ...               index=pd.MultiIndex.from_product([['one', 'two'],
        ...                                                 ['a', 'b']]))
        >>> s
        one  a    1
             b    2
        two  a    3
             b    4
        dtype: int64

        >>> s.unstack(level=-1)
             a  b
        one  1  2
        two  3  4

        >>> s.unstack(level=0)
           one  two
        a    1    3
        b    2    4
        """
        ...
    def map(self, arg, na_action=...) -> Series:
        """
        Map values of Series according to input correspondence.

        Used for substituting each value in a Series with another value,
        that may be derived from a function, a ``dict`` or
        a :class:`Series`.

        Parameters
        ----------
        arg : function, collections.abc.Mapping subclass or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NaN values, without passing them to the
            mapping correspondence.

        Returns
        -------
        Series
            Same index as caller.

        See Also
        --------
        Series.apply : For applying more complex functions on a Series.
        DataFrame.apply : Apply a function row-/column-wise.
        DataFrame.applymap : Apply a function elementwise on a whole DataFrame.

        Notes
        -----
        When ``arg`` is a dictionary, values in Series that are not in the
        dictionary (as keys) are converted to ``NaN``. However, if the
        dictionary is a ``dict`` subclass that defines ``__missing__`` (i.e.
        provides a method for default values), then this default is used
        rather than ``NaN``.

        Examples
        --------
        >>> s = pd.Series(['cat', 'dog', np.nan, 'rabbit'])
        >>> s
        0      cat
        1      dog
        2      NaN
        3   rabbit
        dtype: object

        ``map`` accepts a ``dict`` or a ``Series``. Values that are not found
        in the ``dict`` are converted to ``NaN``, unless the dict has a default
        value (e.g. ``defaultdict``):

        >>> s.map({'cat': 'kitten', 'dog': 'puppy'})
        0   kitten
        1    puppy
        2      NaN
        3      NaN
        dtype: object

        It also accepts a function:

        >>> s.map('I am a {}'.format)
        0       I am a cat
        1       I am a dog
        2       I am a nan
        3    I am a rabbit
        dtype: object

        To avoid applying the function to missing values (and keep them as
        ``NaN``) ``na_action='ignore'`` can be used:

        >>> s.map('I am a {}'.format, na_action='ignore')
        0     I am a cat
        1     I am a dog
        2            NaN
        3  I am a rabbit
        dtype: object
        """
        ...
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
    ) -> FrameOrSeriesUnion:
        """
        Invoke function on values of Series.

        Can be ufunc (a NumPy function that applies to the entire Series)
        or a Python function that only works on single values.

        Parameters
        ----------
        func : function
            Python function or NumPy ufunc to apply.
        convert_dtype : bool, default True
            Try to find better dtype for elementwise function results. If
            False, leave as dtype=object. Note that the dtype is always
            preserved for some extension array dtypes, such as Categorical.
        args : tuple
            Positional arguments passed to func after the series value.
        **kwargs
            Additional keyword arguments passed to func.

        Returns
        -------
        Series or DataFrame
            If func returns a Series object the result will be a DataFrame.

        See Also
        --------
        Series.map: For element-wise operations.
        Series.agg: Only perform aggregating type operations.
        Series.transform: Only perform transforming type operations.

        Notes
        -----
        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        Create a series with typical summer temperatures for each city.

        >>> s = pd.Series([20, 21, 12],
        ...               index=['London', 'New York', 'Helsinki'])
        >>> s
        London      20
        New York    21
        Helsinki    12
        dtype: int64

        Square the values by defining a function and passing it as an
        argument to ``apply()``.

        >>> def square(x):
        ...     return x ** 2
        >>> s.apply(square)
        London      400
        New York    441
        Helsinki    144
        dtype: int64

        Square the values by passing an anonymous function as an
        argument to ``apply()``.

        >>> s.apply(lambda x: x ** 2)
        London      400
        New York    441
        Helsinki    144
        dtype: int64

        Define a custom function that needs additional positional
        arguments and pass these additional arguments using the
        ``args`` keyword.

        >>> def subtract_custom_value(x, custom_value):
        ...     return x - custom_value

        >>> s.apply(subtract_custom_value, args=(5,))
        London      15
        New York    16
        Helsinki     7
        dtype: int64

        Define a custom function that takes keyword arguments
        and pass these arguments to ``apply``.

        >>> def add_custom_values(x, **kwargs):
        ...     for month in kwargs:
        ...         x += kwargs[month]
        ...     return x

        >>> s.apply(add_custom_values, june=30, july=20, august=25)
        London      95
        New York    96
        Helsinki    87
        dtype: int64

        Use a function from the Numpy library.

        >>> s.apply(np.log)
        London      2.995732
        New York    3.044522
        Helsinki    2.484907
        dtype: float64
        """
        ...
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
    ):  # -> Series | None:
        """
        Alter Series index labels or name.

        Function / dict values must be unique (1-to-1). Labels not contained in
        a dict / Series will be left as-is. Extra labels listed don't throw an
        error.

        Alternatively, change ``Series.name`` with a scalar value.

        See the :ref:`user guide <basics.rename>` for more.

        Parameters
        ----------
        axis : {0 or "index"}
            Unused. Accepted for compatibility with DataFrame method only.
        index : scalar, hashable sequence, dict-like or function, optional
            Functions or dict-like are transformations to apply to
            the index.
            Scalar or hashable sequence-like will alter the ``Series.name``
            attribute.

        **kwargs
            Additional keyword arguments passed to the function. Only the
            "inplace" keyword is used.

        Returns
        -------
        Series or None
            Series with index labels or name altered or None if ``inplace=True``.

        See Also
        --------
        DataFrame.rename : Corresponding DataFrame method.
        Series.rename_axis : Set the name of the axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s
        0    1
        1    2
        2    3
        dtype: int64
        >>> s.rename("my_name")  # scalar, changes Series.name
        0    1
        1    2
        2    3
        Name: my_name, dtype: int64
        >>> s.rename(lambda x: x ** 2)  # function, changes labels
        0    1
        1    2
        4    3
        dtype: int64
        >>> s.rename({1: 3, 2: 5})  # mapping, changes labels
        0    1
        3    2
        5    3
        dtype: int64
        """
        ...
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
    ) -> Series:
        """
        Return Series with specified index labels removed.

        Remove elements of a Series based on specifying the index labels.
        When using a multi-index, labels on different levels can be removed
        by specifying the level.

        Parameters
        ----------
        labels : single label or list-like
            Index labels to drop.
        axis : 0, default 0
            Redundant for application on Series.
        index : single label or list-like
            Redundant for application on Series, but 'index' can be used instead
            of 'labels'.
        columns : single label or list-like
            No change is made to the Series; use 'index' or 'labels' instead.
        level : int or level name, optional
            For MultiIndex, level for which the labels will be removed.
        inplace : bool, default False
            If True, do operation inplace and return None.
        errors : {'ignore', 'raise'}, default 'raise'
            If 'ignore', suppress error and only existing labels are dropped.

        Returns
        -------
        Series or None
            Series with specified index labels removed or None if ``inplace=True``.

        Raises
        ------
        KeyError
            If none of the labels are found in the index.

        See Also
        --------
        Series.reindex : Return only specified index labels of Series.
        Series.dropna : Return series without null values.
        Series.drop_duplicates : Return Series with duplicate values removed.
        DataFrame.drop : Drop specified labels from rows or columns.

        Examples
        --------
        >>> s = pd.Series(data=np.arange(3), index=['A', 'B', 'C'])
        >>> s
        A  0
        B  1
        C  2
        dtype: int64

        Drop labels B en C

        >>> s.drop(labels=['B', 'C'])
        A  0
        dtype: int64

        Drop 2nd level label in MultiIndex Series

        >>> midx = pd.MultiIndex(levels=[['lama', 'cow', 'falcon'],
        ...                              ['speed', 'weight', 'length']],
        ...                      codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                             [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        >>> s = pd.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
        ...               index=midx)
        >>> s
        lama    speed      45.0
                weight    200.0
                length      1.2
        cow     speed      30.0
                weight    250.0
                length      1.5
        falcon  speed     320.0
                weight      1.0
                length      0.3
        dtype: float64

        >>> s.drop(labels='weight', level=1)
        lama    speed      45.0
                length      1.2
        cow     speed      30.0
                length      1.5
        falcon  speed     320.0
                length      0.3
        dtype: float64
        """
        ...
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
        self, value, *, axis: Axis | None, inplace: Literal[True], limit=..., downcast=...
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
    def pop(self, item: Hashable) -> Any:
        """
        Return item and drops from series. Raise KeyError if not found.

        Parameters
        ----------
        item : label
            Index of the element that needs to be removed.

        Returns
        -------
        Value that is popped from series.

        Examples
        --------
        >>> ser = pd.Series([1,2,3])

        >>> ser.pop(0)
        1

        >>> ser
        1    2
        2    3
        dtype: int64
        """
        ...
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
    def memory_usage(self, index: bool = ..., deep: bool = ...) -> int:
        """
        Return the memory usage of the Series.

        The memory usage can optionally include the contribution of
        the index and of elements of `object` dtype.

        Parameters
        ----------
        index : bool, default True
            Specifies whether to include the memory usage of the Series index.
        deep : bool, default False
            If True, introspect the data deeply by interrogating
            `object` dtypes for system-level memory consumption, and include
            it in the returned value.

        Returns
        -------
        int
            Bytes of memory consumed.

        See Also
        --------
        numpy.ndarray.nbytes : Total bytes consumed by the elements of the
            array.
        DataFrame.memory_usage : Bytes consumed by a DataFrame.

        Examples
        --------
        >>> s = pd.Series(range(3))
        >>> s.memory_usage()
        152

        Not including the index gives the size of the rest of the data, which
        is necessarily smaller:

        >>> s.memory_usage(index=False)
        24

        The memory footprint of `object` values is ignored by default:

        >>> s = pd.Series(["a", "b"])
        >>> s.values
        array(['a', 'b'], dtype=object)
        >>> s.memory_usage()
        144
        >>> s.memory_usage(deep=True)
        244
        """
        ...
    def isin(self, values) -> Series:
        """
        Whether elements in Series are contained in `values`.

        Return a boolean Series showing whether each element in the Series
        matches an element in the passed sequence of `values` exactly.

        Parameters
        ----------
        values : set or list-like
            The sequence of values to test. Passing in a single string will
            raise a ``TypeError``. Instead, turn a single string into a
            list of one element.

        Returns
        -------
        Series
            Series of booleans indicating if each element is in values.

        Raises
        ------
        TypeError
          * If `values` is a string

        See Also
        --------
        DataFrame.isin : Equivalent method on DataFrame.

        Examples
        --------
        >>> s = pd.Series(['lama', 'cow', 'lama', 'beetle', 'lama',
        ...                'hippo'], name='animal')
        >>> s.isin(['cow', 'lama'])
        0     True
        1     True
        2     True
        3    False
        4     True
        5    False
        Name: animal, dtype: bool

        Passing a single string as ``s.isin('lama')`` will raise an error. Use
        a list of one element instead:

        >>> s.isin(['lama'])
        0     True
        1    False
        2     True
        3    False
        4     True
        5    False
        Name: animal, dtype: bool

        Strings and integers are distinct and are therefore not comparable:

        >>> pd.Series([1]).isin(['1'])
        0    False
        dtype: bool
        >>> pd.Series([1.1]).isin(['1.1'])
        0    False
        dtype: bool
        """
        ...
    def between(self, left, right, inclusive=...) -> Series:
        """
        Return boolean Series equivalent to left <= series <= right.

        This function returns a boolean vector containing `True` wherever the
        corresponding Series element is between the boundary values `left` and
        `right`. NA values are treated as `False`.

        Parameters
        ----------
        left : scalar or list-like
            Left boundary.
        right : scalar or list-like
            Right boundary.
        inclusive : {"both", "neither", "left", "right"}
            Include boundaries. Whether to set each bound as closed or open.

            .. versionchanged:: 1.3.0

        Returns
        -------
        Series
            Series representing whether each element is between left and
            right (inclusive).

        See Also
        --------
        Series.gt : Greater than of series and other.
        Series.lt : Less than of series and other.

        Notes
        -----
        This function is equivalent to ``(left <= ser) & (ser <= right)``

        Examples
        --------
        >>> s = pd.Series([2, 0, 4, 8, np.nan])

        Boundary values are included by default:

        >>> s.between(1, 4)
        0     True
        1    False
        2     True
        3    False
        4    False
        dtype: bool

        With `inclusive` set to ``"neither"`` boundary values are excluded:

        >>> s.between(1, 4, inclusive="neither")
        0     True
        1    False
        2    False
        3    False
        4    False
        dtype: bool

        `left` and `right` can be any scalar value:

        >>> s = pd.Series(['Alice', 'Bob', 'Carol', 'Eve'])
        >>> s.between('Anna', 'Daniel')
        0    False
        1     True
        2     True
        3    False
        dtype: bool
        """
        ...
    @doc(NDFrame.isna, klass=_shared_doc_kwargs["klass"])
    def isna(self) -> Series: ...
    @doc(NDFrame.isna, klass=_shared_doc_kwargs["klass"])
    def isnull(self) -> Series: ...
    @doc(NDFrame.notna, klass=_shared_doc_kwargs["klass"])
    def notna(self) -> Series: ...
    @doc(NDFrame.notna, klass=_shared_doc_kwargs["klass"])
    def notnull(self) -> Series: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self"])
    def dropna(self, axis=..., inplace=..., how=...):  # -> Self@Series | None:
        """
        Return a new Series with missing values removed.

        See the :ref:`User Guide <missing_data>` for more on which values are
        considered missing, and how to work with missing data.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            There is only one axis to drop values from.
        inplace : bool, default False
            If True, do operation inplace and return None.
        how : str, optional
            Not in use. Kept for compatibility.

        Returns
        -------
        Series or None
            Series with NA entries dropped from it or None if ``inplace=True``.

        See Also
        --------
        Series.isna: Indicate missing values.
        Series.notna : Indicate existing (non-missing) values.
        Series.fillna : Replace missing values.
        DataFrame.dropna : Drop rows or columns which contain NA values.
        Index.dropna : Drop missing indices.

        Examples
        --------
        >>> ser = pd.Series([1., 2., np.nan])
        >>> ser
        0    1.0
        1    2.0
        2    NaN
        dtype: float64

        Drop NA values from a Series.

        >>> ser.dropna()
        0    1.0
        1    2.0
        dtype: float64

        Keep the Series with valid entries in the same variable.

        >>> ser.dropna(inplace=True)
        >>> ser
        0    1.0
        1    2.0
        dtype: float64

        Empty strings are not considered NA values. ``None`` is considered an
        NA value.

        >>> ser = pd.Series([np.NaN, 2, pd.NaT, '', None, 'I stay'])
        >>> ser
        0       NaN
        1         2
        2       NaT
        3
        4      None
        5    I stay
        dtype: object
        >>> ser.dropna()
        1         2
        3
        5    I stay
        dtype: object
        """
        ...
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
    def to_timestamp(self, freq=..., how=..., copy=...) -> Series:
        """
        Cast to DatetimeIndex of Timestamps, at *beginning* of period.

        Parameters
        ----------
        freq : str, default frequency of PeriodIndex
            Desired frequency.
        how : {'s', 'e', 'start', 'end'}
            Convention for converting period to timestamp; start of period
            vs. end.
        copy : bool, default True
            Whether or not to return a copy.

        Returns
        -------
        Series with DatetimeIndex
        """
        ...
    def to_period(self, freq=..., copy=...) -> Series:
        """
        Convert Series from DatetimeIndex to PeriodIndex.

        Parameters
        ----------
        freq : str, default None
            Frequency associated with the PeriodIndex.
        copy : bool, default True
            Whether or not to return a copy.

        Returns
        -------
        Series
            Series with index converted to PeriodIndex.
        """
        ...
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
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "lower", "upper"])
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
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "cond", "other"])
    def where(
        self, cond, other=..., inplace=..., axis=..., level=..., errors=..., try_cast=...
    ): ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "cond", "other"])
    def mask(
        self, cond, other=..., inplace=..., axis=..., level=..., errors=..., try_cast=...
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
