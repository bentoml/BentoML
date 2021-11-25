from typing import TYPE_CHECKING, Any, Hashable

import numpy as np
from pandas import DataFrame, Index, MultiIndex, Series
from pandas._typing import ArrayLike, DtypeArg, FrameOrSeries, FrameOrSeriesUnion, Shape
from pandas.core.computation.pytables import PyTablesExpr
from pandas.util._decorators import cache_readonly
from tables import Col, File, Node

"""
High level interface to PyTables for reading and writing pandas data structures
to disk
"""
if TYPE_CHECKING: ...
_version = ...
_default_encoding = ...
Term = PyTablesExpr

class PossibleDataLossError(Exception): ...
class ClosedFileError(Exception): ...
class IncompatibilityWarning(Warning): ...

incompatibility_doc = ...

class AttributeConflictWarning(Warning): ...

attribute_conflict_doc = ...

class DuplicateWarning(Warning): ...

duplicate_doc = ...
performance_doc = ...
_FORMAT_MAP = ...
_AXES_MAP = ...
dropna_doc = ...
format_doc = ...
_table_mod = ...
_table_file_open_policy_is_strict = ...

def to_hdf(
    path_or_buf,
    key: str,
    value: FrameOrSeries,
    mode: str = ...,
    complevel: int | None = ...,
    complib: str | None = ...,
    append: bool = ...,
    format: str | None = ...,
    index: bool = ...,
    min_itemsize: int | dict[str, int] | None = ...,
    nan_rep=...,
    dropna: bool | None = ...,
    data_columns: bool | list[str] | None = ...,
    errors: str = ...,
    encoding: str = ...,
):  # -> None:
    """store this object, close it if we opened it"""
    ...

def read_hdf(
    path_or_buf,
    key=...,
    mode: str = ...,
    errors: str = ...,
    where=...,
    start: int | None = ...,
    stop: int | None = ...,
    columns=...,
    iterator=...,
    chunksize: int | None = ...,
    **kwargs
):  # -> TableIterator:
    """
    Read from the store, close it if we opened it.

    Retrieve pandas object stored in file, optionally based on where
    criteria.

    .. warning::

       Pandas uses PyTables for reading and writing HDF5 files, which allows
       serializing object-dtype data with pickle when using the "fixed" format.
       Loading pickled data received from untrusted sources can be unsafe.

       See: https://docs.python.org/3/library/pickle.html for more.

    Parameters
    ----------
    path_or_buf : str, path object, pandas.HDFStore
        Any valid string path is acceptable. Only supports the local file system,
        remote URLs and file-like objects are not supported.

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        Alternatively, pandas accepts an open :class:`pandas.HDFStore` object.

    key : object, optional
        The group identifier in the store. Can be omitted if the HDF file
        contains a single pandas object.
    mode : {'r', 'r+', 'a'}, default 'r'
        Mode to use when opening the file. Ignored if path_or_buf is a
        :class:`pandas.HDFStore`. Default is 'r'.
    errors : str, default 'strict'
        Specifies how encoding and decoding errors are to be handled.
        See the errors argument for :func:`open` for a full list
        of options.
    where : list, optional
        A list of Term (or convertible) objects.
    start : int, optional
        Row number to start selection.
    stop  : int, optional
        Row number to stop selection.
    columns : list, optional
        A list of columns names to return.
    iterator : bool, optional
        Return an iterator object.
    chunksize : int, optional
        Number of rows to include in an iteration when using an iterator.
    **kwargs
        Additional keyword arguments passed to HDFStore.

    Returns
    -------
    item : object
        The selected object. Return type depends on the object stored.

    See Also
    --------
    DataFrame.to_hdf : Write a HDF file from a DataFrame.
    HDFStore : Low-level access to HDF files.

    Examples
    --------
    >>> df = pd.DataFrame([[1, 1.0, 'a']], columns=['x', 'y', 'z'])
    >>> df.to_hdf('./store.h5', 'data')
    >>> reread = pd.read_hdf('./store.h5')
    """
    ...

class HDFStore:
    """
    Dict-like IO interface for storing pandas objects in PyTables.

    Either Fixed or Table format.

    .. warning::

       Pandas uses PyTables for reading and writing HDF5 files, which allows
       serializing object-dtype data with pickle when using the "fixed" format.
       Loading pickled data received from untrusted sources can be unsafe.

       See: https://docs.python.org/3/library/pickle.html for more.

    Parameters
    ----------
    path : str
        File path to HDF5 file.
    mode : {'a', 'w', 'r', 'r+'}, default 'a'

        ``'r'``
            Read-only; no data can be modified.
        ``'w'``
            Write; a new file is created (an existing file with the same
            name would be deleted).
        ``'a'``
            Append; an existing file is opened for reading and writing,
            and if the file does not exist it is created.
        ``'r+'``
            It is similar to ``'a'``, but the file must already exist.
    complevel : int, 0-9, default None
        Specifies a compression level for data.
        A value of 0 or None disables compression.
    complib : {'zlib', 'lzo', 'bzip2', 'blosc'}, default 'zlib'
        Specifies the compression library to be used.
        As of v0.20.2 these additional compressors for Blosc are supported
        (default if no compressor specified: 'blosc:blosclz'):
        {'blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc', 'blosc:snappy',
         'blosc:zlib', 'blosc:zstd'}.
        Specifying a compression library which is not available issues
        a ValueError.
    fletcher32 : bool, default False
        If applying compression use the fletcher32 checksum.
    **kwargs
        These parameters will be passed to the PyTables open_file method.

    Examples
    --------
    >>> bar = pd.DataFrame(np.random.randn(10, 4))
    >>> store = pd.HDFStore('test.h5')
    >>> store['foo'] = bar   # write to HDF5
    >>> bar = store['foo']   # retrieve
    >>> store.close()

    **Create or load HDF5 file in-memory**

    When passing the `driver` option to the PyTables open_file method through
    **kwargs, the HDF5 file is loaded or created in-memory and will only be
    written when closed:

    >>> bar = pd.DataFrame(np.random.randn(10, 4))
    >>> store = pd.HDFStore('test.h5', driver='H5FD_CORE')
    >>> store['foo'] = bar
    >>> store.close()   # only now, data is written to disk
    """

    _handle: File | None
    _mode: str
    _complevel: int
    _fletcher32: bool
    def __init__(
        self,
        path,
        mode: str = ...,
        complevel: int | None = ...,
        complib=...,
        fletcher32: bool = ...,
        **kwargs
    ) -> None: ...
    def __fspath__(self): ...
    @property
    def root(self):
        """return the root node"""
        ...
    @property
    def filename(self): ...
    def __getitem__(self, key: str): ...
    def __setitem__(self, key: str, value): ...
    def __delitem__(self, key: str): ...
    def __getattr__(self, name: str):
        """allow attribute access to get stores"""
        ...
    def __contains__(self, key: str) -> bool:
        """
        check for existence of this key
        can match the exact pathname or the pathnm w/o the leading '/'
        """
        ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...
    def keys(self, include: str = ...) -> list[str]:
        """
        Return a list of keys corresponding to objects stored in HDFStore.

        Parameters
        ----------

        include : str, default 'pandas'
                When kind equals 'pandas' return pandas objects.
                When kind equals 'native' return native HDF5 Table objects.

                .. versionadded:: 1.1.0

        Returns
        -------
        list
            List of ABSOLUTE path-names (e.g. have the leading '/').

        Raises
        ------
        raises ValueError if kind has an illegal value
        """
        ...
    def __iter__(self): ...
    def items(self):  # -> Generator[tuple[Unknown, Unknown], None, None]:
        """
        iterate on key->group
        """
        ...
    iteritems = ...
    def open(self, mode: str = ..., **kwargs):  # -> None:
        """
        Open the file in the specified mode

        Parameters
        ----------
        mode : {'a', 'w', 'r', 'r+'}, default 'a'
            See HDFStore docstring or tables.open_file for info about modes
        **kwargs
            These parameters will be passed to the PyTables open_file method.
        """
        ...
    def close(self):  # -> None:
        """
        Close the PyTables file handle
        """
        ...
    @property
    def is_open(self) -> bool:
        """
        return a boolean indicating whether the file is open
        """
        ...
    def flush(self, fsync: bool = ...):  # -> None:
        """
        Force all buffered modifications to be written to disk.

        Parameters
        ----------
        fsync : bool (default False)
          call ``os.fsync()`` on the file handle to force writing to disk.

        Notes
        -----
        Without ``fsync=True``, flushing may not guarantee that the OS writes
        to disk. With fsync, the operation will block until the OS claims the
        file has been written; however, other caching layers may still
        interfere.
        """
        ...
    def get(self, key: str):
        """
        Retrieve pandas object stored in file.

        Parameters
        ----------
        key : str

        Returns
        -------
        object
            Same type as object stored in file.
        """
        ...
    def select(
        self,
        key: str,
        where=...,
        start=...,
        stop=...,
        columns=...,
        iterator=...,
        chunksize=...,
        auto_close: bool = ...,
    ):  # -> TableIterator:
        """
        Retrieve pandas object stored in file, optionally based on where criteria.

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.

        Parameters
        ----------
        key : str
            Object being retrieved from file.
        where : list or None
            List of Term (or convertible) objects, optional.
        start : int or None
            Row number to start selection.
        stop : int, default None
            Row number to stop selection.
        columns : list or None
            A list of columns that if not None, will limit the return columns.
        iterator : bool or False
            Returns an iterator.
        chunksize : int or None
            Number or rows to include in iteration, return an iterator.
        auto_close : bool or False
            Should automatically close the store when finished.

        Returns
        -------
        object
            Retrieved object from file.
        """
        ...
    def select_as_coordinates(
        self, key: str, where=..., start: int | None = ..., stop: int | None = ...
    ):  # -> Index | Literal[False]:
        """
        return the selection as an Index

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.


        Parameters
        ----------
        key : str
        where : list of Term (or convertible) objects, optional
        start : integer (defaults to None), row number to start selection
        stop  : integer (defaults to None), row number to stop selection
        """
        ...
    def select_column(
        self, key: str, column: str, start: int | None = ..., stop: int | None = ...
    ):  # -> Series | Literal[False]:
        """
        return a single column from the table. This is generally only useful to
        select an indexable

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.

        Parameters
        ----------
        key : str
        column : str
            The column of interest.
        start : int or None, default None
        stop : int or None, default None

        Raises
        ------
        raises KeyError if the column is not found (or key is not a valid
            store)
        raises ValueError if the column can not be extracted individually (it
            is part of a data block)

        """
        ...
    def select_as_multiple(
        self,
        keys,
        where=...,
        selector=...,
        columns=...,
        start=...,
        stop=...,
        iterator=...,
        chunksize=...,
        auto_close: bool = ...,
    ):  # -> TableIterator:
        """
        Retrieve pandas objects from multiple tables.

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.

        Parameters
        ----------
        keys : a list of the tables
        selector : the table to apply the where criteria (defaults to keys[0]
            if not supplied)
        columns : the columns I want back
        start : integer (defaults to None), row number to start selection
        stop  : integer (defaults to None), row number to stop selection
        iterator : bool, return an iterator, default False
        chunksize : nrows to include in iteration, return an iterator
        auto_close : bool, default False
            Should automatically close the store when finished.

        Raises
        ------
        raises KeyError if keys or selector is not found or keys is empty
        raises TypeError if keys is not a list or tuple
        raises ValueError if the tables are not ALL THE SAME DIMENSIONS
        """
        ...
    def put(
        self,
        key: str,
        value: FrameOrSeries,
        format=...,
        index=...,
        append=...,
        complib=...,
        complevel: int | None = ...,
        min_itemsize: int | dict[str, int] | None = ...,
        nan_rep=...,
        data_columns: list[str] | None = ...,
        encoding=...,
        errors: str = ...,
        track_times: bool = ...,
        dropna: bool = ...,
    ):  # -> None:
        """
        Store object in HDFStore.

        Parameters
        ----------
        key : str
        value : {Series, DataFrame}
        format : 'fixed(f)|table(t)', default is 'fixed'
            Format to use when storing object in HDFStore. Value can be one of:

            ``'fixed'``
                Fixed format.  Fast writing/reading. Not-appendable, nor searchable.
            ``'table'``
                Table format.  Write as a PyTables Table structure which may perform
                worse but allow more flexible operations like searching / selecting
                subsets of the data.
        append : bool, default False
            This will force Table format, append the input data to the existing.
        data_columns : list, default None
            List of columns to create as data columns, or True to use all columns.
            See `here
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#query-via-data-columns>`__.
        encoding : str, default None
            Provide an encoding for strings.
        track_times : bool, default True
            Parameter is propagated to 'create_table' method of 'PyTables'.
            If set to False it enables to have the same h5 files (same hashes)
            independent on creation time.

            .. versionadded:: 1.1.0
        """
        ...
    def remove(self, key: str, where=..., start=..., stop=...):  # -> None:
        """
        Remove pandas object partially by specifying the where condition

        Parameters
        ----------
        key : str
            Node to remove or delete rows from
        where : list of Term (or convertible) objects, optional
        start : integer (defaults to None), row number to start selection
        stop  : integer (defaults to None), row number to stop selection

        Returns
        -------
        number of rows removed (or None if not a Table)

        Raises
        ------
        raises KeyError if key is not a valid store

        """
        ...
    def append(
        self,
        key: str,
        value: FrameOrSeries,
        format=...,
        axes=...,
        index=...,
        append=...,
        complib=...,
        complevel: int | None = ...,
        columns=...,
        min_itemsize: int | dict[str, int] | None = ...,
        nan_rep=...,
        chunksize=...,
        expectedrows=...,
        dropna: bool | None = ...,
        data_columns: list[str] | None = ...,
        encoding=...,
        errors: str = ...,
    ):  # -> None:
        """
        Append to Table in file. Node must already exist and be Table
        format.

        Parameters
        ----------
        key : str
        value : {Series, DataFrame}
        format : 'table' is the default
            Format to use when storing object in HDFStore.  Value can be one of:

            ``'table'``
                Table format. Write as a PyTables Table structure which may perform
                worse but allow more flexible operations like searching / selecting
                subsets of the data.
        append       : bool, default True
            Append the input data to the existing.
        data_columns : list of columns, or True, default None
            List of columns to create as indexed data columns for on-disk
            queries, or True to use all columns. By default only the axes
            of the object are indexed. See `here
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#query-via-data-columns>`__.
        min_itemsize : dict of columns that specify minimum str sizes
        nan_rep      : str to use as str nan representation
        chunksize    : size to chunk the writing
        expectedrows : expected TOTAL row size of this table
        encoding     : default None, provide an encoding for str
        dropna : bool, default False
            Do not write an ALL nan row to the store settable
            by the option 'io.hdf.dropna_table'.

        Notes
        -----
        Does *not* check if data being appended overlaps with existing
        data in the table, so be careful
        """
        ...
    def append_to_multiple(
        self, d: dict, value, selector, data_columns=..., axes=..., dropna=..., **kwargs
    ):  # -> None:
        """
        Append to multiple tables

        Parameters
        ----------
        d : a dict of table_name to table_columns, None is acceptable as the
            values of one node (this will get all the remaining columns)
        value : a pandas object
        selector : a string that designates the indexable table; all of its
            columns will be designed as data_columns, unless data_columns is
            passed, in which case these are used
        data_columns : list of columns to create as data columns, or True to
            use all columns
        dropna : if evaluates to True, drop rows from all tables if any single
                 row in each table has all NaN. Default False.

        Notes
        -----
        axes parameter is currently not accepted

        """
        ...
    def create_table_index(
        self, key: str, columns=..., optlevel: int | None = ..., kind: str | None = ...
    ):  # -> None:
        """
        Create a pytables index on the table.

        Parameters
        ----------
        key : str
        columns : None, bool, or listlike[str]
            Indicate which columns to create an index on.

            * False : Do not create any indexes.
            * True : Create indexes on all columns.
            * None : Create indexes on all columns.
            * listlike : Create indexes on the given columns.

        optlevel : int or None, default None
            Optimization level, if None, pytables defaults to 6.
        kind : str or None, default None
            Kind of index, if None, pytables defaults to "medium".

        Raises
        ------
        TypeError: raises if the node is not a table
        """
        ...
    def groups(self):  # -> list[Unknown]:
        """
        Return a list of all the top-level nodes.

        Each node returned is not a pandas storage object.

        Returns
        -------
        list
            List of objects.
        """
        ...
    def walk(self, where=...):
        """
        Walk the pytables group hierarchy for pandas objects.

        This generator will yield the group path, subgroups and pandas object
        names for each group.

        Any non-pandas PyTables objects that are not a group will be ignored.

        The `where` group itself is listed first (preorder), then each of its
        child groups (following an alphanumerical order) is also traversed,
        following the same procedure.

        Parameters
        ----------
        where : str, default "/"
            Group where to start walking.

        Yields
        ------
        path : str
            Full path to a group (without trailing '/').
        groups : list
            Names (strings) of the groups contained in `path`.
        leaves : list
            Names (strings) of the pandas objects contained in `path`.
        """
        ...
    def get_node(self, key: str) -> Node | None:
        """return the node with the key or None if it does not exist"""
        ...
    def get_storer(self, key: str) -> GenericFixed | Table:
        """return the storer object for a key, raise if not in the file"""
        ...
    def copy(
        self,
        file,
        mode=...,
        propindexes: bool = ...,
        keys=...,
        complib=...,
        complevel: int | None = ...,
        fletcher32: bool = ...,
        overwrite=...,
    ):  # -> HDFStore:
        """
        Copy the existing store to a new file, updating in place.

        Parameters
        ----------
        propindexes : bool, default True
            Restore indexes in copied file.
        keys : list, optional
            List of keys to include in the copy (defaults to all).
        overwrite : bool, default True
            Whether to overwrite (remove and replace) existing nodes in the new store.
        mode, complib, complevel, fletcher32 same as in HDFStore.__init__

        Returns
        -------
        open file handle of the new store
        """
        ...
    def info(self) -> str:
        """
        Print detailed information on the store.

        Returns
        -------
        str
        """
        ...

class TableIterator:
    """
    Define the iteration interface on a table

    Parameters
    ----------
    store : HDFStore
    s     : the referred storer
    func  : the function to execute the query
    where : the where of the query
    nrows : the rows to iterate on
    start : the passed start value (default is None)
    stop  : the passed stop value (default is None)
    iterator : bool, default False
        Whether to use the default iterator.
    chunksize : the passed chunking value (default is 100000)
    auto_close : bool, default False
        Whether to automatically close the store at the end of iteration.
    """

    chunksize: int | None
    store: HDFStore
    s: GenericFixed | Table
    def __init__(
        self,
        store: HDFStore,
        s: GenericFixed | Table,
        func,
        where,
        nrows,
        start=...,
        stop=...,
        iterator: bool = ...,
        chunksize: int | None = ...,
        auto_close: bool = ...,
    ) -> None: ...
    def __iter__(self): ...
    def close(self): ...
    def get_result(self, coordinates: bool = ...): ...

class IndexCol:
    """
    an index column description class

    Parameters
    ----------
    axis   : axis which I reference
    values : the ndarray like converted values
    kind   : a string description of this type
    typ    : the pytables type
    pos    : the position in the pytables

    """

    is_an_indexable = ...
    is_data_indexable = ...
    _info_fields = ...
    name: str
    cname: str
    def __init__(
        self,
        name: str,
        values=...,
        kind=...,
        typ=...,
        cname: str | None = ...,
        axis=...,
        pos=...,
        freq=...,
        tz=...,
        index_name=...,
        ordered=...,
        table=...,
        meta=...,
        metadata=...,
    ) -> None: ...
    @property
    def itemsize(self) -> int: ...
    @property
    def kind_attr(self) -> str: ...
    def set_pos(self, pos: int):  # -> None:
        """set the position of this column in the Table"""
        ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: Any) -> bool:
        """compare 2 col items"""
        ...
    def __ne__(self, other) -> bool: ...
    @property
    def is_indexed(self) -> bool:
        """return whether I am an indexed column"""
        ...
    def convert(
        self, values: np.ndarray, nan_rep, encoding: str, errors: str
    ):  # -> tuple[ndarray | DatetimeIndex, ndarray | DatetimeIndex]:
        """
        Convert the data from this selection to the appropriate pandas type.
        """
        ...
    def take_data(self):
        """return the values"""
        ...
    @property
    def attrs(self): ...
    @property
    def description(self): ...
    @property
    def col(self):  # -> Any | None:
        """return my current col description"""
        ...
    @property
    def cvalues(self):
        """return my cython values"""
        ...
    def __iter__(self): ...
    def maybe_set_size(self, min_itemsize=...):  # -> None:
        """
        maybe set a string col itemsize:
            min_itemsize can be an integer or a dict with this columns name
            with an integer size
        """
        ...
    def validate_names(self): ...
    def validate_and_set(self, handler: AppendableTable, append: bool): ...
    def validate_col(self, itemsize=...):  # -> Any | None:
        """validate this column: return the compared against itemsize"""
        ...
    def validate_attr(self, append: bool): ...
    def update_info(self, info):
        """
        set/update the info for this indexable with the key/value
        if there is a conflict raise/warn as needed
        """
        ...
    def set_info(self, info):  # -> None:
        """set my state from the passed info"""
        ...
    def set_attr(self):  # -> None:
        """set the kind for this column"""
        ...
    def validate_metadata(self, handler: AppendableTable):  # -> None:
        """validate that kind=category does not change the categories"""
        ...
    def write_metadata(self, handler: AppendableTable):  # -> None:
        """set the meta data"""
        ...

class GenericIndexCol(IndexCol):
    """an index which is not represented in the data of the table"""

    @property
    def is_indexed(self) -> bool: ...
    def convert(
        self, values: np.ndarray, nan_rep, encoding: str, errors: str
    ):  # -> tuple[ndarray, ndarray]:
        """
        Convert the data from this selection to the appropriate pandas type.

        Parameters
        ----------
        values : np.ndarray
        nan_rep : str
        encoding : str
        errors : str
        """
        ...
    def set_attr(self): ...

class DataCol(IndexCol):
    """
    a data holding column, by definition this is not indexable

    Parameters
    ----------
    data   : the actual data
    cname  : the column name in the table to hold the data (typically
                values)
    meta   : a string description of the metadata
    metadata : the actual metadata
    """

    is_an_indexable = ...
    is_data_indexable = ...
    _info_fields = ...
    def __init__(
        self,
        name: str,
        values=...,
        kind=...,
        typ=...,
        cname=...,
        pos=...,
        tz=...,
        ordered=...,
        table=...,
        meta=...,
        metadata=...,
        dtype: DtypeArg | None = ...,
        data=...,
    ) -> None: ...
    @property
    def dtype_attr(self) -> str: ...
    @property
    def meta_attr(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: Any) -> bool:
        """compare 2 col items"""
        ...
    def set_data(self, data: ArrayLike): ...
    def take_data(self):  # -> ndarray:
        """return the data"""
        ...
    @classmethod
    def get_atom_string(cls, shape, itemsize): ...
    @classmethod
    def get_atom_coltype(cls, kind: str) -> type[Col]:
        """return the PyTables column class for this column"""
        ...
    @classmethod
    def get_atom_data(cls, shape, kind: str) -> Col: ...
    @classmethod
    def get_atom_datetime64(cls, shape): ...
    @classmethod
    def get_atom_timedelta64(cls, shape): ...
    @property
    def shape(self): ...
    @property
    def cvalues(self):  # -> ndarray:
        """return my cython values"""
        ...
    def validate_attr(self, append):  # -> None:
        """validate that we have the same order as the existing & same dtype"""
        ...
    def convert(
        self, values: np.ndarray, nan_rep, encoding: str, errors: str
    ):  # -> tuple[Unknown, ndarray | DatetimeIndex | Categorical]:
        """
        Convert the data from this selection to the appropriate pandas type.

        Parameters
        ----------
        values : np.ndarray
        nan_rep :
        encoding : str
        errors : str

        Returns
        -------
        index : listlike to become an Index
        data : ndarraylike to become a column
        """
        ...
    def set_attr(self):  # -> None:
        """set the data for this column"""
        ...

class DataIndexableCol(DataCol):
    """represent a data column that can be indexed"""

    is_data_indexable = ...
    def validate_names(self): ...
    @classmethod
    def get_atom_string(cls, shape, itemsize): ...
    @classmethod
    def get_atom_data(cls, shape, kind: str) -> Col: ...
    @classmethod
    def get_atom_datetime64(cls, shape): ...
    @classmethod
    def get_atom_timedelta64(cls, shape): ...

class GenericDataIndexableCol(DataIndexableCol):
    """represent a generic pytables data column"""

    ...

class Fixed:
    """
    represent an object in my store
    facilitate read/write of various types of objects
    this is an abstract base class

    Parameters
    ----------
    parent : HDFStore
    group : Node
        The group node where the table resides.
    """

    pandas_kind: str
    format_type: str = ...
    obj_type: type[FrameOrSeriesUnion]
    ndim: int
    encoding: str
    parent: HDFStore
    group: Node
    errors: str
    is_table = ...
    def __init__(
        self, parent: HDFStore, group: Node, encoding: str = ..., errors: str = ...
    ) -> None: ...
    @property
    def is_old_version(self) -> bool: ...
    @property
    def version(self) -> tuple[int, int, int]:
        """compute and set our version"""
        ...
    @property
    def pandas_type(self): ...
    def __repr__(self) -> str:
        """return a pretty representation of myself"""
        ...
    def set_object_info(self):  # -> None:
        """set my pandas type & version"""
        ...
    def copy(self): ...
    @property
    def shape(self): ...
    @property
    def pathname(self): ...
    @property
    def attrs(self): ...
    def set_attrs(self):  # -> None:
        """set our object attributes"""
        ...
    def get_attrs(self):  # -> None:
        """get our object attributes"""
        ...
    @property
    def storable(self):
        """return my storable"""
        ...
    @property
    def is_exists(self) -> bool: ...
    @property
    def nrows(self): ...
    def validate(self, other):  # -> Literal[True] | None:
        """validate against an existing storable"""
        ...
    def validate_version(self, where=...):  # -> Literal[True]:
        """are we trying to operate on an old version?"""
        ...
    def infer_axes(self):  # -> bool:
        """
        infer the axes of my storer
        return a boolean indicating if we have a valid storer or not
        """
        ...
    def read(
        self, where=..., columns=..., start: int | None = ..., stop: int | None = ...
    ): ...
    def write(self, **kwargs): ...
    def delete(
        self, where=..., start: int | None = ..., stop: int | None = ...
    ):  # -> None:
        """
        support fully deleting the node in its entirety (only) - where
        specification must be None
        """
        ...

class GenericFixed(Fixed):
    """a generified fixed version"""

    _index_type_map = ...
    _reverse_index_map = ...
    attributes: list[str] = ...
    def validate_read(self, columns, where):  # -> None:
        """
        raise if any keywords are passed which are not-None
        """
        ...
    @property
    def is_exists(self) -> bool: ...
    def set_attrs(self):  # -> None:
        """set our object attributes"""
        ...
    def get_attrs(self):  # -> None:
        """retrieve our attributes"""
        ...
    def write(self, obj, **kwargs): ...
    def read_array(
        self, key: str, start: int | None = ..., stop: int | None = ...
    ):  # -> Any | ndarray | DatetimeIndex:
        """read an array for the specified node (off of group"""
        ...
    def read_index(
        self, key: str, start: int | None = ..., stop: int | None = ...
    ) -> Index: ...
    def write_index(self, key: str, index: Index): ...
    def write_multi_index(self, key: str, index: MultiIndex): ...
    def read_multi_index(
        self, key: str, start: int | None = ..., stop: int | None = ...
    ) -> MultiIndex: ...
    def read_index_node(
        self, node: Node, start: int | None = ..., stop: int | None = ...
    ) -> Index: ...
    def write_array_empty(self, key: str, value: ArrayLike):  # -> None:
        """write a 0-len array"""
        ...
    def write_array(self, key: str, obj: FrameOrSeries, items: Index | None = ...): ...

class SeriesFixed(GenericFixed):
    pandas_kind = ...
    attributes = ...
    name: Hashable
    @property
    def shape(self): ...
    def read(
        self, where=..., columns=..., start: int | None = ..., stop: int | None = ...
    ): ...
    def write(self, obj, **kwargs): ...

class BlockManagerFixed(GenericFixed):
    attributes = ...
    nblocks: int
    @property
    def shape(self) -> Shape | None: ...
    def read(
        self, where=..., columns=..., start: int | None = ..., stop: int | None = ...
    ): ...
    def write(self, obj, **kwargs): ...

class FrameFixed(BlockManagerFixed):
    pandas_kind = ...
    obj_type = DataFrame

class Table(Fixed):
    """
    represent a table:
        facilitate read/write of various types of tables

    Attrs in Table Node
    -------------------
    These are attributes that are store in the main table node, they are
    necessary to recreate these tables when read back in.

    index_axes    : a list of tuples of the (original indexing axis and
        index column)
    non_index_axes: a list of tuples of the (original index axis and
        columns on a non-indexing axis)
    values_axes   : a list of the columns which comprise the data of this
        table
    data_columns  : a list of the columns that we are allowing indexing
        (these become single columns in values_axes), or True to force all
        columns
    nan_rep       : the string to use for nan representations for string
        objects
    levels        : the names of levels
    metadata      : the names of the metadata columns
    """

    pandas_kind = ...
    format_type: str = ...
    table_type: str
    levels: int | list[Hashable] = ...
    is_table = ...
    index_axes: list[IndexCol]
    non_index_axes: list[tuple[int, Any]]
    values_axes: list[DataCol]
    data_columns: list
    metadata: list
    info: dict
    def __init__(
        self,
        parent: HDFStore,
        group: Node,
        encoding=...,
        errors: str = ...,
        index_axes=...,
        non_index_axes=...,
        values_axes=...,
        data_columns=...,
        info=...,
        nan_rep=...,
    ) -> None: ...
    @property
    def table_type_short(self) -> str: ...
    def __repr__(self) -> str:
        """return a pretty representation of myself"""
        ...
    def __getitem__(self, c: str):  # -> IndexCol | None:
        """return the axis for c"""
        ...
    def validate(self, other):  # -> None:
        """validate against an existing table"""
        ...
    @property
    def is_multi_index(self) -> bool:
        """the levels attribute is 1 or a list in the case of a multi-index"""
        ...
    def validate_multiindex(
        self, obj: FrameOrSeriesUnion
    ) -> tuple[DataFrame, list[Hashable]]:
        """
        validate that we can store the multi-index; reset and return the
        new object
        """
        ...
    @property
    def nrows_expected(self) -> int:
        """based on our axes, compute the expected nrows"""
        ...
    @property
    def is_exists(self) -> bool:
        """has this table been created"""
        ...
    @property
    def storable(self): ...
    @property
    def table(self):  # -> Any | None:
        """return the table group (this is my storable)"""
        ...
    @property
    def dtype(self): ...
    @property
    def description(self): ...
    @property
    def axes(self): ...
    @property
    def ncols(self) -> int:
        """the number of total columns in the values axes"""
        ...
    @property
    def is_transposed(self) -> bool: ...
    @property
    def data_orientation(self):  # -> tuple[int, ...]:
        """return a tuple of my permutated axes, non_indexable at the front"""
        ...
    def queryables(self) -> dict[str, Any]:
        """return a dict of the kinds allowable columns for this object"""
        ...
    def index_cols(self):  # -> list[tuple[Unknown, str]]:
        """return a list of my index cols"""
        ...
    def values_cols(self) -> list[str]:
        """return a list of my values cols"""
        ...
    def write_metadata(self, key: str, values: np.ndarray):  # -> None:
        """
        Write out a metadata array to the key as a fixed-format Series.

        Parameters
        ----------
        key : str
        values : ndarray
        """
        ...
    def read_metadata(self, key: str):  # -> TableIterator | None:
        """return the meta data array for this key"""
        ...
    def set_attrs(self):  # -> None:
        """set our table type & indexables"""
        ...
    def get_attrs(self):  # -> None:
        """retrieve our attributes"""
        ...
    def validate_version(self, where=...):  # -> None:
        """are we trying to operate on an old version?"""
        ...
    def validate_min_itemsize(self, min_itemsize):  # -> None:
        """
        validate the min_itemsize doesn't contain items that are not in the
        axes this needs data_columns to be defined
        """
        ...
    @cache_readonly
    def indexables(self):  # -> list[Unknown]:
        """create/cache the indexables if they don't exist"""
        ...
    def create_index(self, columns=..., optlevel=..., kind: str | None = ...):  # -> None:
        """
        Create a pytables index on the specified columns.

        Parameters
        ----------
        columns : None, bool, or listlike[str]
            Indicate which columns to create an index on.

            * False : Do not create any indexes.
            * True : Create indexes on all columns.
            * None : Create indexes on all columns.
            * listlike : Create indexes on the given columns.

        optlevel : int or None, default None
            Optimization level, if None, pytables defaults to 6.
        kind : str or None, default None
            Kind of index, if None, pytables defaults to "medium".

        Raises
        ------
        TypeError if trying to create an index on a complex-type column.

        Notes
        -----
        Cannot index Time64Col or ComplexCol.
        Pytables must be >= 3.0.
        """
        ...
    @classmethod
    def get_object(cls, obj, transposed: bool):
        """return the data for this obj"""
        ...
    def validate_data_columns(self, data_columns, min_itemsize, non_index_axes):
        """
        take the input data_columns and min_itemize and create a data
        columns spec
        """
        ...
    def process_axes(self, obj, selection: Selection, columns=...):  # -> DataFrame:
        """process axes filters"""
        ...
    def create_description(
        self, complib, complevel: int | None, fletcher32: bool, expectedrows: int | None
    ) -> dict[str, Any]:
        """create the description of the table from the axes & values"""
        ...
    def read_coordinates(
        self, where=..., start: int | None = ..., stop: int | None = ...
    ):  # -> Index | Literal[False]:
        """
        select coordinates (row numbers) from a table; return the
        coordinates object
        """
        ...
    def read_column(
        self, column: str, where=..., start: int | None = ..., stop: int | None = ...
    ):  # -> Series | Literal[False]:
        """
        return a single column from the table, generally only indexables
        are interesting
        """
        ...

class WORMTable(Table):
    """
    a write-once read-many table: this format DOES NOT ALLOW appending to a
    table. writing is a one-time operation the data are stored in a format
    that allows for searching the data on disk
    """

    table_type = ...
    def read(
        self, where=..., columns=..., start: int | None = ..., stop: int | None = ...
    ):
        """
        read the indices and the indexing array, calculate offset rows and return
        """
        ...
    def write(self, **kwargs):
        """
        write in a format that we can search later on (but cannot append
        to): write out the indices and the values using _write_array
        (e.g. a CArray) create an indexing table so that we can search
        """
        ...

class AppendableTable(Table):
    """support the new appendable table formats"""

    table_type = ...
    def write(
        self,
        obj,
        axes=...,
        append=...,
        complib=...,
        complevel=...,
        fletcher32=...,
        min_itemsize=...,
        chunksize=...,
        expectedrows=...,
        dropna=...,
        nan_rep=...,
        data_columns=...,
        track_times=...,
    ): ...
    def write_data(self, chunksize: int | None, dropna: bool = ...):  # -> None:
        """
        we form the data into a 2-d including indexes,values,mask write chunk-by-chunk
        """
        ...
    def write_data_chunk(
        self,
        rows: np.ndarray,
        indexes: list[np.ndarray],
        mask: np.ndarray | None,
        values: list[np.ndarray],
    ):  # -> None:
        """
        Parameters
        ----------
        rows : an empty memory space where we are putting the chunk
        indexes : an array of the indexes
        mask : an array of the masks
        values : an array of the values
        """
        ...
    def delete(self, where=..., start: int | None = ..., stop: int | None = ...): ...

class AppendableFrameTable(AppendableTable):
    """support the new appendable table formats"""

    pandas_kind = ...
    table_type = ...
    ndim = ...
    obj_type: type[FrameOrSeriesUnion] = ...
    @property
    def is_transposed(self) -> bool: ...
    @classmethod
    def get_object(cls, obj, transposed: bool):
        """these are written transposed"""
        ...
    def read(
        self, where=..., columns=..., start: int | None = ..., stop: int | None = ...
    ): ...

class AppendableSeriesTable(AppendableFrameTable):
    """support the new appendable table formats"""

    pandas_kind = ...
    table_type = ...
    ndim = ...
    obj_type = Series
    @property
    def is_transposed(self) -> bool: ...
    @classmethod
    def get_object(cls, obj, transposed: bool): ...
    def write(self, obj, data_columns=..., **kwargs):  # -> None:
        """we are going to write this as a frame table"""
        ...
    def read(
        self, where=..., columns=..., start: int | None = ..., stop: int | None = ...
    ) -> Series: ...

class AppendableMultiSeriesTable(AppendableSeriesTable):
    """support the new appendable table formats"""

    pandas_kind = ...
    table_type = ...
    def write(self, obj, **kwargs):  # -> None:
        """we are going to write this as a frame table"""
        ...

class GenericTable(AppendableFrameTable):
    """a table that read/writes the generic pytables table format"""

    pandas_kind = ...
    table_type = ...
    ndim = ...
    obj_type = DataFrame
    levels: list[Hashable]
    @property
    def pandas_type(self) -> str: ...
    @property
    def storable(self): ...
    def get_attrs(self):  # -> None:
        """retrieve our attributes"""
        ...
    @cache_readonly
    def indexables(self):  # -> list[GenericIndexCol | GenericDataIndexableCol]:
        """create the indexables from the table description"""
        ...
    def write(self, **kwargs): ...

class AppendableMultiFrameTable(AppendableFrameTable):
    """a frame with a multi-index"""

    table_type = ...
    obj_type = DataFrame
    ndim = ...
    _re_levels = ...
    @property
    def table_type_short(self) -> str: ...
    def write(self, obj, data_columns=..., **kwargs): ...
    def read(
        self, where=..., columns=..., start: int | None = ..., stop: int | None = ...
    ): ...

class Selection:
    """
    Carries out a selection operation on a tables.Table object.

    Parameters
    ----------
    table : a Table object
    where : list of Terms (or convertible to)
    start, stop: indices to start and/or stop selection

    """

    def __init__(
        self, table: Table, where=..., start: int | None = ..., stop: int | None = ...
    ) -> None: ...
    def generate(self, where):  # -> PyTablesExpr | None:
        """where can be a : dict,list,tuple,string"""
        ...
    def select(self):  # -> Any:
        """
        generate the selection
        """
        ...
    def select_coords(self):  # -> Any | ndarray:
        """
        generate the selection
        """
        ...
